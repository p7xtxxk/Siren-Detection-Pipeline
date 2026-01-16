import time
import queue
from collections import deque

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

# ====================== CONFIG ======================
MODEL_SR = 22050

WIN_SEC = 1.2
HOP_SEC = 0.15

WIN = int(MODEL_SR * WIN_SEC)
HOP = int(MODEL_SR * HOP_SEC)

N_MELS = 64
EXPECTED_FRAMES = 130

PLOT_THRESH = 0.2
MODEL_PATH = "siren_model.tflite"

# Detection logic
EMA_ALPHA = 0.8
SCORE_DECAY = 0.95
SCORE_THRESH = 1.2
ALERT_COOLDOWN = 3.0

# Audio handling
FIXED_GAIN = 20.0
ENERGY_SCALE = 40.0

# Band-energy gate (Hz)
BAND_LOW = 700
BAND_HIGH = 1800
BAND_THRESH = 0.015

# ====================== MIC DEVICE ======================
MIC_DEVICE_INDEX = 18  # Microphone Array (Realtek HD Audio Mic input), Windows WDM-KS (2 in, 0 out)
device_info = sd.query_devices(MIC_DEVICE_INDEX, "input")
MIC_SR = int(device_info["default_samplerate"])
print(f"🎤 Using mic: {device_info['name']} @ {MIC_SR} Hz")

# ====================== GLOBALS ======================
audio_q = queue.Queue()
buffer = np.zeros(0, dtype=np.float32)

times = deque(maxlen=6000)
energy_vals = deque(maxlen=6000)
prob_vals = deque(maxlen=6000)

ema_prob = 0.0
score = 0.0
last_alert_time = 0.0
START_TIME = time.time()

chunks_received = 0  # debug counter

# ====================== MODEL ======================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()
out_det = interpreter.get_output_details()
print("✅ TFLite model loaded")

# ====================== FEATURE EXTRACTION ======================
def extract_logmel(x):
    # x = x / (np.max(np.abs(x)) + 1e-6)
    x = librosa.util.normalize(x)

    S = librosa.feature.melspectrogram(
        y=x,
        sr=MODEL_SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS,
        power=2.0
    )
    logS = librosa.power_to_db(S, ref=np.max)

    if logS.shape[1] < EXPECTED_FRAMES:
        logS = np.pad(logS, ((0, 0), (0, EXPECTED_FRAMES - logS.shape[1])), mode="constant")
    else:
        logS = logS[:, :EXPECTED_FRAMES]

    return logS[np.newaxis, ..., np.newaxis].astype(np.float32)

# ====================== INFERENCE ======================
def infer(segment):
    global ema_prob

    # Energy (visualization only)
    energy = np.sqrt(np.mean(segment ** 2))
    energy_norm = np.clip(energy * ENERGY_SCALE, 0.0, 1.0)

    # Band-energy gate
    spec = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1 / MODEL_SR)
    band_energy = np.mean(spec[(freqs > BAND_LOW) & (freqs < BAND_HIGH)])

    if band_energy < BAND_THRESH:
        return energy_norm, ema_prob

    # Model inference
    X = extract_logmel(segment)
    interpreter.set_tensor(in_det[0]["index"], X)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(out_det[0]["index"])[0][0])

    # EMA smoothing
    ema_prob = EMA_ALPHA * ema_prob + (1 - EMA_ALPHA) * prob
    return energy_norm, ema_prob

# ====================== AUDIO CALLBACK ======================
def audio_callback(indata, frames, time_info, status):
    global chunks_received
    if status:
        print("Audio status:", status)
    mono = indata[:, 0].astype(np.float32)
    mono -= np.mean(mono)
    mono *= FIXED_GAIN
    mono = np.clip(mono, -1.0, 1.0)

    if MIC_SR != MODEL_SR:
        # Fast resampling: upsample/downsample using polyphase
        # ratio = MODEL_SR / MIC_SR
        # resample_poly expects integers; compute gcd
        from math import gcd
        g = gcd(int(MIC_SR), int(MODEL_SR))
        up = int(MODEL_SR // g)
        down = int(MIC_SR // g)
        mono = resample_poly(mono, up, down).astype(np.float32)

    audio_q.put(mono)
    chunks_received += 1

# ====================== AUDIO STREAM ======================
stream = sd.InputStream(
    samplerate=MIC_SR,
    channels=1,
    device=MIC_DEVICE_INDEX,
    blocksize=4096,
    dtype="float32",
    callback=audio_callback
)
stream.start()
print("🎤 Microphone started")

# ====================== MATPLOTLIB BACKEND ======================
# Prefer interactive backend; fallback to default if unavailable
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

energy_line, = ax1.plot([], [], color="blue", label="Audio Energy")
ax1.set_ylim(0, 1)
ax1.set_ylabel("Energy")
ax1.legend(loc="upper left")

prob_line, = ax2.plot([], [], color="green", label="Siren Probability")
ax2.axhline(PLOT_THRESH, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.2")
ax2.set_ylim(0, 1)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Time (s)")
ax2.legend(loc="upper left")

fig.suptitle("High-Confidence Real-Time Siren Detection")

# ====================== MAIN LOOP (manual, robust) ======================
print("⏱️ Entering update loop...")
last_debug = time.time()

try:
    while True:
        # Drain queue quickly
        drained = False
        while not audio_q.empty():
            drained = True
            buffer = np.concatenate([buffer, audio_q.get()])

        # Process windows
        while len(buffer) >= WIN:
            segment = buffer[:WIN]
            buffer = buffer[HOP:]
            energy, prob = infer(segment)

            # Score accumulation
            score = score * SCORE_DECAY + prob
            now = time.time()
            if score > SCORE_THRESH and now - last_alert_time > ALERT_COOLDOWN:
                print(f"🚨 SIREN CONFIRMED | score={score:.2f}")
                last_alert_time = now
                score = 0.0

            t = now - START_TIME
            times.append(t)
            energy_vals.append(energy)
            prob_vals.append(prob)

        # Update plot if we have data
        if len(times) > 0:
            energy_line.set_data(times, energy_vals)
            prob_line.set_data(times, prob_vals)
            ax2.set_xlim(0, max(10, times[-1] + 0.5))
            plt.pause(0.05)  # drive GUI event loop

        # Debug heartbeat every 2s
        if time.time() - last_debug > 2.0:
            print(f"Chunks received: {chunks_received} | Queue size: {audio_q.qsize()} | Points: {len(times)}")
            last_debug = time.time()

        # Small sleep to avoid busy loop
        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    print("🛑 Stopped")
