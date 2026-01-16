import time
import queue

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ====================== CONFIG ======================
MODEL_SR = 22050

WIN_SEC = 1.2        # <<< LONGER WINDOW (CRITICAL)
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
SCORE_THRESH = 1.2      # <<< integrated confidence
ALERT_COOLDOWN = 3.0

# Audio handling
FIXED_GAIN = 20.0
ENERGY_SCALE = 40.0

# Band-energy gate (Hz)
BAND_LOW = 700
BAND_HIGH = 1800
BAND_THRESH = 0.015

# ====================== MIC DEVICE ======================
MIC_DEVICE_INDEX = 18   # check for -> Microphone Array (Realtek(R) Audio), Windows WASAPI (2 in, 0 out)

device_info = sd.query_devices(MIC_DEVICE_INDEX, "input")
MIC_SR = int(device_info["default_samplerate"])
print(f"🎤 Using mic: {device_info['name']} @ {MIC_SR} Hz")

# ====================== GLOBALS ======================
audio_q = queue.Queue()
buffer = np.zeros(0, dtype=np.float32)

times = []
energy_vals = []
prob_vals = []

ema_prob = 0.0
score = 0.0
last_alert_time = 0.0
START_TIME = time.time()

# ====================== MODEL ======================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("TFLite model loaded")

# ====================== FEATURE EXTRACTION ======================
def extract_logmel(x):
    x = x / (np.max(np.abs(x)) + 1e-6)   # <<< BETTER THAN librosa.normalize

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
        logS = np.pad(
            logS,
            ((0, 0), (0, EXPECTED_FRAMES - logS.shape[1])),
            mode="constant"
        )
    else:
        logS = logS[:, :EXPECTED_FRAMES]

    return logS[np.newaxis, ..., np.newaxis].astype(np.float32)

# ====================== INFERENCE ======================
def infer(segment):
    global ema_prob

    # --- Energy (visualization only)
    energy = np.sqrt(np.mean(segment ** 2))
    energy_norm = np.clip(energy * ENERGY_SCALE, 0.0, 1.0)

    # --- Band-energy gate (FAST, EFFECTIVE)
    spec = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1 / MODEL_SR)
    band_energy = np.mean(spec[(freqs > BAND_LOW) & (freqs < BAND_HIGH)])

    if band_energy < BAND_THRESH:
        return energy_norm, ema_prob

    # --- Model inference
    X = extract_logmel(segment)
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    interpreter.set_tensor(in_det[0]["index"], X)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(out_det[0]["index"])[0][0])

    # EMA smoothing
    ema_prob = EMA_ALPHA * ema_prob + (1 - EMA_ALPHA) * prob
    return energy_norm, ema_prob

# ====================== AUDIO CALLBACK ======================
def audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0].astype(np.float32)
    mono -= np.mean(mono)

    mono *= FIXED_GAIN
    mono = np.clip(mono, -1.0, 1.0)

    if MIC_SR != MODEL_SR:
        mono = librosa.resample(mono, orig_sr=MIC_SR, target_sr=MODEL_SR)

    audio_q.put(mono)

# ====================== AUDIO STREAM ======================
stream = sd.InputStream(
    samplerate=MIC_SR,
    channels=1,
    device=MIC_DEVICE_INDEX,
    blocksize=4096,
    callback=audio_callback
)
stream.start()
print("🎤 Microphone started")

# ====================== PLOTS ======================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

energy_line, = ax1.plot([], [], color="blue", label="Audio Energy")
ax1.set_ylim(0, 1)
ax1.set_ylabel("Energy")
ax1.legend()

prob_line, = ax2.plot([], [], color="green", label="Siren Probability")
ax2.axhline(PLOT_THRESH, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.2")
ax2.set_ylim(0, 1)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Time (s)")
ax2.legend()

fig.suptitle("High-Confidence Real-Time Siren Detection")

# ====================== UPDATE LOOP ======================
def update(frame):
    global buffer, score, last_alert_time

    while not audio_q.empty():
        buffer = np.concatenate([buffer, audio_q.get()])

        t = time.time() - START_TIME

        if len(buffer) >= WIN:
            segment = buffer[:WIN]
            buffer = buffer[HOP:]
            energy, prob = infer(segment)
        else:
            energy, prob = 0.0, ema_prob

        # --- Score accumulation (KEY UPGRADE)
        score = score * SCORE_DECAY + prob

        if score > SCORE_THRESH and time.time() - last_alert_time > ALERT_COOLDOWN:
            print(f"🚨 SIREN CONFIRMED | score={score:.2f}")
            last_alert_time = time.time()
            score = 0.0

        times.append(t)
        energy_vals.append(energy)
        prob_vals.append(prob)

    energy_line.set_data(times, energy_vals)
    prob_line.set_data(times, prob_vals)
    ax2.set_xlim(0, max(10, times[-1] + 0.5) if times else 10)

    return energy_line, prob_line

ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
plt.show()

# ====================== CLEANUP ======================
stream.stop()
stream.close()
print("🛑 Stopped")
