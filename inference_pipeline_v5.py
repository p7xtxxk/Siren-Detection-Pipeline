import time
import queue

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ====================== CONFIG ======================
SR = 22050
WIN_SEC = 0.5
HOP_SEC = 0.1

WIN = int(SR * WIN_SEC)
HOP = int(SR * HOP_SEC)

N_MELS = 64
EXPECTED_FRAMES = 130   # <<< MUST match model training
THRESH = 0.215

MODEL_PATH = "siren_model.tflite"   # change if needed

# ====================== GLOBALS ======================
audio_q = queue.Queue()
buffer = np.zeros(0, dtype=np.float32)

times = []
values = []

START_TIME = time.time()

# ====================== MODEL ======================
interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ TFLite model loaded")
except Exception as e:
    print("⚠️ Model not found — using audio energy instead")
    interpreter = None

# ====================== FEATURE EXTRACTION ======================
def extract_logmel(x):
    S = librosa.feature.melspectrogram(
        y=x,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )
    logS = librosa.power_to_db(S, ref=np.max)

    # ---- FIX SHAPE (PAD / TRIM) ----
    if logS.shape[1] < EXPECTED_FRAMES:
        pad = EXPECTED_FRAMES - logS.shape[1]
        logS = np.pad(logS, ((0, 0), (0, pad)), mode="constant")
    else:
        logS = logS[:, :EXPECTED_FRAMES]

    return logS[np.newaxis, ..., np.newaxis].astype(np.float32)

# ====================== INFERENCE ======================
def infer(segment):
    # Fallback: always produce something
    energy = float(np.sqrt(np.mean(segment ** 2)))

    if interpreter is None:
        return energy

    try:
        in_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()

        X = extract_logmel(segment)
        interpreter.set_tensor(in_det[0]["index"], X)
        interpreter.invoke()

        return float(interpreter.get_tensor(out_det[0]["index"])[0][0])
    except Exception:
        return energy

# ====================== AUDIO CALLBACK ======================
def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata[:, 0].astype(np.float32))

# ====================== AUDIO STREAM ======================
stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    blocksize=2048,
    callback=audio_callback
)
stream.start()
print("🎤 Microphone started")

# ====================== REAL-TIME PLOT ======================
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], color="blue", lw=2, label="Live Audio")
ax.axhline(THRESH, color="red", linestyle="--", label="Threshold")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")
ax.set_title("Real-Time Continuous Siren Detection")
ax.set_ylim(0, 1)
ax.legend()

def update(frame):
    global buffer

    # Pull audio from queue
    while not audio_q.empty():
        buffer = np.concatenate([buffer, audio_q.get()])

    # Sliding-window processing
    if len(buffer) >= WIN:
        segment = buffer[:WIN]
        buffer = buffer[HOP:]

        value = infer(segment)
        t = time.time() - START_TIME

        times.append(t)
        values.append(value)

    # Update graph (grows continuously)
    line.set_data(times, values)
    ax.set_xlim(0, max(10, times[-1] + 0.5) if times else 10)

    return line,

# Disable frame caching warning
ani = FuncAnimation(
    fig,
    update,
    interval=100,
    cache_frame_data=False
)

plt.show()

# ====================== CLEANUP ======================
stream.stop()
stream.close()
print("🛑 Stopped")
