import os
import time
import queue
import threading

import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import sounddevice as sd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
SR = 22050
WIN_SEC = 0.5
HOP_SEC = 0.1
WIN = int(SR * WIN_SEC)
HOP = int(SR * HOP_SEC)
N_MELS = 64
THRESH = 0.215
REFRESH_MS = 200

# ===================== GLOBALS (THREAD SAFE) =====================
audio_q = queue.Queue()
PROBS = []
TIMES = []

RUN_FLAG = False
START_TIME = 0.0

stream = None
worker_thread = None

# ===================== FEATURE EXTRACTION =====================
def extract_logmel(x):
    S = librosa.feature.melspectrogram(
        y=x,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )
    logS = librosa.power_to_db(S, ref=np.max)
    return logS[np.newaxis, ..., np.newaxis].astype(np.float32)

# ===================== TFLITE =====================
@st.cache_resource
def load_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def infer(interpreter, X):
    try:
        in_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()
        interpreter.set_tensor(in_det[0]["index"], X)
        interpreter.invoke()
        return float(interpreter.get_tensor(out_det[0]["index"])[0][0])
    except Exception:
        return 0.0

# ===================== AUDIO CALLBACK =====================
def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata[:, 0].astype(np.float32))

# ===================== AUDIO THREAD =====================
def audio_worker(interpreter):
    global RUN_FLAG, PROBS, TIMES

    buffer = np.zeros(0, dtype=np.float32)

    while RUN_FLAG:
        try:
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        buffer = np.concatenate([buffer, chunk])

        if len(buffer) >= WIN:
            seg = buffer[:WIN]
            buffer = buffer[HOP:]

            value = infer(interpreter, extract_logmel(seg)) if interpreter else float(np.sqrt(np.mean(seg**2)))
            t = time.time() - START_TIME

            PROBS.append(value)
            TIMES.append(t)

# ===================== UI =====================
st.set_page_config(layout="wide")
st.title("🚨 Real-Time Growing Audio Graph (STABLE)")

model_path = st.text_input(
    "Path to TFLite model:",
    value=r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\siren_model.tflite"
)

interpreter = load_tflite(model_path) if os.path.exists(model_path) else None

col1, col2, col3 = st.columns(3)
start = col1.button("▶ Start")
stop = col2.button("⏹ Stop")
clear = col3.button("🧹 Clear")

# ===================== START =====================
if start and not RUN_FLAG:
    RUN_FLAG = True
    PROBS.clear()
    TIMES.clear()

    START_TIME = time.time()

    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=2048,
        callback=audio_callback
    )
    stream.start()

    worker_thread = threading.Thread(
        target=audio_worker,
        args=(interpreter,),
        daemon=True
    )
    worker_thread.start()

    st.rerun()

# ===================== STOP =====================
if stop and RUN_FLAG:
    RUN_FLAG = False
    time.sleep(0.2)

    if stream:
        stream.stop()
        stream.close()

    st.rerun()

# ===================== CLEAR =====================
if clear:
    PROBS.clear()
    TIMES.clear()
    st.rerun()

# ===================== LIVE GRAPH =====================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(TIMES, PROBS, lw=2)
ax.axhline(THRESH, color="red", linestyle="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")
ax.set_ylim(0, 1)
ax.set_title("Live Audio Detection (Growing in Time)")
st.pyplot(fig)

# ===================== AUTO REFRESH =====================
if RUN_FLAG:
    time.sleep(REFRESH_MS / 1000)
    st.rerun()
