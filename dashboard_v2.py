# [dashboard_realtime.py]

import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import sounddevice as sd
import matplotlib.pyplot as plt
import queue
import threading
import time

# --- Config ---
SR = 22050
WIN_SEC = 1.5
HOP_SEC = 0.1
WIN = int(SR * WIN_SEC)
N_MELS = 64
THRESH = 0.215

# --- Session state initialization ---
if ("running" not in st.session_state):
    st.session_state["running"] = False
if ("probs" not in st.session_state):
    st.session_state["probs"] = []
if ("times" not in st.session_state):
    st.session_state["times"] = []
if ("stream" not in st.session_state):
    st.session_state["stream"] = None
if ("thread" not in st.session_state):
    st.session_state["thread"] = None
if ("audio_queue" not in st.session_state):
    st.session_state["audio_queue"] = queue.Queue()

# --- Feature extraction ---
def extract_logmel(x, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)

def preprocess_segment(seg):
    logmel = extract_logmel(seg)
    logmel = logmel[..., np.newaxis][np.newaxis, ...]
    return logmel.astype(np.float32)

# --- TFLite inference ---
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, X):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], X)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details[0]["index"])[0][0])
    return prob

# --- Audio callback (lightweight) ---
def audio_callback(indata, frames, time_info, status):
    if (status):
        print(status)
    st.session_state["audio_queue"].put(indata.copy())

# --- Background processing thread ---
def process_audio(interpreter):
    buffer = np.zeros(0, dtype=np.float32)
    t = (0.0 if (len(st.session_state["times"]) == 0) else (st.session_state["times"][-1] + HOP_SEC))
    while (st.session_state["running"]):
        if (not st.session_state["audio_queue"].empty()):
            chunk = st.session_state["audio_queue"].get().flatten().astype(np.float32)
            # Resample chunk to SR if device samplerate differs (sounddevice uses stream samplerate)
            # Assuming stream uses SR; if not, set InputStream samplerate=SR to enforce
            buffer = np.concatenate([buffer, chunk])
            while (len(buffer) >= WIN):
                seg = buffer[:WIN]
                buffer = buffer[int(HOP_SEC * SR):]
                X = preprocess_segment(seg)
                prob = run_tflite_inference(interpreter, X)
                st.session_state["probs"].append(prob)
                st.session_state["times"].append(t)
                t += HOP_SEC
        else:
            time.sleep(0.01)

# --- UI ---
st.title("🚨 Real-Time Siren Detection Dashboard")

model_path = st.text_input(
    label="Path to TFLite model:",
    value=r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\siren_model.tflite"
)

interpreter = None
if (model_path and os.path.exists(model_path)):
    interpreter = load_tflite_model(model_path)
else:
    st.warning("Model path is invalid. Please provide a valid .tflite file path.")

col1, col2, col3 = st.columns(3)
with col1:
    start_button = st.button("Start Detection")
with col2:
    stop_button = st.button("Stop Detection")
with col3:
    clear_button = st.button("Clear Graph")

# --- Start detection ---
if (start_button and interpreter):
    if (not st.session_state["running"]):
        st.session_state["running"] = True
        st.write("🎤 Listening... Play siren audio nearby or speak.")
        try:
            st.session_state["stream"] = sd.InputStream(
                samplerate=SR,
                channels=1,
                callback=audio_callback,
                blocksize=4096
            )
            st.session_state["stream"].start()
        except Exception as e:
            st.session_state["running"] = False
            st.error(f"Failed to start audio stream: {e}")
        if (st.session_state["running"]):
            st.session_state["thread"] = threading.Thread(
                target=process_audio,
                args=(interpreter,),
                daemon=True
            )
            st.session_state["thread"].start()
    else:
        st.info("Detection is already running.")

# --- Stop detection ---
if (stop_button):
    if (st.session_state["running"]):
        st.session_state["running"] = False
        if (st.session_state["stream"] is not None):
            try:
                st.session_state["stream"].stop()
                st.session_state["stream"].close()
            except Exception as e:
                st.warning(f"Stream stop warning: {e}")
            st.session_state["stream"] = None
        st.write("🛑 Detection stopped.")
    else:
        st.info("Detection is not running.")

# --- Clear graph ---
if (clear_button):
    st.session_state["probs"].clear()
    st.session_state["times"].clear()
    st.session_state["audio_queue"] = queue.Queue()
    st.write("🧹 Cleared graph and buffers.")

# --- Live plot ---
placeholder = st.empty()
if (len(st.session_state["probs"]) > 0):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state["times"], st.session_state["probs"], label="Siren probability")
    ax.axhline(THRESH, color="r", linestyle="--", label=f"Threshold={THRESH}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title("Real-Time Detection Curve")
    ax.legend()
    placeholder.pyplot(fig)
else:
    st.info("Graph will appear here once audio is processed. Press Start to begin.")
