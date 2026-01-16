import os
import random
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import streamlit as st

# --- Config ---
SR = 22050
WIN_SEC = 1.5
HOP_SEC = 0.1
WIN = int(SR * WIN_SEC)
N_MELS = 64
THRESH = 0.215

# --- Feature extraction ---
def extract_logmel(x, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=n_mels)
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
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    return prob

# --- Continuous detection ---
def continuous_detection(file_path, interpreter, sr=SR, win_sec=WIN_SEC, hop_sec=HOP_SEC, thresh=THRESH):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    step = int(hop_sec * sr)
    win = int(win_sec * sr)
    times, probs = [], []

    for start in range(0, len(y) - win + 1, step):
        seg = y[start:start+win]
        X = preprocess_segment(seg)
        prob = run_tflite_inference(interpreter, X)
        times.append(start / sr)
        probs.append(prob)

    # Find exact onset/offset points
    detections = []
    active = False
    onset = None
    for t, p in zip(times, probs):
        if not active and p >= thresh:
            active = True
            onset = t
        elif active and p < thresh:
            active = False
            detections.append((onset, t))
    if active:
        detections.append((onset, times[-1]))

    return times, probs, detections

# --- Visualization ---
def plot_results(file_path, times, probs, detections):
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024,
                                       hop_length=256, n_mels=N_MELS)
    log_S = librosa.power_to_db(S, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Spectrogram
    librosa.display.specshow(log_S, sr=SR, hop_length=256,
                             x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title(f"Log-Mel Spectrogram: {os.path.basename(file_path)}")
    fig.colorbar(axes[0].collections[0], ax=axes[0], format='%+2.0f dB')

    # Probability curve
    axes[1].plot(times, probs, label="Siren probability")
    axes[1].axhline(THRESH, color='r', linestyle='--', label=f"Threshold={THRESH}")
    for onset, offset in detections:
        axes[1].axvspan(onset, offset, color='red', alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Continuous Detection Curve")
    axes[1].legend()

    plt.tight_layout()
    return fig

# --- Streamlit UI ---
st.title("🚨 Siren Detection Dashboard")

model_path = st.text_input("Path to TFLite model:", 
    r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\siren_model.tflite")

siren_dir = st.text_input("Siren folder:", 
    r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\data\siren")
nonsiren_dir = st.text_input("Non-Siren folder:", 
    r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\data\non_siren")

interpreter = None
if model_path and os.path.exists(model_path):
    interpreter = load_tflite_model(model_path)

choice = st.radio("Choose input:", ["Random Siren", "Random Non-Siren", "Upload File"])

file_path = None
if choice == "Random Siren" and os.path.exists(siren_dir):
    file_path = os.path.join(siren_dir, random.choice(os.listdir(siren_dir)))
elif choice == "Random Non-Siren" and os.path.exists(nonsiren_dir):
    file_path = os.path.join(nonsiren_dir, random.choice(os.listdir(nonsiren_dir)))
elif choice == "Upload File":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        tmp_path = f"temp_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        file_path = tmp_path

if file_path and interpreter:
    st.write(f"### Analyzing: {os.path.basename(file_path)}")
    times, probs, detections = continuous_detection(file_path, interpreter)
    for onset, offset in detections:
        st.write(f"Siren detected from {onset:.2f}s to {offset:.2f}s")
    fig = plot_results(file_path, times, probs, detections)
    st.pyplot(fig)
