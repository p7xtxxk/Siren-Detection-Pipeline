import os
import random
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display

# --- Config (match training) ---
SR = 22050
WIN_SEC = 1.5
HOP_SEC = 0.1   # smaller hop for near-continuous detection
WIN = int(SR * WIN_SEC)
N_MELS = 64
THRESH = 0.215  # tuned threshold from training

# --- Feature extraction (same as training) ---
def extract_logmel(x, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)

def preprocess_segment(seg):
    logmel = extract_logmel(seg)
    logmel = logmel[..., np.newaxis][np.newaxis, ...]  # add channel + batch
    return logmel.astype(np.float32)

# --- TFLite inference ---
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

# --- Utility: pick one random .wav from a folder ---
def pick_random_wav(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# --- Visualization ---
def plot_results(file_path, times, probs, detections):
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024,
                                       hop_length=256, n_mels=N_MELS)
    log_S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(14, 6))

    # Spectrogram
    plt.subplot(2,1,1)
    librosa.display.specshow(log_S, sr=SR, hop_length=256,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Log-Mel Spectrogram: {os.path.basename(file_path)}")

    # Probability curve
    plt.subplot(2,1,2)
    plt.plot(times, probs, label="Siren probability")
    plt.axhline(THRESH, color='r', linestyle='--', label=f"Threshold={THRESH}")
    for onset, offset in detections:
        plt.axvspan(onset, offset, color='red', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("Continuous Detection Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    siren_dir = r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\data\siren"
    nonsiren_dir = r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\data\non_siren"
    model_path = r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\siren_model.tflite"

    interpreter = load_tflite_model(model_path)

    # Bias-free random selection
    test_paths = []
    p1 = pick_random_wav(siren_dir)
    p2 = pick_random_wav(nonsiren_dir)
    if p1: test_paths.append(p1)
    if p2: test_paths.append(p2)

    for path in test_paths:
        times, probs, detections = continuous_detection(path, interpreter)
        print(f"\nFile: {os.path.basename(path)}")
        print("Exact detections:")
        for onset, offset in detections:
            print(f"  Siren detected from {onset:.2f}s to {offset:.2f}s")

        plot_results(path, times, probs, detections)
