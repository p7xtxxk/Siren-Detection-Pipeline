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
HOP_SEC = 0.75
WIN = int(SR * WIN_SEC)
HOP = int(SR * HOP_SEC)
N_MELS = 64
THRESH = 0.215  # tuned threshold from training

# --- Feature extraction (same as training) ---
def extract_logmel(x, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)

def file_to_segments(path):
    x, sr = librosa.load(path, sr=SR, mono=True)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))  # amplitude normalization
    feats = []
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    for start in range(0, max(len(x) - WIN + 1, 1), HOP):
        seg = x[start:start+WIN]
        feats.append(extract_logmel(seg))
    return np.stack(feats)

def preprocess_file(path):
    feats = file_to_segments(path)
    feats = feats[..., np.newaxis]  # add channel dim
    return feats.astype(np.float32)

# --- TFLite inference ---
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, X, thresh=THRESH):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    results = []
    for i in range(len(X)):
        sample = np.expand_dims(X[i], axis=0)  # batch=1
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        label = "SIREN" if prob >= thresh else "NON-SIREN"
        results.append((prob, label))
    return results

# --- Hysteresis & timeline merging ---
def merge_timeline(results, win_sec=WIN_SEC, hop_sec=HOP_SEC, min_consecutive=2):
    intervals = []
    current_label = None
    current_start = None
    consec_siren = 0

    for i, (prob, label) in enumerate(results):
        start = i * hop_sec
        end = start + win_sec

        effective_label = label
        if label == "SIREN":
            consec_siren += 1
            if consec_siren < min_consecutive:
                effective_label = "NON-SIREN"
        else:
            consec_siren = 0

        if current_label is None:
            current_label = effective_label
            current_start = start
            last_end = end
        else:
            if effective_label == current_label:
                last_end = end
            else:
                intervals.append((current_start, last_end, current_label))
                current_label = effective_label
                current_start = start
                last_end = end

    if current_label is not None:
        intervals.append((current_start, last_end, current_label))

    return intervals

# --- Utility: pick one random .wav from a folder ---
def pick_random_wav(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# --- Visualization: plot continuous log-mel spectrogram ---
def plot_logmel(file_path, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=256):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Log-Mel Spectrogram: {os.path.basename(file_path)}")
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
        X = preprocess_file(path)
        results = run_tflite_inference(interpreter, X, thresh=THRESH)
        intervals = merge_timeline(results, win_sec=WIN_SEC, hop_sec=HOP_SEC, min_consecutive=2)

        print(f"\nFile: {os.path.basename(path)}")
        print("Raw windows:")
        for i, (prob, label) in enumerate(results):
            start = i * HOP_SEC
            end = start + WIN_SEC
            print(f"  {start:.2f}-{end:.2f}s → {label} (prob={prob:.4f})")

        print("\nMerged timeline (hysteresis=2):")
        for (start, end, label) in intervals:
            print(f"  {start:.2f}-{end:.2f}s → {label}")

        # Plot log-mel spectrogram for this file
        plot_logmel(path)
