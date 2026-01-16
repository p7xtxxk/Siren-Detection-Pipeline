import os
import numpy as np
import librosa
import tensorflow as tf

# Audio config (must match training)
SR = 22050
WIN_SEC = 1.5
HOP_SEC = 0.75
WIN = int(SR * WIN_SEC)
HOP = int(SR * HOP_SEC)
N_MELS = 64
THRESH = 0.215  # tuned threshold from training

# -----------------------------
# Feature extraction (same as training)
# -----------------------------
def extract_logmel(x, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)

def file_to_segments(path):
    x, sr = librosa.load(path, sr=SR, mono=True)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))  # normalize amplitude
    feats = []
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    for start in range(0, max(len(x) - WIN + 1, 1), HOP):
        seg = x[start:start+WIN]
        feats.append(extract_logmel(seg))
    return np.stack(feats)

# -----------------------------
# Preprocess for CNN input
# -----------------------------
def preprocess_file(path):
    feats = file_to_segments(path)
    feats = feats[..., np.newaxis]  # add channel dim
    return feats.astype(np.float32)

# -----------------------------
# Load TFLite model
# -----------------------------
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, X):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    results = []
    for i in range(len(X)):
        sample = np.expand_dims(X[i], axis=0)  # batch=1
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        label = "SIREN" if prob > THRESH else "NON-SIREN"
        results.append((prob, label))
    return results

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    model_path = r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\siren_model.tflite"
    test_file = r"C:\Users\prate\Downloads\College Academics\Minor Project\Minor Project\Siren-Detection-Pipeline\data\siren\5-150409-A-42.wav"

    interpreter = load_tflite_model(model_path)
    X = preprocess_file(test_file)
    results = run_tflite_inference(interpreter, X)

    print(f"\nResults for {os.path.basename(test_file)}:")
    for i, (prob, label) in enumerate(results):
        start = i * HOP / SR
        end = start + WIN_SEC
        print(f"  {start:.2f}-{end:.2f}s → {label} (prob={prob:.4f})")

