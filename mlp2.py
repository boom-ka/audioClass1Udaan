import streamlit as st
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import joblib
import wave
from df import init_df, enhance
import threading
import time
from scipy.signal import butter, lfilter, spectrogram

# === Load models ===
mlp_model = joblib.load("mlp_emotion_classifier_best_model2.joblib")
encoder = joblib.load("emotion_encoder.joblib")
scaler = joblib.load("feature_scaler.joblib")
model, df_state, _ = init_df()

# === Parameters ===
SAMPLE_RATE = 16000
BLOCKSIZE = 16000
DENOISE_BLEND = 1
SILENCE_THRESHOLD = 0.015

# === Global buffers ===
audio_buffer = np.zeros(BLOCKSIZE)
latest_cleaned_audio = np.zeros(BLOCKSIZE)
current_emotion = "Neutral"
current_confidence = 0.0
buffer_lock = threading.Lock()

# === Filter, Normalize, and Denoise ===
def bandpass_filter(data, low=300, high=3400, fs=SAMPLE_RATE):
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

def preprocess(audio):
    filtered = bandpass_filter(audio)
    return normalize(filtered)

def denoise(audio):
    tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        tensor_audio = tensor_audio.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = enhance(model, df_state, tensor_audio.cpu())
    return output.numpy().flatten()

def is_silence(audio):
    return np.sqrt(np.mean(audio ** 2)) < SILENCE_THRESHOLD

def extract_features_from_array(y, sr=16000):
    features = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    features = np.hstack((features, mfccs))
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, chroma))
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.hstack((features, mel))
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, contrast))
    y_harm = librosa.effects.harmonic(y)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr).T, axis=0)
    features = np.hstack((features, tonnetz))
    return features

# === Audio callback ===
def audio_callback(indata, frames, time_info, status):
    global audio_buffer, latest_cleaned_audio, current_emotion, current_confidence
    raw_audio = indata[:, 0]
    processed = preprocess(raw_audio)
    cleaned = denoise(processed)
    output_audio = np.zeros_like(cleaned) if is_silence(processed) else DENOISE_BLEND * cleaned + (1 - DENOISE_BLEND) * processed
    with buffer_lock:
        audio_buffer = raw_audio
        latest_cleaned_audio = output_audio
    try:
        features = extract_features_from_array(output_audio)
        features_scaled = scaler.transform([features])
        probs = mlp_model.predict_proba(features_scaled)[0]
        idx = np.argmax(probs)
        current_emotion = encoder.inverse_transform([idx])[0]
        current_confidence = probs[idx]
    except:
        current_emotion = "Error"
        current_confidence = 0.0

# === Audio thread ===
def start_stream():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        while True:
            time.sleep(0.1)

# === Streamlit App ===
st.set_page_config(page_title="Real-Time Emotion Recognition", layout="wide")
st.title("🎤 Real-Time Emotion Recognition with DeepFilterNet + MLP")

st.sidebar.header("📊 Current Emotion")
st.sidebar.markdown("Emotion updates every ~0.5s")
emotion_text = st.sidebar.empty()
confidence_text = st.sidebar.empty()

# === Start audio thread once ===
if 'thread_started' not in st.session_state:
    audio_thread = threading.Thread(target=start_stream, daemon=True)
    audio_thread.start()
    st.session_state.thread_started = True

# === Main plotting loop ===
plot_placeholders = [st.empty() for _ in range(5)]

while True:
    with buffer_lock:
        raw = audio_buffer.copy()
        clean = latest_cleaned_audio.copy()

    # --- Plotting ---
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))

    axs[0].plot(raw)
    axs[0].set_title("Time Domain (Raw)")

    f, t, Sxx = spectrogram(raw, SAMPLE_RATE)
    axs[1].imshow(10 * np.log10(Sxx + 1e-10), aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    axs[1].set_title("Spectrogram (Raw)")

    f, t, Sxx = spectrogram(clean, SAMPLE_RATE)
    axs[2].imshow(10 * np.log10(Sxx + 1e-10), aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    axs[2].set_title("Spectrogram (Cleaned)")

    fft_raw = np.abs(np.fft.rfft(raw))
    freqs_raw = np.fft.rfftfreq(len(raw), d=1/SAMPLE_RATE)
    axs[3].plot(freqs_raw, fft_raw)
    axs[3].set_title("FFT (Raw)")

    fft_clean = np.abs(np.fft.rfft(clean))
    freqs_clean = np.fft.rfftfreq(len(clean), d=1/SAMPLE_RATE)
    axs[4].plot(freqs_clean, fft_clean)
    axs[4].set_title("FFT (Cleaned)")

    for i in range(5):
        plot_placeholders[i].pyplot(fig, clear_figure=True)

    emotion_text.markdown(f"### Emotion: `{current_emotion}`")
    confidence_text.markdown(f"**Confidence**: `{current_confidence:.2f}`")

    time.sleep(0.5)
