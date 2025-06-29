import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, lfilter
import threading
import torch
import wave
import atexit
import torch.nn.functional as F  # Add this at the top if not already
from df import init_df, enhance
model_df, df_state, _ = init_df()
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForSequenceClassification,Wav2Vec2Config
config = Wav2Vec2Config.from_pretrained("./wav2vec2-lg-xlsr-en-speech-emotion-recognition")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model_emotion = Wav2Vec2ForSequenceClassification.from_pretrained(
    "./wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    config=config  # use this explicitly
)
model_emotion.eval()

model_emotion.to(device)

# EMOTION_LABELS = ['angry', 'neutral', 'happy', 'sad', 'fearful', 'surprised']

id2label = {
    "0": "angry",
    "1": "calm",
    "2": "disgust",
    "3": "fearful",
    "4": "happy",
    "5": "neutral",
    "6": "sad",
    "7": "surprised",
}

def classify_emotion_wav2vec2(audio, sample_rate=16000):
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs["input_values"]
    with torch.no_grad():
        logits = model_emotion(input_values.to(device)).logits
    predicted_id = torch.argmax(logits, dim=1).item()
    # print("predicted_id:", predicted_id, type(predicted_id))
    # print("id2label keys:", model_emotion.config.id2label.keys())
    predicted_label = id2label[predicted_id]

    return predicted_label


# === Globals ===
SAMPLE_RATE = 16000
BLOCKSIZE = 8000  # 0.5 sec
DENOISE_BLEND = 0.6
SILENCE_THRESHOLD = 0.015

audio_buffer = np.zeros(BLOCKSIZE)
latest_cleaned_audio = np.zeros(BLOCKSIZE)
raw_audio_storage = []
clean_audio_storage = []

# === Bandpass Filter ===
def bandpass_filter(data, low=300, high=3400, fs=SAMPLE_RATE):
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

# === Normalize ===
def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

# === Preprocess ===
def preprocess(audio):
    filtered = bandpass_filter(audio)
    normalized = normalize(filtered)
    return normalized

# === DeepFilterNet Denoising ===
def denoise(audio):
    tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        tensor_audio = tensor_audio.to("cuda")
        model_df.to("cuda")
    with torch.no_grad():
        output = enhance(model_df, df_state, tensor_audio.cpu())
    return output.numpy().flatten()

# === Silence Detection ===
def is_silence(audio, threshold=SILENCE_THRESHOLD):
    energy = np.sqrt(np.mean(audio ** 2))
    return energy < threshold

# === Save Audio ===
def save_audio(filename, data):
    flat = np.concatenate(data)
    scaled = np.int16(flat / (np.max(np.abs(flat)) + 1e-6) * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(scaled.tobytes())

# === Save on Exit ===
def on_exit():
    print("\nSaving audio...")
    save_audio("radio_raw.wav", raw_audio_storage)
    save_audio("radio_cleaned.wav", clean_audio_storage)
    print("Done.")

atexit.register(on_exit)

# === Audio Callback ===
def audio_callback(indata, frames, time, status):
    global audio_buffer, latest_cleaned_audio
    audio_buffer = indata[:, 0]
    raw_audio_storage.append(audio_buffer.copy())

    processed = preprocess(audio_buffer)
    cleaned = denoise(processed)

    if is_silence(processed):
        output_audio = np.zeros_like(cleaned)
    else:
        output_audio = DENOISE_BLEND * cleaned + (1 - DENOISE_BLEND) * processed

    latest_cleaned_audio = output_audio
    clean_audio_storage.append(output_audio.copy())

# === Real-Time Emotion Classification ===
def classify_emotion_wav2vec2(audio, sample_rate=16000):
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model_emotion(input_values).logits

    probs = F.softmax(logits, dim=-1)
    predicted_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_id].item()

    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[str(predicted_id)]
    return predicted_label.lower(), confidence



def emotion_classifier_thread():
    while True:
        if len(clean_audio_storage) >= 2:
            combined = np.concatenate(clean_audio_storage[-2:])
            if len(combined) >= SAMPLE_RATE:
                segment = combined[-SAMPLE_RATE:]
                emotion, confidence = classify_emotion_wav2vec2(segment)
                print(f"[Emotion Detected] {emotion} (Confidence: {confidence:.2f})")
        sd.sleep(1000)  # classify every 1s

# === Plotting ===
def plot_live():
    plt.ion()
    fig, axs = plt.subplots(3, figsize=(10, 6))

    while True:
        axs[0].cla()
        axs[0].plot(audio_buffer)
        axs[0].set_title("Time Domain (Raw)")
        axs[0].set_ylim([-1, 1])

        fft_raw = np.abs(fft(audio_buffer))[:len(audio_buffer)//2]
        axs[1].cla()
        axs[1].plot(fft_raw)
        axs[1].set_title("Frequency Domain (Raw)")

        fft_clean = np.abs(fft(latest_cleaned_audio))[:len(latest_cleaned_audio)//2]
        axs[2].cla()
        axs[2].plot(fft_clean)
        axs[2].set_title("Frequency Domain (Cleaned, Speech-Gated)")

        plt.tight_layout()
        plt.pause(0.05)

# === Start Audio ===
def start_audio_stream():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        sd.sleep(1000000)

# === Main ===
if __name__ == "__main__":
    try:
        threading.Thread(target=start_audio_stream, daemon=True).start()
        threading.Thread(target=emotion_classifier_thread, daemon=True).start()
        plot_live()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
