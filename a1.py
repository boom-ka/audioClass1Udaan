import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import threading
from df import init_df, enhance
import torch
# This initializes the default pre-trained model (typically DeepFilterNet2)
model, df_state, _ = init_df()
# Buffer for audio
audio_buffer = np.zeros(16000)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0]


def denoise(audio):
    tensor_audio = torch.from_numpy(audio.astype(np.float32))

    if tensor_audio.ndim == 1:
        tensor_audio = tensor_audio.unsqueeze(0)  # [samples] -> [1, samples]

    if torch.cuda.is_available():
        tensor_audio = tensor_audio.to("cuda")
        model.to("cuda")

    print("Using device:", tensor_audio.device)
    with torch.no_grad():
        # Move audio back to CPU before passing to DeepFilterNet enhance (if required)
        if tensor_audio.device.type == 'cuda':
            tensor_audio_cpu = tensor_audio.cpu()
        else:
            tensor_audio_cpu = tensor_audio

        output = enhance(model, df_state, tensor_audio_cpu)

    return output.numpy().flatten()



def start_audio_stream():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, blocksize=16000):
        sd.sleep(1000000)

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

        cleaned = denoise(audio_buffer)
        fft_clean = np.abs(fft(cleaned))[:len(cleaned)//2]
        axs[2].cla()
        axs[2].plot(fft_clean)
        axs[2].set_title("Frequency Domain (Cleaned)")

        plt.tight_layout()
        plt.pause(0.05)

# Start audio + plotting
threading.Thread(target=start_audio_stream, daemon=True).start()
plot_live()
