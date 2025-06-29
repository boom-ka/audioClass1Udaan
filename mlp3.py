import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import wave
import io
import base64
from datetime import datetime

# Audio processing imports
import sounddevice as sd
import librosa
from scipy.signal import butter, lfilter, spectrogram
from scipy.fft import fft
import joblib
import torch
from df import init_df, enhance

# Set page config
st.set_page_config(
    page_title="🎙️ Real-Time Audio Processing",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #4CAF50; }
    .status-inactive { background-color: #f44336; }
    .status-warning { background-color: #FF9800; }
    
    .emotion-card {
        background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .processing-card {
        background: linear-gradient(45deg, #4ECDC4, #44A08D);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_processing' not in st.session_state:
    st.session_state.audio_processing = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = queue.Queue()
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = np.zeros(16000)
if 'clean_buffer' not in st.session_state:
    st.session_state.clean_buffer = np.zeros(16000)
if 'raw_audio_storage' not in st.session_state:
    st.session_state.raw_audio_storage = []
if 'clean_audio_storage' not in st.session_state:
    st.session_state.clean_audio_storage = []

# Audio processing configuration
SAMPLE_RATE = 16000
BLOCKSIZE = 16000
SILENCE_THRESHOLD = 0.015

# Load ML models (with error handling)
@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    models = {}
    try:
        models['mlp'] = joblib.load("mlp_emotion_classifier_best_model2.joblib")
        models['encoder'] = joblib.load("emotion_encoder.joblib")
        models['scaler'] = joblib.load("feature_scaler.joblib")
        st.success("✅ ML models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}")
        st.info("💡 Using mock models for demonstration")
        # Mock models for demo
        models['mlp'] = None
        models['encoder'] = None
        models['scaler'] = None
    
    try:
        models['deepfilter'], models['df_state'], _ = init_df()
        st.success("✅ DeepFilterNet loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading DeepFilterNet: {e}")
        models['deepfilter'] = None
        models['df_state'] = None
    
    return models

# Audio processing functions
def bandpass_filter(data, low=300, high=3400, fs=SAMPLE_RATE):
    """Apply bandpass filter"""
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

def normalize_audio(audio):
    """Normalize audio to [-1, 1]"""
    return audio / (np.max(np.abs(audio)) + 1e-6)

def preprocess_audio(audio):
    """Preprocess audio before denoising"""
    filtered = bandpass_filter(audio)
    normalized = normalize_audio(filtered)
    return normalized

def denoise_audio(audio, model, df_state):
    """Denoise audio using DeepFilterNet"""
    if model is None:
        # Mock denoising for demo
        return audio * 0.8
    
    try:
        tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        if torch.cuda.is_available():
            tensor_audio = tensor_audio.to("cuda")
            model.to("cuda")
        with torch.no_grad():
            output = enhance(model, df_state, tensor_audio.cpu())
        return output.numpy().flatten()
    except Exception as e:
        st.error(f"Denoising error: {e}")
        return audio

def is_silence(audio, threshold=SILENCE_THRESHOLD):
    """Check if audio is silence"""
    energy = np.sqrt(np.mean(audio ** 2))
    return energy < threshold

def extract_features_from_array(y, sr=16000):
    """Extract features for emotion classification"""
    features = np.array([])
    
    try:
        # MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        features = np.hstack((features, mfccs))
        
        # Chroma
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        features = np.hstack((features, chroma))
        
        # Mel spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        features = np.hstack((features, mel))
        
        # Spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        features = np.hstack((features, contrast))
        
        # Tonnetz
        y_harm = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr).T, axis=0)
        features = np.hstack((features, tonnetz))
        
    except Exception as e:
        # Return dummy features if extraction fails
        features = np.random.randn(193)  # Typical feature size
    
    return features

def predict_emotion(audio, models):
    """Predict emotion from audio"""
    start_time = time.perf_counter()
    
    if models['mlp'] is None:
        # Mock prediction for demo
        emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'fearful', 'disgusted']
        emotion = np.random.choice(emotions)
        confidence = np.random.uniform(0.6, 0.95)
        processing_time = (time.perf_counter() - start_time) * 1000
        return emotion, confidence, processing_time
    
    try:
        features = extract_features_from_array(audio, sr=SAMPLE_RATE)
        features_scaled = models['scaler'].transform([features])
        probs = models['mlp'].predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]
        emotion = models['encoder'].inverse_transform([predicted_idx])[0]
        
        processing_time = (time.perf_counter() - start_time) * 1000
        return emotion, confidence, processing_time
        
    except Exception as e:
        st.error(f"Emotion prediction error: {e}")
        return "unknown", 0.0, 0.0

def process_audio_chunk(indata, models, denoise_blend=1.0):
    """Process a chunk of audio data"""
    start_time = time.perf_counter()
    
    # Store raw audio
    st.session_state.raw_audio_storage.append(indata.copy())
    
    # Preprocessing
    processed = preprocess_audio(indata)
    
    # Denoising
    denoise_start = time.perf_counter()
    cleaned = denoise_audio(processed, models['deepfilter'], models['df_state'])
    denoise_time = (time.perf_counter() - denoise_start) * 1000
    
    # Conditional blending
    if is_silence(processed):
        output_audio = np.zeros_like(cleaned)
    else:
        output_audio = denoise_blend * cleaned + (1 - denoise_blend) * processed
    
    # Store cleaned audio
    st.session_state.clean_audio_storage.append(output_audio.copy())
    
    # Update buffers
    st.session_state.audio_buffer = indata
    st.session_state.clean_buffer = output_audio
    
    # Emotion prediction
    emotion, confidence, emotion_time = predict_emotion(output_audio, models)
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    # Store results
    result = {
        'timestamp': datetime.now(),
        'emotion': emotion,
        'confidence': confidence,
        'denoise_time': denoise_time,
        'emotion_time': emotion_time,
        'total_time': total_time,
        'volume': np.sqrt(np.mean(indata ** 2)),
        'is_silence': is_silence(processed)
    }
    
    return result

# Visualization functions
def create_time_domain_plot(raw_data, clean_data):
    """Create time domain visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Raw Audio', 'Cleaned Audio'),
        vertical_spacing=0.1
    )
    
    time_axis = np.linspace(0, len(raw_data) / SAMPLE_RATE, len(raw_data))
    
    fig.add_trace(
        go.Scatter(x=time_axis, y=raw_data, name='Raw', line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time_axis, y=clean_data, name='Cleaned', line=dict(color='#4ECDC4')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        title_text="Time Domain Analysis",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_frequency_domain_plot(raw_data, clean_data):
    """Create frequency domain visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Raw Audio Spectrum', 'Cleaned Audio Spectrum'),
        vertical_spacing=0.1
    )
    
    # FFT
    raw_fft = np.abs(np.fft.rfft(raw_data))
    clean_fft = np.abs(np.fft.rfft(clean_data))
    freqs = np.fft.rfftfreq(len(raw_data), d=1/SAMPLE_RATE)
    
    fig.add_trace(
        go.Scatter(x=freqs, y=raw_fft, name='Raw', line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=freqs, y=clean_fft, name='Cleaned', line=dict(color='#4ECDC4')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        title_text="Frequency Domain Analysis",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 4000])
    fig.update_yaxes(title_text="Magnitude")
    
    return fig

def create_spectrogram_plot(audio_data, title="Spectrogram"):
    """Create spectrogram visualization"""
    f, t, Sxx = spectrogram(audio_data, SAMPLE_RATE, nperseg=256, noverlap=128)
    
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx + 1e-10),
        x=t,
        y=f,
        colorscale='Viridis',
        colorbar=dict(title="Power (dB)")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_yaxes(range=[0, 4000])
    
    return fig

def create_emotion_history_plot(emotion_history):
    """Create emotion history visualization"""
    if not emotion_history:
        return go.Figure()
    
    df = pd.DataFrame(emotion_history)
    
    fig = px.line(df, x='timestamp', y='confidence', color='emotion',
                  title="Emotion Detection History",
                  labels={'confidence': 'Confidence', 'timestamp': 'Time'})
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def save_audio_to_wav(audio_data, filename):
    """Save audio data to WAV format"""
    if not audio_data:
        return None
    
    # Concatenate all audio chunks
    full_audio = np.concatenate(audio_data)
    
    # Normalize and convert to int16
    normalized = full_audio / (np.max(np.abs(full_audio)) + 1e-6)
    audio_int16 = (normalized * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return buffer.getvalue()

# Main app
def main():
    # Load models
    models = load_models()
    
    # Header
    st.markdown('<h1 class="main-header">🎙️ Real-Time Audio Processing</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced emotion detection and noise reduction</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Audio Controls")
    
    # Audio processing controls
    denoise_blend = st.sidebar.slider("Denoise Level", 0.0, 1.0, 1.0, 0.1)
    silence_threshold = st.sidebar.slider("Silence Threshold", 0.001, 0.1, 0.015, 0.001)
    
    # Update global threshold
    global SILENCE_THRESHOLD
    SILENCE_THRESHOLD = silence_threshold
    
    # Processing controls
    st.sidebar.subheader("📊 Processing")
    process_button = st.sidebar.button("🎙️ Start Processing" if not st.session_state.audio_processing else "⏹️ Stop Processing")
    
    if process_button:
        st.session_state.audio_processing = not st.session_state.audio_processing
        if st.session_state.audio_processing:
            st.sidebar.success("Audio processing started!")
        else:
            st.sidebar.info("Audio processing stopped!")
    
    # Audio file management
    st.sidebar.subheader("💾 Audio Management")
    if st.sidebar.button("Save Raw Audio"):
        if st.session_state.raw_audio_storage:
            audio_data = save_audio_to_wav(st.session_state.raw_audio_storage, "raw_audio.wav")
            if audio_data:
                st.sidebar.download_button(
                    label="Download Raw Audio",
                    data=audio_data,
                    file_name="raw_audio.wav",
                    mime="audio/wav"
                )
        else:
            st.sidebar.warning("No audio data to save!")
    
    if st.sidebar.button("Save Cleaned Audio"):
        if st.session_state.clean_audio_storage:
            audio_data = save_audio_to_wav(st.session_state.clean_audio_storage, "cleaned_audio.wav")
            if audio_data:
                st.sidebar.download_button(
                    label="Download Cleaned Audio",
                    data=audio_data,
                    file_name="cleaned_audio.wav",
                    mime="audio/wav"
                )
        else:
            st.sidebar.warning("No audio data to save!")
    
    if st.sidebar.button("Clear All Data"):
        st.session_state.raw_audio_storage = []
        st.session_state.clean_audio_storage = []
        st.session_state.emotion_history = []
        st.session_state.performance_history = []
        st.sidebar.success("All data cleared!")
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    # Status indicators
    with col1:
        status_color = "🟢" if st.session_state.audio_processing else "🔴"
        status_text = "Active" if st.session_state.audio_processing else "Inactive"
        st.markdown(f"**Audio Status:** {status_color} {status_text}")
    
    with col2:
        st.markdown(f"**Sample Rate:** {SAMPLE_RATE} Hz")
    
    with col3:
        frames_processed = len(st.session_state.raw_audio_storage)
        st.markdown(f"**Frames Processed:** {frames_processed}")
    
    with col4:
        total_duration = frames_processed * BLOCKSIZE / SAMPLE_RATE
        st.markdown(f"**Duration:** {total_duration:.1f}s")
    
    # Simulate audio processing (in real app, this would be connected to actual audio stream)
    if st.session_state.audio_processing:
        # Generate mock audio data for demonstration
        mock_audio = np.random.randn(BLOCKSIZE) * 0.1
        result = process_audio_chunk(mock_audio, models, denoise_blend)
        
        # Update histories
        st.session_state.emotion_history.append(result)
        st.session_state.performance_history.append(result)
        
        # Keep only last 100 entries
        if len(st.session_state.emotion_history) > 100:
            st.session_state.emotion_history = st.session_state.emotion_history[-100:]
        if len(st.session_state.performance_history) > 100:
            st.session_state.performance_history = st.session_state.performance_history[-100:]
    
    # Current emotion display
    if st.session_state.emotion_history:
        latest = st.session_state.emotion_history[-1]
        emotion_emoji = {
            'happy': '😊', 'sad': '😢', 'angry': '😠', 'neutral': '😐',
            'surprised': '😮', 'fearful': '😨', 'disgusted': '🤢'
        }.get(latest['emotion'], '😐')
        
        st.markdown(f"""
        <div class="emotion-card">
            <h2>{emotion_emoji} {latest['emotion'].title()}</h2>
            <p>Confidence: {latest['confidence']:.2%}</p>
            <p>Volume: {latest['volume']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    if st.session_state.performance_history:
        latest = st.session_state.performance_history[-1]
        
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("Denoise Time", f"{latest['denoise_time']:.2f} ms")
        with pcol2:
            st.metric("Emotion Time", f"{latest['emotion_time']:.2f} ms")
        with pcol3:
            st.metric("Total Time", f"{latest['total_time']:.2f} ms")
    
    # Visualizations
    st.subheader("📊 Real-Time Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Time Domain", "Frequency Domain", "Spectrograms", "Emotion History"])
    
    with tab1:
        if len(st.session_state.audio_buffer) > 0 and len(st.session_state.clean_buffer) > 0:
            fig = create_time_domain_plot(st.session_state.audio_buffer, st.session_state.clean_buffer)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if len(st.session_state.audio_buffer) > 0 and len(st.session_state.clean_buffer) > 0:
            fig = create_frequency_domain_plot(st.session_state.audio_buffer, st.session_state.clean_buffer)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(st.session_state.audio_buffer) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_spectrogram_plot(st.session_state.audio_buffer, "Raw Audio Spectrogram")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = create_spectrogram_plot(st.session_state.clean_buffer, "Cleaned Audio Spectrogram")
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        if st.session_state.emotion_history:
            fig = create_emotion_history_plot(st.session_state.emotion_history)
            st.plotly_chart(fig, use_container_width=True)
    
    # Processing log
    st.subheader("📝 Processing Log")
    if st.session_state.performance_history:
        # Show last 10 entries
        recent_entries = st.session_state.performance_history[-10:]
        
        log_data = []
        for entry in recent_entries:
            log_data.append({
                'Time': entry['timestamp'].strftime('%H:%M:%S'),
                'Emotion': entry['emotion'],
                'Confidence': f"{entry['confidence']:.2%}",
                'Denoise (ms)': f"{entry['denoise_time']:.2f}",
                'Emotion (ms)': f"{entry['emotion_time']:.2f}",
                'Total (ms)': f"{entry['total_time']:.2f}",
                'Silence': "Yes" if entry['is_silence'] else "No"
            })
        
        df = pd.DataFrame(log_data)
        st.dataframe(df, use_container_width=True)
    
    # Auto-refresh for real-time updates
    if st.session_state.audio_processing:
        time.sleep(0.1)  # Small delay to prevent overwhelming
        st.rerun()

if __name__ == "__main__":
    main()