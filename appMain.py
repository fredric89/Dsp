import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
from scipy.signal import butter, lfilter
import soundfile as sf
from scipy.interpolate import interp1d

# Session state to manage app start
if "started" not in st.session_state:
    st.session_state.started = False

# 🎬 Landing Page
if not st.session_state.started:
    st.set_page_config(page_title="Voice Pitch Detector", layout="centered")

    # Custom CSS for blur effect and layout
    st.markdown("""
        <style>
        .blurred-background {
            background-image: url('https://your-image-url-here.jpg');
            background-size: cover;
            background-position: center;
            filter: blur(8px);
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        .content-box {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            margin-top: 100px;
        }
        </style>
        <div class="blurred-background"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.title("🎶 Voice Pitch Detection App")
    st.image("landing.png", use_column_width=True)

    st.markdown("""
    ### 👋 Welcome!
    This app allows you to upload a voice or tone recording and visualize its pitch over time using signal processing techniques.

    #### 📌 About the App
    - Developed by **Group 2, National University**
    - Uses **bandpass filtering** and **autocorrelation** to estimate pitch
    - Visualizes both the waveform and pitch contour
    - Ideal for speech analysis, music studies, and acoustic research

    Click the button below to get started!
    """)
    if st.button("▶️ Start"):
        st.session_state.started = True
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# 🟢 MAIN APP STARTS HERE
st.set_page_config(page_title="Voice Pitch Detection", layout="wide")
st.title("🎵 Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

# Sidebar: Upload and filter settings
st.sidebar.header("Upload Audio File")
audio_file = st.sidebar.file_uploader("Upload a voice or tone file (WAV/MP3)", type=["wav", "mp3"])

st.sidebar.header("Bandpass Filter Settings")
lowcut = st.sidebar.slider("Lowcut Frequency (Hz)", min_value=20, max_value=500, value=50, step=10)
highcut = st.sidebar.slider("Highcut Frequency (Hz)", min_value=480, max_value=2000, value=1000, step=10)

# Filtering functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Autocorrelation pitch detection
def autocorrelation_pitch(y, sr, frame_size, hop_size):
    num_frames = 1 + int((len(y) - frame_size) / hop_size)
    for i in range(num_frames):
        start = i * hop_size
        frame = y[start:start+frame_size]
        if np.all(frame == 0):
            continue
        frame -= np.mean(frame)
        autocorr = np.correlate(frame, frame, mode='full')[frame_size:]
        d = np.diff(autocorr)
        start_peak_candidates = np.where(d > 0)[0]
        if start_peak_candidates.size == 0:
            continue
        start_peak = start_peak_candidates[0]
        start_peak
        if autocorr[peak] > 0:
            pitch = sr / peak
        else:
            pitch = 0
        pitches[i] = pitch if 50 < pitch < 1000 else 0
        times[i] = start / sr

    if np.any(pitches > 0):
        valid = pitches > 0
        interp = interp1d(times[valid], pitches[valid], kind='linear', fill_value='extrapolate')
        pitches = interp(times)
    return times, pitches

# Main Logic
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(audio_file, format='audio/wav')
    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Sampling Rate:** {sr} Hz")

    fig_raw, ax_raw = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax_raw)
    ax_raw.set(title='Original Audio (Before Filtering)')
    st.pyplot(fig_raw)

    y_filtered = bandpass_filter(y, lowcut, highcut, sr)
    y_filtered = np.nan_to_num(y_filtered)

    if np.max(np.abs(y_filtered)) > 1e-5:
        y_filtered /= np.max(np.abs(y_filtered))

    filtered_path = tmp_path.replace(".wav", "_filtered.wav")
    sf.write(filtered_path, y_filtered, sr)
    st.audio(filtered_path, format='audio/wav')

    if np.all(np.abs(y_filtered) < 1e-5):
        st.warning("⚠️ Filtered signal is too quiet or empty. Adjust the bandpass filter range.")
    else:
        frame_duration = 0.03
        frame_size = int(sr * frame_duration)
        hop_size = frame_size // 2
        times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
        librosa.display.waveshow(y_filtered, sr=sr, ax=ax[0])
        ax[0].set(title='Filtered Audio Waveform')
        ax[0].label_outer()

        ax[1].plot(times, pitches, label='Estimated Pitch (Hz)', color='r')
        ax[1].set(title='Pitch Over Time', xlabel='Time (s)', ylabel='Pitch (Hz)')
        ax[1].legend()
        ax[1].grid(True)

        st.pyplot(fig)

        if np.any(pitches > 0):
            avg_pitch = np.mean(pitches[pitches > 0])
            st.markdown("### 🎯 Average Estimated Pitch")
            st.write(f"**{avg_pitch:.2f} Hz**")
        else:
            st.warning("❌ No valid pitch detected. Try uploading a clearer audio sample.")

    os.unlink(tmp_path)
else:
    st.info("📁 Please upload an audio file to start pitch detection.")

st.markdown("---")
st.markdown("**Tip:** Use clear recordings for better pitch detection. Adjust filter sliders to focus on specific frequencies.")
