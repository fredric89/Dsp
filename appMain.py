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

st.set_page_config(page_title="Voice Pitch Detector", layout="wide")

# Session state to control start screen
if 'started' not in st.session_state:
    st.session_state.started = False

# Start Screen
if not st.session_state.started:
    st.title("🎤 Voice Pitch Detection System")
    st.markdown("Developed by Group 2, National University")
    st.markdown("---")
    st.markdown("This system analyzes uploaded audio to detect and visualize voice pitch.")
    if st.button("▶️ Start"):
        st.session_state.started = True
        st.experimental_rerun()
    st.stop()

# MAIN APP STARTS HERE

st.title("🎵 Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

# Sidebar: Upload audio and filter settings
st.sidebar.header("Upload Audio File")
audio_file = st.sidebar.file_uploader("Upload a voice or tone file (WAV/MP3)", type=["wav", "mp3"])

st.sidebar.header("Bandpass Filter Settings")
lowcut = st.sidebar.slider("Lowcut Frequency (Hz)", min_value=20, max_value=500, value=50, step=10)
highcut = st.sidebar.slider("Highcut Frequency (Hz)", min_value=480, max_value=2000, value=1000, step=10)

# Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Pitch detection with autocorrelation
def autocorrelation_pitch(y, sr, frame_size, hop_size):
    num_frames = 1 + int((len(y) - frame_size) / hop_size)
    pitches = np.zeros(num_frames)
    times = np.zeros(num_frames)

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
        peak = np.argmax(autocorr[start_peak:]) + start_peak

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

# Main logic
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(audio_file, format='audio/wav')
    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Sampling Rate:** {sr} Hz")

    # Display original waveform
    fig_raw, ax_raw = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax_raw)
    ax_raw.set(title='Original Audio (Before Filtering)')
    st.pyplot(fig_raw)

    # Apply bandpass filter
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
        frame_duration = 0.03  # 30ms
        frame_size = int(sr * frame_duration)
        hop_size = frame_size // 2

        times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

        # Plot filtered waveform and pitch
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
