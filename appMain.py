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

st.title("Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

st.sidebar.header("Upload Settings")
audio_file = st.sidebar.file_uploader("Upload a pre-recorded voice file (WAV/MP3)", type=["wav", "mp3"])

st.sidebar.header("Testing Parameters")
expected_pitch = st.sidebar.number_input("Expected Pitch (Hz)", min_value=20.0, max_value=2000.0, step=1.0)
margin_type = st.sidebar.radio("Select Accuracy Margin", ["Music (±1 Hz)", "Speech (±5 Hz)"])
margin = 1 if "Music" in margin_type else 5

# Bandpass filter
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

    # Interpolate missing values
    if np.any(pitches > 0):
        valid = pitches > 0
        interp = interp1d(times[valid], pitches[valid], kind='linear', fill_value='extrapolate')
        pitches = interp(times)
    return times, pitches

# Accuracy calculation
def calculate_accuracy(detected_pitches, ground_truth_freq, margin):
    valid = detected_pitches > 0
    correct = np.sum(np.abs(detected_pitches[valid] - ground_truth_freq) <= margin)
    total = np.sum(valid)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(audio_file, format='audio/wav')
    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Sampling Rate:** {sr} Hz")

    # Plot original waveform
    fig_raw, ax_raw = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax_raw)
    ax_raw.set(title='Original Audio (Before Filtering)')
    st.pyplot(fig_raw)

    # Bandpass filter
    y_filtered = bandpass_filter(y, lowcut=50, highcut=1000, fs=sr, order=4)
    y_filtered = np.nan_to_num(y_filtered)

    if np.max(np.abs(y_filtered)) > 1e-5:
        y_filtered /= np.max(np.abs(y_filtered))

    # Save and play filtered audio
    filtered_path = tmp_path.replace(".wav", "_filtered.wav")
    sf.write(filtered_path, y_filtered, sr)
    st.audio(filtered_path, format='audio/wav')

    if np.all(np.abs(y_filtered) < 1e-5):
        st.warning("⚠️ Filtered signal is too quiet or empty. Try loosening the filter or checking the recording.")
    else:
        frame_duration = 0.03  # 30 ms
        frame_size = int(sr * frame_duration)
        hop_size = frame_size // 2

        times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

        # Plot waveform and pitch
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
        librosa.display.waveshow(y_filtered, sr=sr, ax=ax[0])
        ax[0].set(title='Filtered Audio Waveform')
        ax[0].label_outer()

        ax[1].plot(times, pitches, label='Estimated Pitch (Hz)', color='r')
        ax[1].set(title='Pitch Over Time (Autocorrelation)', xlabel='Time (s)', ylabel='Pitch (Hz)')
        ax[1].legend()
        ax[1].grid(True)

        st.pyplot(fig)

        # Accuracy and average pitch
        if np.any(pitches > 0):
            avg_pitch = np.mean(pitches[pitches > 0])
            accuracy = calculate_accuracy(pitches, expected_pitch, margin)

            st.markdown("### 🧪 Pitch Detection Results")
            st.write(f"**Average Detected Pitch:** {avg_pitch:.2f} Hz")
            st.write(f"**Expected Pitch:** {expected_pitch:.2f} Hz")
            st.write(f"**Accuracy (±{margin} Hz):** {accuracy:.2f}%")
        else:
            st.warning("❌ No valid pitch detected. Please check the audio signal or try a different sample.")

    os.unlink(tmp_path)
else:
    st.info("Please upload a voice recording to begin pitch detection.")

st.markdown("---")
st.markdown("**Tip:** If the waveform is flat or there's no audio, check your recording or try a different voice sample.")
