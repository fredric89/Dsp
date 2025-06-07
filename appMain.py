import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import soundfile as sf

st.title("Enhanced Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

st.sidebar.header("Upload Settings")
audio_file = st.sidebar.file_uploader("Upload a pre-recorded voice file (WAV/MP3)", type=["wav", "mp3"])

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def autocorrelation_pitch(y, sr, frame_size, hop_size):
    num_frames = 1 + int((len(y) - frame_size) / hop_size)
    pitches = np.zeros(num_frames)
    times = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        frame = y[start:start+frame_size]
        frame = frame - np.mean(frame)
        autocorr = np.correlate(frame, frame, mode='full')[frame_size:]

        d = np.diff(autocorr)
        rising_edges = np.where(d > 0)[0]

        if rising_edges.size == 0:
            pitch = 0
        else:
            start_peak = rising_edges[0]
            peak = np.argmax(autocorr[start_peak:]) + start_peak
            if autocorr[peak] > 0:
                pitch = sr / peak
            else:
                pitch = 0

        pitches[i] = pitch if 50 < pitch < 1000 else 0
        times[i] = start / sr

    non_zero = pitches > 0
    if np.sum(non_zero) > 1:
        interp_func = interp1d(times[non_zero], pitches[non_zero], kind='linear', fill_value='extrapolate')
        pitches = interp_func(times)

    return times, pitches

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(audio_file, format='audio/wav')
    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Sampling Rate:** {sr} Hz")

    # 🎧 Show original waveform before filtering
    fig_orig, ax_orig = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax_orig, alpha=0.6)
    ax_orig.set(title="Original Audio Waveform")
    st.pyplot(fig_orig)

    # 🧹 Preprocessing
    y_filtered = bandpass_filter(y, 60, 300, sr, order=4)
    y_filtered = np.nan_to_num(y_filtered, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize only if signal has energy
    max_val = np.max(np.abs(y_filtered))
    if max_val > 0.01:
        y_filtered = y_filtered / max_val

    # Optional: Play filtered audio
    filtered_path = tmp_path.replace(".wav", "_filtered.wav")
    sf.write(filtered_path, y_filtered, sr)
    st.markdown("**Filtered Audio Playback:**")
    st.audio(filtered_path, format='audio/wav')

    # 🎯 Pitch Detection
    frame_duration = 0.03  # 30 ms
    frame_size = int(sr * frame_duration)
    hop_size = frame_size // 2

    times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

    # 📊 Final Plot
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    librosa.display.waveshow(y_filtered, sr=sr, ax=ax[0], alpha=0.6)
    ax[0].set(title='Filtered Audio Waveform')
    ax[0].label_outer()

    ax[1].plot(times, pitches, label='Estimated Pitch (Hz)', color='r')
    ax[1].set(title='Pitch Over Time (Autocorrelation)', xlabel='Time (s)', ylabel='Pitch (Hz)')
    ax[1].grid(True)
    ax[1].legend()

    st.pyplot(fig)

    os.unlink(tmp_path)

else:
    st.info("Please upload a voice recording to begin pitch detection.")

st.markdown("---")
st.markdown("""
**System Workflow Summary:**

- 🎙️ **Input**: Pre-recorded voice from a microphone or file  
- 🎛️ **Pre-processing**: Mono conversion, bandpass filtering (60–300 Hz), noise cleanup  
- 🔍 **Processing**: Framing (30ms), autocorrelation pitch detection, interpolation  
- 📊 **Output**: Waveform and pitch-over-time graph

**Note:** You now see both the original and filtered waveform for better comparison.
""")
