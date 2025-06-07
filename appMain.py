import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

st.title("Enhanced Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

st.sidebar.header("Upload Settings")
audio_file = st.sidebar.file_uploader("Upload a pre-recorded voice file (WAV/MP3)", type=["wav", "mp3"])

# Bandpass filter for human voice
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Autocorrelation-based pitch detection
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
        start_peak = np.where(d > 0)[0][0]
        peak = np.argmax(autocorr[start_peak:]) + start_peak
        
        if autocorr[peak] > 0:
            pitch = sr / peak
        else:
            pitch = 0

        pitches[i] = pitch if 50 < pitch < 1000 else 0  # human voice range
        times[i] = start / sr

    # Interpolation of unvoiced segments
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

    # Preprocessing
    y_filtered = bandpass_filter(y, 85, 255, sr, order=6)

    frame_duration = 0.03  # 30 ms
    frame_size = int(sr * frame_duration)
    hop_size = frame_size // 2

    times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

    # Plot waveform and pitch contour
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

- 🎙️ Input: Pre-recorded voice from a microphone or file  
- 🎛️ Pre-processing: Mono conversion, bandpass filtering, noise reduction  
- 🔍 Processing: Framing, pitch detection using autocorrelation  
- 📊 Output: Real-time-like visualization of waveform and pitch contour

**Note:** Silent frames are interpolated to ensure pitch continuity.
""")
