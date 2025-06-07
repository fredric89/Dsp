import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
import scipy.signal

st.title("Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

st.sidebar.header("Upload Settings")
audio_file = st.sidebar.file_uploader("Upload a pre-recorded voice file (WAV/MP3)", type=["wav", "mp3"])

if audio_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # Load the audio using librosa
y, sr = librosa.load(tmp_path, sr=None, mono=True)
duration = librosa.get_duration(y=y, sr=sr)

st.audio(audio_file, format='audio/wav')
st.write(f"**Duration:** {duration:.2f} seconds")
st.write(f"**Sampling Rate:** {sr} Hz")

# --- Pre-processing functions ---
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return signal  # fallback to unfiltered if cutoff is invalid
    sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')
    return scipy.signal.sosfilt(sos, signal)

# --- Apply Pre-processing Steps ---
y = bandpass_filter(y, 80, 300, sr)               # Human speech frequency range
y = librosa.effects.preemphasis(y)                # Pre-emphasis to enhance clarity
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN or inf
y = np.clip(y, -1e3, 1e3)                         # Prevent overflow in plot

# Optional debug: display signal stats
st.write("**Signal Stats After Preprocessing**")
st.json({
    "min": float(np.min(y)),
    "max": float(np.max(y)),
    "mean": float(np.mean(y)),
    "std": float(np.std(y)),
})

    # Windowing and overlapping
    frame_length = 2048
    hop_length = 512

    # Pitch detection using autocorrelation
    def detect_pitch_autocorr(y, sr, frame_length=2048, hop_length=512):
        pitches = []
        times = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            frame = frame - np.mean(frame)
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            d = np.diff(corr)
            pos_slope = np.where(d > 0)[0]
            if len(pos_slope) == 0:
                pitches.append(0)
                times.append(i / sr)
                continue
            start = pos_slope[0]
            peak = np.argmax(corr[start:]) + start
            pitch = sr / peak if peak != 0 else 0
            pitches.append(pitch if 50 <= pitch <= 1000 else 0)
            times.append(i / sr)
        return np.array(times), np.array(pitches)

    times, f0 = detect_pitch_autocorr(y, sr, frame_length, hop_length)

    # Plot waveform and pitch contour
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, ax=ax[0], alpha=0.6)
    ax[0].set(title='Filtered Audio Waveform')
    ax[0].label_outer()

    ax[1].plot(times, f0, label='Estimated Pitch (Hz)', color='r')
    ax[1].set(title='Pitch Over Time', xlabel='Time (s)', ylabel='Pitch (Hz)')
    ax[1].grid(True)
    ax[1].legend()

    st.pyplot(fig)

    os.unlink(tmp_path)  # Clean up the temporary file

else:
    st.info("Please upload a voice recording to begin pitch detection.")

st.markdown("---")
st.markdown("**Note:** This tool uses the autocorrelation method for pitch estimation with noise filtering, bandpass range, and real-time frame analysis to ensure accuracy under typical speaking conditions.")
