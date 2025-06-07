import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
import soundfile as sf
import scipy.signal

st.title("🎤 Voice Pitch Detection and Visualization")
st.markdown("Developed by Group 2, National University")

# === AUDIO INPUT SECTION ===
st.sidebar.header("Audio Input")
audio_file = st.sidebar.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(tmp_path, format='audio/wav')
    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Sampling Rate:** {sr} Hz")

    # === PREPROCESSING SECTION ===
    def bandpass_filter(signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        if low >= high:
            return signal
        sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')
        return scipy.signal.sosfilt(sos, signal)

    def normalize_audio(signal):
        return signal / np.max(np.abs(signal))

    y = bandpass_filter(y, 80, 300, sr)
    y = librosa.effects.preemphasis(y)
    y = normalize_audio(y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1e3, 1e3)

    st.write("**Signal Stats After Preprocessing**")
    st.json({
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
    })

    # === PITCH DETECTION SECTION ===
    frame_length = 2048
    hop_length = 512

    pitch_method = st.sidebar.selectbox("Pitch Detection Method", ["Autocorrelation", "YIN"])

    if pitch_method == "Autocorrelation":
        def detect_pitch_autocorr(y, sr):
            pitches, times = [], []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                frame -= np.mean(frame)
                corr = np.correlate(frame, frame, mode='full')[len(frame)-1:]
                d = np.diff(corr)
                pos = np.where(d > 0)[0]
                if len(pos) == 0:
                    pitches.append(0)
                else:
                    start = pos[0]
                    peak = np.argmax(corr[start:]) + start
                    pitch = sr / peak if peak > 0 else 0
                    pitches.append(pitch if 50 <= pitch <= 1000 else 0)
                times.append(i / sr)
            return np.array(times), np.array(pitches)

        times, f0 = detect_pitch_autocorr(y, sr)

    else:  # YIN method
        f0 = librosa.yin(y, fmin=50, fmax=1000, sr=sr, frame_length=frame_length, hop_length=hop_length)
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # === VISUALIZATION SECTION ===
    st.subheader("🔍 Audio Analysis")

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(12, 8))

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0], alpha=0.6)
    ax[0].set(title="Filtered Waveform")

    # Pitch over time
    ax[1].plot(times, f0, color='r', label="Pitch (Hz)")
    ax[1].set(title="Pitch Over Time", ylabel="Hz")
    ax[1].legend()
    ax[1].grid(True)

    # Spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax[2])
    ax[2].set(title="Spectrogram")
    fig.colorbar(img, ax=ax[2], format="%+2.0f dB")

    st.pyplot(fig)

    # Cleanup
    os.unlink(tmp_path)

else:
    st.info("Please upload a WAV or MP3 audio file to start analysis.")

st.markdown("---")
st.markdown("**Note:** Supports autocorrelation and YIN pitch estimation, file upload only, and visual feedback via waveform, pitch line, and spectrogram.")
