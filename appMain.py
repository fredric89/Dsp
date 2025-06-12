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

# Session state to manage app navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# 🎬 Landing Page
if st.session_state.page in ["home", "about"]:
    st.set_page_config(page_title="Voice Pitch Detector", layout="centered")

    # Custom CSS for gradient background with blur effect and navigation
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .nav-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 2rem;
        }
        
        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .nav-logo {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        
        .nav-link {
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            border-radius: 25px;
        }
        
        .nav-link:hover {
            color: white;
            background: rgba(255, 255, 255, 0.1);
        }
        
        .nav-link.active {
            color: white;
            background: rgba(255, 255, 255, 0.2);
        }
        
        .main-content {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 3rem 2rem;
            margin: 8rem auto 2rem;
            max-width: 600px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            text-align: center;
        }
        
        .title {
            color: white;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.2rem;
            margin-bottom: 3rem;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 3rem;
            text-align: left;
        }
        
        .feature-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .feature-title {
            color: white;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .feature-desc {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        .about-content {
            text-align: left;
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.8;
        }
        
        .about-section {
            margin-bottom: 2rem;
        }
        
        .about-title {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("""
        <div class="nav-container">
            <div class="nav-content">
                <div class="nav-logo">🎶 Voice Pitch Detector</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with col4:
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()

    if st.session_state.page == "home":
        # Home page content
        st.markdown("""
            <div class="main-content">
                <div class="title">🎶 Voice Pitch Detector</div>
                <div class="subtitle">Advanced Audio Analysis Tool</div>
                
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-icon">🎯</div>
                        <div class="feature-title">Accurate Detection</div>
                        <div class="feature-desc">Real-time pitch detection using autocorrelation algorithms</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🔧</div>
                        <div class="feature-title">Custom Filtering</div>
                        <div class="feature-desc">Adjustable bandpass filters for precise analysis</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Visual Analysis</div>
                        <div class="feature-desc">Interactive waveform and pitch visualization</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">📈</div>
                        <div class="feature-title">Statistics</div>
                        <div class="feature-desc">Comprehensive pitch statistics and metrics</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Center the start button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Start Analysis", key="start_btn", use_container_width=True):
                st.session_state.page = "app"
                st.rerun()
    
    elif st.session_state.page == "about":
        # About page content
        st.markdown("""
            <div class="main-content">
                <div class="title">About This Project</div>
                
                <div class="about-content">
                    <div class="about-section">
                        <div class="about-title">🎓 Academic Project</div>
                        <p>This Voice Pitch Detection application was developed by <strong>Group 2</strong> from <strong>National University</strong> as part of our signal processing and audio analysis coursework.</p>
                    </div>
                    
                    <div class="about-section">
                        <div class="about-title">🔬 Technical Implementation</div>
                        <p>The application uses advanced digital signal processing techniques including:</p>
                        <ul>
                            <li><strong>Autocorrelation Algorithm:</strong> For accurate pitch estimation</li>
                            <li><strong>Butterworth Bandpass Filtering:</strong> To isolate frequency ranges</li>
                            <li><strong>Librosa Library:</strong> For professional audio processing</li>
                            <li><strong>Real-time Visualization:</strong> Using Matplotlib for data representation</li>
                        </ul>
                    </div>
                    
                    <div class="about-section">
                        <div class="about-title">🎯 Applications</div>
                        <p>This tool can be used for:</p>
                        <ul>
                            <li>Speech analysis and linguistics research</li>
                            <li>Music education and vocal training</li>
                            <li>Audio quality assessment</li>
                            <li>Acoustic research and analysis</li>
                        </ul>
                    </div>
                    
                    <div class="about-section">
                        <div class="about-title">💻 Technology Stack</div>
                        <p>Built using Python with Streamlit, NumPy, SciPy, Librosa, and Matplotlib for a comprehensive audio analysis experience.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Back to Home button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🏠 Back to Home", key="back_home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
    
    st.stop()

# 🟢 MAIN APP STARTS HERE
elif st.session_state.page == "app":
    st.set_page_config(page_title="Voice Pitch Detection", layout="wide")

# Custom CSS for main app
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    .main-subtitle {
        color: rgba(255, 255, 255, 0.8);
        text-align: center;
        margin: 0;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎵 Voice Pitch Detection and Visualization</h1>
        <p class="main-subtitle">Developed by Group 2, National University</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar: Upload and filter settings
st.sidebar.header("📁 Upload Audio File")
audio_file = st.sidebar.file_uploader("Upload a voice or tone file (WAV/MP3)", type=["wav", "mp3"])

st.sidebar.header("🔧 Bandpass Filter Settings")
lowcut = st.sidebar.slider("Lowcut Frequency (Hz)", min_value=20, max_value=500, value=50, step=10)
highcut = st.sidebar.slider("Highcut Frequency (Hz)", min_value=480, max_value=2000, value=1000, step=10)

# Add reset button in sidebar
if st.sidebar.button("🔄 Back to Home"):
    st.session_state.page = "home"
    st.rerun()

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

# Autocorrelation pitch detection (Fixed)
def autocorrelation_pitch(y, sr, frame_size, hop_size):
    num_frames = 1 + int((len(y) - frame_size) / hop_size)
    pitches = np.zeros(num_frames)
    times = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        frame = y[start:start+frame_size]
        
        if np.all(frame == 0):
            pitches[i] = 0
            times[i] = start / sr
            continue
            
        frame -= np.mean(frame)
        autocorr = np.correlate(frame, frame, mode='full')[frame_size:]
        
        # Find the first peak after the zero lag
        d = np.diff(autocorr)
        start_peak_candidates = np.where(d > 0)[0]
        
        if start_peak_candidates.size == 0:
            pitches[i] = 0
            times[i] = start / sr
            continue
            
        start_peak = start_peak_candidates[0]
        
        # Find peaks in the autocorrelation
        peaks = []
        for j in range(start_peak, len(autocorr) - 1):
            if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1]:
                peaks.append(j)
        
        if peaks and autocorr[peaks[0]] > 0:
            pitch = sr / peaks[0]
            pitches[i] = pitch if 50 < pitch < 1000 else 0
        else:
            pitches[i] = 0
            
        times[i] = start / sr

    # Interpolate missing values
    if np.any(pitches > 0):
        valid = pitches > 0
        if np.sum(valid) > 1:  # Need at least 2 points for interpolation
            interp = interp1d(times[valid], pitches[valid], kind='linear', fill_value='extrapolate')
            pitches = interp(times)
            
    return times, pitches

# Main Logic
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Audio info
        col1, col2 = st.columns(2)
        with col1:
            st.audio(audio_file, format='audio/wav')
        with col2:
            st.metric("Duration", f"{duration:.2f} seconds")
            st.metric("Sampling Rate", f"{sr} Hz")

        # Original waveform
        st.subheader("📊 Original Audio Waveform")
        fig_raw, ax_raw = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax_raw)
        ax_raw.set_title('Original Audio (Before Filtering)')
        ax_raw.grid(True, alpha=0.3)
        st.pyplot(fig_raw)
        plt.close(fig_raw)

        # Apply bandpass filter
        y_filtered = bandpass_filter(y, lowcut, highcut, sr)
        y_filtered = np.nan_to_num(y_filtered)

        # Normalize filtered signal
        if np.max(np.abs(y_filtered)) > 1e-5:
            y_filtered /= np.max(np.abs(y_filtered))

        # Save filtered audio
        filtered_path = tmp_path.replace(".wav", "_filtered.wav")
        sf.write(filtered_path, y_filtered, sr)
        
        st.subheader("🔧 Filtered Audio")
        st.audio(filtered_path, format='audio/wav')

        if np.all(np.abs(y_filtered) < 1e-5):
            st.warning("⚠️ Filtered signal is too quiet or empty. Try adjusting the bandpass filter range.")
        else:
            # Pitch detection
            frame_duration = 0.03  # 30ms frames
            frame_size = int(sr * frame_duration)
            hop_size = frame_size // 2
            
            with st.spinner("🔍 Analyzing pitch..."):
                times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

            # Visualization
            st.subheader("📈 Pitch Analysis Results")
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
            
            # Filtered waveform
            librosa.display.waveshow(y_filtered, sr=sr, ax=ax[0])
            ax[0].set_title('Filtered Audio Waveform')
            ax[0].grid(True, alpha=0.3)
            ax[0].label_outer()

            # Pitch contour
            ax[1].plot(times, pitches, label='Estimated Pitch (Hz)', color='red', linewidth=2)
            ax[1].set_title('Pitch Over Time')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Pitch (Hz)')
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            ax[1].set_ylim(0, max(1000, np.max(pitches) * 1.1) if np.any(pitches > 0) else 1000)

            st.pyplot(fig)
            plt.close(fig)

            # Statistics
            if np.any(pitches > 0):
                valid_pitches = pitches[pitches > 0]
                avg_pitch = np.mean(valid_pitches)
                min_pitch = np.min(valid_pitches)
                max_pitch = np.max(valid_pitches)
                std_pitch = np.std(valid_pitches)
                
                st.subheader("📊 Pitch Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Pitch", f"{avg_pitch:.2f} Hz")
                with col2:
                    st.metric("Min Pitch", f"{min_pitch:.2f} Hz")
                with col3:
                    st.metric("Max Pitch", f"{max_pitch:.2f} Hz")
                with col4:
                    st.metric("Std Deviation", f"{std_pitch:.2f} Hz")
            else:
                st.error("❌ No valid pitch detected. Try uploading a clearer audio sample or adjusting the filter settings.")

        # Cleanup
        os.unlink(tmp_path)
        if os.path.exists(filtered_path):
            os.unlink(filtered_path)
            
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
else:
    st.info("📁 Please upload an audio file to start pitch detection.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <strong>💡 Tips:</strong> Use clear recordings for better pitch detection. 
        Adjust filter sliders to focus on specific frequency ranges.
        <br><br>
        <em>Developed with ❤️ by Group 2, National University</em>
    </div>
""", unsafe_allow_html=True)
