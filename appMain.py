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

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page in ["home", "about"]:
    st.set_page_config(page_title="Voice Pitch Detector", layout="centered")

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
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .nav-logo {
            color: white;
            font-size: 1.8rem;
            font-weight: bold;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.6);
        }
        .main-content {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 3rem 2rem;
            margin: 8rem auto 2rem;
            max-width: 600px;
            box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.4);
            text-align: center;
        }
        .title {
            color: white;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.6);
        }
        .subtitle {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.2rem;
            margin-bottom: 3rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="nav-container">
            <div class="nav-content">
                <div class="nav-logo">🎵 PitchScope: Your Voice Frequency Visualizer</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.page == "home":
        st.markdown("""
            <div class="main-content">
                <div class="title">Voice Pitch Detector</div>
                <div class="subtitle">Analyze and visualize your voice pitch with ease.</div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start Analysis", key="start_btn", use_container_width=True):
                st.session_state.page = "app"
                st.rerun()

        st.markdown("""
            <style>
            div.stButton > button:first-child {
                font-size: 1.5rem;
                padding: 1rem 2rem;
                border-radius: 12px;
                background-color: #ffffff;
                color: #222222;
                font-weight: bold;
                border: none;
            }
            div.stButton > button:first-child:hover {
                background-color: #dddddd;
            }
            </style>
        """, unsafe_allow_html=True)

    elif st.session_state.page == "about":
        st.markdown("""
            <div class="main-content">
                <div class="title">About</div>
                <div class="about-content">
                    <p>This tool was developed by Group 2 from National University as part of our coursework in audio signal processing.</p>
                    <p>It uses Python, Streamlit, and Librosa for real-time pitch detection and visualization.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Back to Home", key="back_home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()

    st.stop()
