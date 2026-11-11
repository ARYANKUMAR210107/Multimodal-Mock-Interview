"""
Settings & Platform Configuration Page
======================================
Application preferences, API endpoint configuration, UI customization,
and system runtime parameters. Zero credential exposure.
"""
from __future__ import annotations

import os
import streamlit as st
from frontend.components.ui import page_header, section_header, status_pill
from frontend.services.api_client import client


def render_settings(nav_to_fn):
    page_header(
        kicker="Preferences & Config",
        title="Settings & Platform Configuration",
        description="Configure application preferences, audio recording parameters, API endpoints, and view environment details.",
    )

    health = client.get_system_health()

    c1, c2 = st.columns(2, gap="large")

    with c1, st.container(border=True):
        section_header("Appearance & Display", "Customize the user interface look and feel")
        
        current_theme = st.session_state.get("theme_mode", "dark")
        new_theme = st.radio(
            "Color Theme",
            options=["dark", "light"],
            index=0 if current_theme == "dark" else 1,
            format_func=lambda x: "🌙 Dark Mode (Default Navy & Cyan)" if x == "dark" else "☀️ Light Mode (Clean Slate)",
            key="settings_theme_radio"
        )
        if new_theme != current_theme:
            st.session_state["theme_mode"] = new_theme
            st.rerun()

        st.caption("Active palette: Electric Cyan (#06B6D4) and Electric Blue (#3B82F6) on deep navy.")

        st.write("")
        section_header("Audio & Workspace Settings", "Interview recording parameters")
        
        sample_rate = st.selectbox(
            "Target Audio Sample Rate",
            options=[16000, 22050, 44100],
            index=1,
            help="Librosa feature extractor resamples input audio to 22,050 Hz for acoustic analysis."
        )
        
        max_duration = st.slider(
            "Max Answer Recording Time (seconds)",
            min_value=30,
            max_value=300,
            value=120,
            step=15,
            help="Default limit for candidate voice response recordings."
        )
        st.caption(f"Configured for up to {max_duration} seconds ({max_duration // 60}m {max_duration % 60}s) per question response.")

    with c2, st.container(border=True):
        section_header("Network & Service Endpoints", "Microservice connection configuration")

        backend_url = os.getenv("MOCK_INTERVIEW_API_URL", "http://127.0.0.1:8000")
        st.text_input("Backend API Base URL", value=backend_url, disabled=True, help="Set via MOCK_INTERVIEW_API_URL environment variable.")
        
        backend_status = "online" if health.get("online") else "offline"
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                <span style="font-size: 0.88rem; color: var(--iai-muted);">Connection Status:</span>
                <span>{status_pill(backend_status, 'Connected' if backend_status == 'online' else 'Unreachable')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        section_header("Storage & Persistence", "Database and media directory status")
        
        st.markdown(
            """
            - **SQLite Database:** `data/interviewai.sqlite3`
            - **Audio Recording Directory:** `data/audio_recordings/`
            - **ML Model Directory:** `models/`
            - **Dataset Repository:** `dataset/`
            """
        )

        st.write("")
        section_header("About InterviewAI", "Build and version information")
        st.markdown(
            """
            **InterviewAI — Intelligent Multimodal Mock Interview Assessment System**  
            - **Version:** `2.0.0 (Production Architecture)`  
            - **Architecture:** FastAPI Microservice + Streamlit Enterprise Frontend  
            - **Speech-to-Text:** Google Gemini Multimodal STT  
            - **NLP Semantic Engine:** TF-IDF Cosine Similarity & Domain Keyword Matching  
            - **Acoustic Profiler:** Librosa Pitch (YIN), Energy (RMS), Rate, MFCCs  
            - **Employability Classifier:** Gradient Boosting Classifier  
            """
        )

