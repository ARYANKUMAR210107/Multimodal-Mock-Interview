"""
=============================================================================
InterviewAI — Intelligent Multimodal Mock Interview Assessment System
Frontend Application Entrypoint & Modular Router
=============================================================================
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
import streamlit as st

# Setup path resolution
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
ROOT_DIR = PROJECT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Page icon resolution
_icon_file = BASE_DIR / "assets" / "interviewai_icon.png"
_fav_icon = str(_icon_file) if _icon_file.exists() else "🎙️"

# Configure wide layout and page metadata
st.set_page_config(
    page_title="InterviewAI · Multimodal Mock Interview Platform",
    page_icon=_fav_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")
load_dotenv()

# Import Modular Page Renderers
from frontend.pages.dashboard import render_dashboard
from frontend.pages.students import render_students
from frontend.pages.new_assessment import render_new_assessment
from frontend.pages.interview_workspace import render_interview_workspace
from frontend.pages.assessment_history import render_assessment_history
from frontend.pages.analytics import render_analytics
from frontend.pages.question_bank import render_question_bank
from frontend.pages.system_status import render_system_status
from frontend.pages.auth_page import render_login, render_logout_button
from frontend.services.api_client import client


def load_theme():
    """Inject the unified design system CSS with dark/light mode tokens."""
    css_path = BASE_DIR / "styles" / "theme.css"
    if not css_path.exists():
        css_path = BASE_DIR / "theme.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        theme_class = "light-theme" if st.session_state.get("theme_mode") == "light" else "dark-theme"
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        if theme_class == "light-theme":
            st.markdown("<script>document.documentElement.setAttribute('data-theme', 'light');</script>", unsafe_allow_html=True)


def nav_to(page_name: str, **kwargs):
    """Centralized navigation dispatcher."""
    st.session_state["nav"] = page_name
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


def render_sidebar():
    """Render the clean 10-item navigation sidebar with branding & active context."""
    with st.sidebar:
        # 1. Branding Header
        logo_path = BASE_DIR / "assets" / "interviewai_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
            st.markdown("<hr style='margin: 0.35rem 0 0.75rem; border-color: var(--iai-border);'>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="nav-brand">
                    <div class="nav-mark">IA</div>
                    <div>
                        <div class="nav-brand-name">InterviewAI</div>
                        <div class="nav-brand-sub">Mock Interview Assessment</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        current_nav = st.session_state.get("nav", "Dashboard")
        user_role = st.session_state.get("user", {}).get("role", "student")

        # 2. Main Navigation (kept simple per UX requirements)
        st.markdown('<div class="nav-group-title">MAIN NAVIGATION</div>', unsafe_allow_html=True)

        if user_role == "admin":
            main_nav_items = [
                ("Dashboard", "Dashboard"),
                ("Start Interview", "Start Interview"),
                ("My Assessments", "My Assessments"),
                ("My Reports", "My Reports"),
                ("My Profile", "My Profile"),
            ]
        else:
            main_nav_items = [
                ("Dashboard", "Dashboard"),
                ("Start Interview", "Start Interview"),
                ("My Assessments", "My Assessments"),
                ("My Reports", "My Reports"),
                ("My Profile", "My Profile"),
            ]

        for page_id, label in main_nav_items:
            is_active = (current_nav == page_id)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_btn_{page_id}", type=btn_type, use_container_width=True):
                if current_nav != page_id:
                    nav_to(page_id)

        # 4. Active Interview Session Context Card
        active_sess_id = st.session_state.get("active_interview_session_id")
        if active_sess_id:
            st.write("")
            with st.container(border=True):
                st.markdown('<div style="font-size: 0.72rem; color: var(--iai-secondary); font-weight: 750; letter-spacing: 0.1em;">ACTIVE SESSION</div>', unsafe_allow_html=True)
                ok, sess = client.get_session(active_sess_id)
                if ok:
                    s_name = sess.get("student", {}).get("full_name", "Candidate")
                    st.markdown(f"**{s_name}**")
                    st.caption(f"{sess.get('domain')} · {sess.get('question_count')} Questions")
                    if current_nav != "Active Workspace":
                        if st.button("Enter Workspace ➔", key="btn_side_workspace", type="primary", use_container_width=True):
                            nav_to("Active Workspace")
                else:
                    st.caption(f"ID: `{active_sess_id}`")
                    st.caption(f"⚠️ {sess if isinstance(sess, str) else ''}")

        # 5. Footer / Logout
        st.write("")
        st.markdown("<hr style='margin: 0.75rem 0; border-color: var(--iai-border);'>", unsafe_allow_html=True)
        if st.session_state.get("is_authenticated", False):
            render_logout_button()
        else:
            if st.button("Sign In", key="btn_sidebar_login", type="primary", use_container_width=True):
                nav_to("Login")

        st.write("")
        t_col1, t_col2 = st.columns([1.5, 1])
        with t_col1:
            st.caption("Theme Mode")
        with t_col2:
            current_theme = st.session_state.get("theme_mode", "dark")
            theme_btn_label = "☀️ Light" if current_theme == "dark" else "🌙 Dark"
            if st.button(theme_btn_label, key="btn_theme_toggle", use_container_width=True):
                st.session_state["theme_mode"] = "light" if current_theme == "dark" else "dark"
                st.rerun()

        st.caption("InterviewAI v2.0 · Final Year B.Tech")


def render_standalone_multimodal_report():
    """Renderer for direct or test-driven Multimodal Report sessions."""
    report = st.session_state.get("voice_report")
    question = st.session_state.get("voice_question", "")
    digest = st.session_state.get("voice_audio_digest", "")
    transcript = st.session_state.get("voice_transcript_editor", "")

    raw_ctx = "\0".join([question or "", digest or "", transcript or ""])
    expected_context = hashlib.sha256(raw_ctx.encode()).hexdigest()
    current_context = st.session_state.get("voice_report_context")

    if current_context and current_context != expected_context:
        st.warning("The recording or question was changed after this report was generated.")
        return

    if not report:
        st.info("No completed report yet.")
        return

    score = report.get("text_assessment_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        score_val = f"{float(score):.1%}"
    else:
        score_val = "Not available"

    st.markdown(f"### {question}")
    st.metric("Text-only practice score", score_val)


def main():
    """Main routing controller."""
    load_theme()
    render_sidebar()

    is_authenticated = st.session_state.get("is_authenticated", False)

    # Redirect to login if not authenticated (except on Login page)
    current_page = st.session_state.get("nav", "Dashboard")
    if not is_authenticated and current_page != "Login":
        render_login(nav_to)
        return

    # Route Dispatcher with role-based navigation and compatibility aliases
    if current_page == "Login":
        render_login(nav_to)
    elif current_page in ("Dashboard", "Admin Dashboard", "dashboard"):
        render_dashboard(nav_to)
    elif current_page in ("Students", "Student Directory", "Student Registration", "Student Profile", "Student Management", "My Profile"):
        tab_name = "Directory"
        if current_page == "Student Registration":
            tab_name = "Register"
        elif current_page in ("Student Profile", "My Profile"):
            tab_name = "Profile"
        elif current_page == "Student Management":
            tab_name = "Directory"
        render_students(nav_to, initial_tab=tab_name)
    elif current_page in ("New Assessment", "Create Interview", "Start Interview"):
        render_new_assessment(nav_to)
    elif current_page in ("Active Workspace", "Interview Workspace", "Text Interview", "Voice Interview", "Active Sessions", "Assessment Monitoring"):
        render_interview_workspace(nav_to)
    elif current_page in ("Assessment History", "Assessment Reports", "Interview History", "My Assessments", "My Reports"):
        render_assessment_history(nav_to)
    elif current_page == "Multimodal Report":
        render_standalone_multimodal_report()
    elif current_page in ("Analytics", "Exploratory Data Analysis", "Model Analytics"):
        render_analytics(nav_to)
    elif current_page in ("Question Bank",):
        render_question_bank(nav_to)
    elif current_page in ("System Health", "API & System Status", "API Status", "About Project"):
        render_system_status(nav_to)
    else:
        render_dashboard(nav_to)


if __name__ == "__main__":
    main()
