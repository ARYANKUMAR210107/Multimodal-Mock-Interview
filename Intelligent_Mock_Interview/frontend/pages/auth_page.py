"""
Simple, professional authentication UI for InterviewAI.
"""
from __future__ import annotations

import streamlit as st

from frontend.services.api_client import client


def render_login(nav_to):
    """Render centered login/registration experience."""
    if st.session_state.get("is_authenticated"):
        st.session_state["nav"] = "Dashboard"
        st.rerun()
        return

    mode = st.session_state.get("auth_mode", "login")
    is_login = mode == "login"

    # Minimal centered card
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.markdown(
            """
            <div style="margin-top:1.5rem; padding:1.5rem 1.5rem 1.25rem; background:var(--iai-surface,#111a2e);
                        border:1px solid var(--iai-border,#1e293b); border-radius:12px;">
              <div style="text-align:center; margin-bottom:1rem;">
                <div style="font-weight:800; font-size:1.1rem; color:var(--iai-primary,#06b6d4); letter-spacing:.02em;">InterviewAI</div>
                <h2 style="margin:.35rem 0 .2rem; color:var(--iai-text-heading,#fff); font-size:1.6rem;">{heading}</h2>
                <div style="color:var(--iai-muted,#94a3b8); font-size:.95rem;">{subtitle}</div>
              </div>
            """.format(
                heading="Welcome Back" if is_login else "Create Account",
                subtitle="Sign in to continue your interview practice." if is_login else "Register to start interview practice.",
            ),
            unsafe_allow_html=True,
        )

        if is_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="you@example.com")
                show_pw = st.checkbox("Show password", value=False, key="login_show_pw")
                password = st.text_input("Password", type="default" if show_pw else "password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if st.button("Create Account", key="to_register", use_container_width=True):
                st.session_state["auth_mode"] = "register"
                st.rerun()

            # Forgot Password shown only if actually supported (not implemented as self-service reset) -> omit broken action
            # st.caption("Forgot password? Contact your administrator.")

            if submitted:
                email_clean = (email or "").strip().lower()
                if not email_clean or not password:
                    st.error("Please enter your email and password.")
                elif "@" not in email_clean or "." not in email_clean.split("@")[-1]:
                    st.error("Please enter a valid email address.")
                else:
                    ok, res = client.login(email_clean, password)
                    if ok:
                        st.success("Signed in successfully.")
                        st.session_state["nav"] = "Dashboard"
                        st.rerun()
                    else:
                        st.error(f"Unable to sign in: {res}")
        else:
            with st.form("register_form", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="you@example.com")
                show_pw = st.checkbox("Show password", value=False, key="register_show_pw")
                password = st.text_input("Password", type="default" if show_pw else "password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="default" if show_pw else "password", placeholder="Confirm your password")
                submitted = st.form_submit_button("Register", type="primary", use_container_width=True)

            if st.button("Back to Login", key="to_login", use_container_width=True):
                st.session_state["auth_mode"] = "login"
                st.rerun()

            if submitted:
                email_clean = (email or "").strip().lower()
                full_name_clean = (full_name or "").strip()
                if not full_name_clean or not email_clean or not password or not confirm_password:
                    st.error("Full Name, Email, Password, and Confirm Password are required.")
                elif "@" not in email_clean or "." not in email_clean.split("@")[-1]:
                    st.error("Please enter a valid email address.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    payload = {
                        "full_name": full_name_clean,
                        "email": email_clean,
                        "password": password,
                        "roll_number": None,
                        "course": "",
                        "department": "",
                        "semester_year": "",
                        "skills": [],
                        "target_job_role": "",
                        "preferred_domain": "",
                    }
                    ok, res = client.register(payload)
                    if ok:
                        st.success("Account created. You are signed in.")
                        st.session_state["nav"] = "Dashboard"
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {res}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Use your existing InterviewAI account credentials to sign in.")


def render_logout_button():
    """Render logout action in sidebar."""
    user = st.session_state.get("user", {}) or {}
    student = st.session_state.get("student", {}) or {}
    name = student.get("full_name") or user.get("email") or "User"
    st.caption(f"Signed in as {name}")
    if st.button("Logout", key="btn_logout", type="secondary", use_container_width=True):
        client.logout()
        st.session_state["auth_mode"] = "login"
        st.session_state["nav"] = "Dashboard"
        st.rerun()
