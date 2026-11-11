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
    subtitle = "Welcome back. Sign in to continue." if mode == "login" else "Create your account to start interview practice."

    st.markdown(
        f"""
        <div style="max-width:460px;margin:2.5rem auto 0;padding:0 1rem;">
          <div style="background:var(--iai-surface,#111a2e);border:1px solid var(--iai-border,#1e293b);border-radius:14px;padding:2rem 1.75rem;box-shadow:0 8px 24px rgba(0,0,0,.18);">
            <div style="text-align:center;margin-bottom:1.25rem;">
              <div style="display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,#06b6d4,#3b82f6);color:#020617;font-weight:800;font-size:1.2rem;margin-bottom:.75rem;">IA</div>
              <div style="color:var(--iai-text-heading,#fff);font-size:1.6rem;font-weight:800;letter-spacing:-.02em;">InterviewAI</div>
              <div style="color:var(--iai-muted,#94a3b8);font-size:.92rem;margin-top:.35rem;">{subtitle}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "login":
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@university.edu")
            show_pw = st.checkbox("Show password", value=False, key="login_show_pw")
            password = st.text_input("Password", type="default" if show_pw else "password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.caption("Forgot password: contact your administrator.")
        with c2:
            if st.button("Create account", key="to_register", use_container_width=True):
                st.session_state["auth_mode"] = "register"
                st.rerun()

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
            full_name = st.text_input("Full name *", placeholder="e.g. Priya Sharma")
            email = st.text_input("Email *", placeholder="you@university.edu")
            r1, r2 = st.columns(2)
            with r1:
                course = st.text_input("Course", placeholder="B.Tech")
                department = st.text_input("Department", placeholder="Computer Science")
                semester_year = st.text_input("Semester / Year", placeholder="Final Year")
            with r2:
                roll_number = st.text_input("Roll number", placeholder="Optional")
                target_job_role = st.text_input("Target job role", placeholder="e.g. Software Engineer")
                preferred_domain = st.text_input("Preferred domain", placeholder="e.g. Python")
            show_pw = st.checkbox("Show password", value=False, key="register_show_pw")
            p1, p2 = st.columns(2)
            with p1:
                password = st.text_input("Password *", type="default" if show_pw else "password", placeholder="At least 8 characters")
            with p2:
                confirm_password = st.text_input("Confirm password *", type="default" if show_pw else "password", placeholder="Re-enter password")
            submitted = st.form_submit_button("Register", type="primary", use_container_width=True)

        if st.button("Back to login", key="to_login", use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.rerun()

        if submitted:
            email_clean = (email or "").strip().lower()
            if not full_name.strip() or not email_clean or not password or not confirm_password:
                st.error("Full name, email, password, and confirm password are required.")
            elif "@" not in email_clean or "." not in email_clean.split("@")[-1]:
                st.error("Please enter a valid email address.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                payload = {
                    "full_name": full_name.strip(),
                    "email": email_clean,
                    "password": password,
                    "roll_number": roll_number.strip() or None,
                    "course": course.strip(),
                    "department": department.strip(),
                    "semester_year": semester_year.strip(),
                    "skills": [],
                    "target_job_role": target_job_role.strip(),
                    "preferred_domain": preferred_domain.strip(),
                }
                ok, res = client.register(payload)
                if ok:
                    st.success("Account created. You are signed in.")
                    st.session_state["nav"] = "Dashboard"
                    st.rerun()
                else:
                    st.error(f"Registration failed: {res}")

    st.markdown("</div></div>", unsafe_allow_html=True)


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
