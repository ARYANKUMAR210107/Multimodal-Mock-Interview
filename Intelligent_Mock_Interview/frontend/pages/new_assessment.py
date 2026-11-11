"""
New Assessment Configuration Page
=================================
Configure and instantiate a new persistent interview session.
Integrates student metadata, domain requirements, difficulty calibration,
and question count availability verification.
"""
from __future__ import annotations

import streamlit as st
from frontend.components.ui import page_header, section_header, empty_state
from frontend.services.api_client import client


def render_new_assessment(nav_to_fn):
    page_header(
        kicker="Session Setup",
        title="Create New Assessment",
        description="Configure an adaptive technical mock interview session tailored to the candidate's target role and domain.",
    )

    # 1. Fetch Students
    students = client.get_students(limit=100)
    if not students:
        empty_state(
            "👤",
            "No Registered Candidates Found",
            "An interview session requires an enrolled student record to persist answers, assessments, and reports.",
            "Register Candidate First",
            on_click=lambda: nav_to_fn("Students", tab="Register"),
        )
        return

    # Student selector mapping
    preselected_id = st.session_state.get("assessment_student_id")
    student_map = {f"{s['full_name']} (Roll: {s.get('roll_number') or s['id']})": s["id"] for s in students}
    options_list = list(student_map.keys())

    default_idx = 0
    if preselected_id:
        for idx, (lbl, sid) in enumerate(student_map.items()):
            if sid == preselected_id:
                default_idx = idx
                break

    # 2. Configuration Form
    with st.container(border=True):
        section_header("1. Select Candidate", "Choose the student undergoing assessment")
        chosen_label = st.selectbox("Registered Candidate *", options_list, index=default_idx)
        selected_student_id = student_map[chosen_label]

    st.write("")

    with st.container(border=True):
        section_header("2. Interview Domain & Track", "Select technical area for questions")
        c1, c2 = st.columns(2)
        with c1:
            domains = [
                "Machine Learning",
                "Python",
                "Data Structures & Algorithms",
                "DBMS",
                "Operating Systems",
                "Computer Networks",
                "Artificial Intelligence",
                "Java",
                "C++",
                "Web Development",
                "Software Engineering",
                "HR & Behavioral",
            ]
            domain = st.selectbox("Technical Track / Domain *", domains, key="flow_assessment_domain")
        with c2:
            difficulty = st.selectbox("Target Difficulty *", ["Any", "Beginner", "Intermediate", "Advanced"], index=0)

    st.write("")

    with st.container(border=True):
        section_header("3. Assessment Parameters", "Question count, modality, and time limits")
        p1, p2, p3 = st.columns(3)
        with p1:
            question_count = st.selectbox("Number of Questions *", [5, 10, 15, 20], index=0)
        with p2:
            mode = st.selectbox("Interview Modality *", [
                ("mixed", "Mixed (Text & Voice Assessment)"),
                ("text", "Text Only Assessment"),
                ("voice", "Voice Only Assessment")
            ], format_func=lambda x: x[1])
        with p3:
            time_limit = st.selectbox("Time Limit (Optional)", [None, 15, 30, 45, 60], format_func=lambda x: "No Limit" if x is None else f"{x} Minutes")

        # Question Availability Check
        existing_q = client.get_questions(domain=domain, difficulty=difficulty if difficulty != "Any" else "", limit=50)
        avail_cnt = len(existing_q)
        if avail_cnt < question_count:
            st.warning(f"⚠️ Note: Question bank currently contains {avail_cnt} question(s) matching '{domain}' ({difficulty}). The session will include up to {min(avail_cnt, question_count)} questions.")
        else:
            st.success(f"✓ Verified: {avail_cnt} question(s) available in '{domain}' ({difficulty}).")

    st.write("")

    # 4. Launch Action
    col_l, col_r = st.columns([1, 1], gap="medium")
    with col_r:
        if st.button("Create assessment and start", type="primary", use_container_width=True):
            payload = {
                "student_id": selected_student_id,
                "domain": domain,
                "difficulty": difficulty,
                "mode": mode[0],
                "question_count": question_count,
                "time_limit_minutes": time_limit,
                "title": f"{domain} Interview",
            }
            with st.spinner("Configuring session and assigning benchmark questions…"):
                success, res = client.create_session(payload)
                if success:
                    session_id = res.get("id")
                    st.session_state["active_interview_session_id"] = session_id
                    st.success(f"Session created successfully! Session ID: `{session_id}`")
                    nav_to_fn("Active Workspace")
                else:
                    st.error(f"Failed to create session: {res}")
    with col_l:
        if st.button("Cancel", use_container_width=True):
            nav_to_fn("Dashboard")
