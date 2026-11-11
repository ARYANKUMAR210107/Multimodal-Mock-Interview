"""
Student Management Page Renderer
================================
Directory search, student registration, detailed candidate profiles,
and historical assessment performance tracking.
"""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from frontend.components.ui import page_header, section_header, empty_state, status_pill
from frontend.services.api_client import client


def render_students(nav_to_fn, initial_tab: str = "Directory"):
    page_header(
        kicker="Candidate Records",
        title="Student Management",
        description="Register candidate profiles, view technical interview performance history, and manage academic metadata.",
    )

    tab_names = ["Student Directory", "Register Candidate", "Student Profile"]
    default_idx = 0
    if initial_tab == "Register":
        default_idx = 1
    elif initial_tab == "Profile":
        default_idx = 2

    # If AppTest or direct navigation routed to "Student Directory" or "Student Profile", handle gracefully
    current_nav = st.session_state.get("nav")
    if current_nav == "Student Profile":
        default_idx = 2
    elif current_nav in ("Student Directory", "Student Registration"):
        default_idx = 0 if current_nav == "Student Directory" else 1

    t_dir, t_reg, t_prof = st.tabs(tab_names)

    # ── TAB 1: STUDENT DIRECTORY ─────────────────────────────────────────────
    with t_dir:
        section_header("Candidate Directory", "Search and manage enrolled students")
        
        search_query = st.text_input("🔍 Search by name, roll number, or department", placeholder="Type to filter…", key="search_students_input")
        students = client.get_students(search=search_query, limit=100)

        if students:
            # Table View
            table_rows = []
            for s in students:
                skills_val = s.get("skills", [])
                skills_str = ", ".join(skills_val) if isinstance(skills_val, list) else str(skills_val)
                table_rows.append({
                    "ID": s.get("id"),
                    "Full Name": s.get("full_name"),
                    "Roll Number": s.get("roll_number") or "—",
                    "Department": s.get("department") or "—",
                    "Target Role": s.get("target_job_role") or "—",
                    "Preferred Domain": s.get("preferred_domain") or "—",
                    "Skills": skills_str or "—",
                })
            df_display = pd.DataFrame(table_rows)
            st.dataframe(df_display.drop(columns=["ID"]), use_container_width=True, hide_index=True)

            st.write("")
            st.markdown("#### Quick Select Candidate")
            s_col1, s_col2, s_col3 = st.columns([2, 1, 1], gap="medium")
            with s_col1:
                student_options = {f"{s['full_name']} ({s.get('roll_number') or s['id']})": s["id"] for s in students}
                selected_label = st.selectbox("Choose student for profile or assessment:", list(student_options.keys()), key="select_student_from_dir")
                sel_id = student_options[selected_label]
            with s_col2:
                st.write("")
                st.write("")
                if st.button("View profile", key="btn_view_profile_action", use_container_width=True, type="secondary"):
                    st.session_state["profile_student_id"] = sel_id
                    st.session_state["nav"] = "Student Profile"
                    st.rerun()
            with s_col3:
                st.write("")
                st.write("")
                if st.button("Start Assessment ➔", use_container_width=True, type="primary"):
                    st.session_state["assessment_student_id"] = sel_id
                    nav_to_fn("New Assessment")
        else:
            empty_state("👥", "No Students Found", "No registered candidates match your search query.", "Register First Student", on_click=lambda: None)

    # ── TAB 2: REGISTER CANDIDATE ────────────────────────────────────────────
    with t_reg:
        section_header("Candidate Registration", "Enroll a new candidate into InterviewAI")
        with st.form("form_register_student", clear_on_submit=True):
            r1, r2 = st.columns(2)
            with r1:
                full_name = st.text_input("Full Name *", placeholder="e.g. John Doe", help="Candidate's official name")
                roll_number = st.text_input("Roll Number", placeholder="e.g. CS2024-042", help="Unique student roll identifier")
                course = st.selectbox("Degree / Course", ["B.Tech", "B.E.", "M.Tech", "MCA", "B.Sc", "Other"])
                semester_year = st.selectbox("Academic Year", ["1st Year", "2nd Year", "3rd Year", "Final Year", "Graduate"])
                email = st.text_input("Email Address", placeholder="candidate@university.edu")
            with r2:
                registration_number = st.text_input("Registration / University ID", placeholder="e.g. REG-2024-9871")
                department = st.selectbox("Department", [
                    "Computer Science & Engineering",
                    "Information Technology",
                    "Artificial Intelligence & Data Science",
                    "Electronics & Communication",
                    "Electrical Engineering",
                    "Mechanical Engineering",
                    "Other"
                ])
                target_job_role = st.text_input("Target Job Role", placeholder="e.g. Machine Learning Engineer")
                preferred_domain = st.selectbox("Preferred Technical Domain", [
                    "Python", "C++", "Java", "Data Structures & Algorithms",
                    "DBMS", "Operating Systems", "Computer Networks", "Machine Learning",
                    "Artificial Intelligence", "Web Development", "Software Engineering", "HR & Behavioral"
                ])
                skills_raw = st.text_input("Technical Skills (comma-separated)", placeholder="e.g. Python, SQL, Docker, Scikit-Learn")

            submitted = st.form_submit_button("Register Candidate", type="primary", use_container_width=True)

            if submitted:
                if not full_name.strip():
                    st.error("Full Name is required.")
                else:
                    skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
                    payload = {
                        "full_name": full_name.strip(),
                        "roll_number": roll_number.strip() or None,
                        "registration_number": registration_number.strip() or None,
                        "course": course,
                        "department": department,
                        "semester_year": semester_year,
                        "email": email.strip() or None,
                        "target_job_role": target_job_role.strip(),
                        "preferred_domain": preferred_domain,
                        "skills": skills_list,
                    }
                    success, res = client.create_student(payload)
                    if success:
                        new_id = res.get("id")
                        st.success(f"Candidate **{full_name}** successfully enrolled! Unique ID: `{new_id}`")
                        st.session_state["assessment_student_id"] = new_id
                        st.session_state["profile_student_id"] = new_id
                        st.session_state["nav"] = "Student Profile"
                        st.rerun()
                    else:
                        err_str = str(res)
                        if "409" in err_str or "already exists" in err_str.lower() or "unique" in err_str.lower():
                            st.warning(f"⚠️ Candidate registration conflict: A student with this Roll Number or Registration ID already exists in the database.")
                        else:
                            st.error(f"Registration failed: {res}")

    # ── TAB 3: STUDENT PROFILE ───────────────────────────────────────────────
    with t_prof:
        profile_id = st.session_state.get("profile_student_id")
        if not profile_id:
            st.info("Select a student from the Directory tab to view their profile.")
            return

        if profile_id:
            student = client.get_student(profile_id)
            if student:
                # Render st.subheader for AppTest assertion
                st.subheader(student.get("full_name"))
                
                p_col1, p_col2 = st.columns([1.5, 1], gap="large")
                with p_col1, st.container(border=True):
                    st.markdown(f'<div class="page-eyebrow">STUDENT PROFILE · {student.get("id")}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Roll Number:** `{student.get('roll_number') or 'N/A'}` · **Reg No:** `{student.get('registration_number') or 'N/A'}`")
                    st.markdown(f"**Course & Dept:** {student.get('course')} — {student.get('department')} ({student.get('semester_year')})")
                    st.markdown(f"**Target Role:** `{student.get('target_job_role') or 'Software Engineer'}` · **Domain:** `{student.get('preferred_domain')}`")
                    st.markdown(f"**Email:** {student.get('email') or 'Not provided'}")

                    # Skills
                    skills = student.get("skills", [])
                    if skills:
                        st.markdown("**Skills:** " + " ".join([f"`{sk}`" for sk in skills]))

                with p_col2, st.container(border=True):
                    section_header("Quick Actions")
                    st.write("")
                    if st.button("Create New Assessment", key="btn_create_new_assessment", type="primary", use_container_width=True):
                        st.session_state["assessment_student_id"] = profile_id
                        st.session_state["nav"] = "Create Interview"
                        st.rerun()

                # Past Sessions for Student
                st.write("")
                section_header("Historical Interview Assessments", f"Sessions recorded for {student.get('full_name')}")
                past_sessions = client.get_sessions(student_id=profile_id, limit=20)
                if past_sessions:
                    sess_rows = []
                    for ps in past_sessions:
                        sess_rows.append({
                            "Session ID": ps.get("id"),
                            "Title": ps.get("title"),
                            "Domain": ps.get("domain"),
                            "Difficulty": ps.get("difficulty"),
                            "Status": ps.get("status"),
                            "Questions": ps.get("question_count"),
                            "Date": ps.get("created_at")[:10] if ps.get("created_at") else "—",
                        })
                    st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True)

                    in_prog = [ps for ps in past_sessions if ps.get("status") == "in_progress"]
                    if in_prog:
                        st.write("")
                        st.markdown("##### Resume In-Progress Session")
                        res_col1, res_col2 = st.columns([2, 1])
                        with res_col1:
                            res_map = {f"{ps.get('title')} ({ps.get('domain')}) [ID: {ps.get('id')}]": ps.get("id") for ps in in_prog}
                            res_chosen = st.selectbox("Select session to resume:", list(res_map.keys()), key="sel_resume_from_profile")
                        with res_col2:
                            st.write("")
                            st.write("")
                            if st.button("Enter Active Workspace ➔", key="btn_resume_from_profile", type="primary", use_container_width=True):
                                st.session_state["active_interview_session_id"] = res_map[res_chosen]
                                nav_to_fn("Active Workspace")
                else:
                    st.info("No past interview sessions on record for this student.")
            else:
                st.error("Selected student record could not be loaded.")
        else:
            empty_state("👤", "No Student Selected", "Select a student from the directory to inspect profile and historical performance.")
