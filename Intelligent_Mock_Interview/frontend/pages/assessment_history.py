"""
Assessment History & Final Reports Page
=======================================
Inspect historical mock interview sessions, review question-by-question
multimodal reports, and export verified JSON assessment records.
"""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from frontend.components.ui import page_header, section_header, empty_state, tier_badge
from frontend.services.api_client import client


def render_assessment_history(nav_to_fn):
    page_header(
        kicker="Audit & Records",
        title="Assessment History & Reports",
        description="Review comprehensive candidate evaluation reports, question breakdown scores, acoustic measurements, and export assessment records.",
    )

    # 1. Fetch sessions
    sessions = client.get_sessions(limit=50)
    if not sessions:
        empty_state("📂", "No Assessment History", "No mock interview sessions have been recorded yet.", "Start New Assessment", on_click=lambda: nav_to_fn("New Assessment"))
        return

    # Filter controls
    f_col1, f_col2 = st.columns([1.5, 1])
    with f_col1:
        status_filter = st.selectbox("Filter Sessions by Status:", ["All Sessions", "Completed", "In Progress"], index=0)
    with f_col2:
        student_filter = st.text_input("Filter by Candidate Name:", placeholder="Type to filter…")

    filtered_sessions = sessions
    if status_filter == "Completed":
        filtered_sessions = [s for s in filtered_sessions if s.get("status") == "completed"]
    elif status_filter == "In Progress":
        filtered_sessions = [s for s in filtered_sessions if s.get("status") == "in_progress"]

    if student_filter:
        filtered_sessions = [s for s in filtered_sessions if student_filter.lower() in s.get("student", {}).get("full_name", "").lower()]

    # Table of Sessions
    table_data = []
    for s in filtered_sessions:
        st_info = s.get("student", {})
        table_data.append({
            "Session ID": s.get("id"),
            "Candidate": st_info.get("full_name", "Unknown"),
            "Roll Number": st_info.get("roll_number") or "—",
            "Domain": s.get("domain"),
            "Difficulty": s.get("difficulty"),
            "Questions": s.get("question_count"),
            "Status": s.get("status", "").replace("_", " ").title(),
            "Date": s.get("created_at")[:10] if s.get("created_at") else "—",
        })

    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.info("No sessions match the selected filter criteria.")
        return

    st.write("")
    st.write("")

    # 2. Select Session to Inspect
    section_header("Detailed Session Report Inspection", "Select an interview session to view the final evaluation report")
    
    preselected_session = st.session_state.get("view_report_session_id")
    sess_map = {f"{s.get('student', {}).get('full_name')} · {s.get('domain')} ({s.get('status')}) [ID: {s.get('id')}]": s.get("id") for s in filtered_sessions}
    
    default_idx = 0
    if preselected_session:
        for idx, sid in enumerate(sess_map.values()):
            if sid == preselected_session:
                default_idx = idx
                break

    chosen_sess_label = st.selectbox("Select Session to Inspect:", list(sess_map.keys()), index=default_idx)
    target_session_id = sess_map[chosen_sess_label]

    # Fetch final report or session details
    final_report = client.get_final_report(target_session_id)
    ok, session_data = client.get_session(target_session_id)

    if not ok:
        err_msg = str(session_data)
        if "own sessions" in err_msg:
            st.error(
                f"⚠️ **Access denied.** You do not own session `{target_session_id}`."
            )
        elif "not found" in err_msg.lower():
            st.error(f"⚠️ **Session not found.** `{target_session_id}` does not exist.")
        else:
            st.error(f"⚠️ **Could not load session** `{target_session_id}`: {err_msg}")
        return

    st.write("")

    # 3. Report Header Card
    stud = session_data.get("student", {})
    with st.container(border=True):
        rh1, rh2 = st.columns([2.5, 1])
        with rh1:
            st.markdown(f'<div class="page-eyebrow">FINAL ASSESSMENT REPORT · SESSION {target_session_id}</div>', unsafe_allow_html=True)
            st.markdown(f"## {stud.get('full_name')} — {session_data.get('domain')} Assessment")
            st.markdown(f"**Roll Number:** `{stud.get('roll_number') or 'N/A'}` · **Target Role:** `{stud.get('target_job_role') or 'Software Engineer'}`")
            st.markdown(f"**Course & Dept:** {stud.get('course')} ({stud.get('department')}) · **Difficulty:** `{session_data.get('difficulty')}`")
        with rh2:
            st.write("")
            st.download_button(
                "📥 Export Report JSON",
                data=json.dumps(final_report or session_data, indent=2),
                file_name=f"interviewai_report_{target_session_id.lower()}.json",
                mime="application/json",
                use_container_width=True,
            )
            if session_data.get("status") == "in_progress":
                if st.button("Resume In-Progress Session ➔", type="primary", use_container_width=True):
                    st.session_state["active_interview_session_id"] = target_session_id
                    nav_to_fn("Active Workspace")

    # 4. Question-by-Question Evaluation Accordion
    st.write("")
    section_header("Question-by-Question Breakdown", "Review candidate answers, NLP relevance scores, and acoustic metrics")

    q_list = session_data.get("questions", [])
    for q_item in q_list:
        pos = q_item.get("position")
        q_obj = q_item.get("question", {})
        q_text = q_obj.get("question_text", "Question")
        status = q_item.get("status", "pending")
        
        # Responses
        text_resp = q_item.get("text_response") or {}
        voice_resp = q_item.get("voice_response") or {}

        with st.expander(f"Question {pos}: {q_text[:75]}… [{status.upper()}]"):
            st.markdown(f"**Full Question:** {q_text}")
            if q_obj.get("reference_answer"):
                st.caption(f"**Benchmark Reference:** {q_obj.get('reference_answer')}")

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("##### ✍️ Written Response Assessment")
                ans_text = text_resp.get("answer_text") or "No written answer recorded."
                st.markdown(f"> *\"{ans_text}\"*")
                t_eval = text_resp.get("text_assessment")
                if t_eval:
                    st.metric("Text Quality Score", f"{t_eval.get('quality_score', 0):.1%}")
                    st.markdown(tier_badge(t_eval.get('quality_score', 0)), unsafe_allow_html=True)
                    sub1, sub2 = st.columns(2)
                    sub1.caption(f"Reference Cosine: {t_eval.get('relevance_score', 0):.1%}")
                    sub2.caption(f"Keyword Overlap: {t_eval.get('keyword_overlap', 0):.1%}")
                else:
                    st.caption("No semantic assessment recorded.")

            with c2:
                st.markdown("##### 🎙️ Voice & Spoken Assessment")
                transcript = voice_resp.get("corrected_transcript") or voice_resp.get("original_transcript") or "No audio transcript recorded."
                st.markdown(f"> *\"{transcript}\"*")
                v_meas = voice_resp.get("audio_measurements")
                if v_meas:
                    a1, a2 = st.columns(2)
                    a1.caption(f"Speaking Rate: {v_meas.get('speaking_rate_wpm', '—')} WPM")
                    a2.caption(f"Pause Ratio: {v_meas.get('pause_ratio', 0):.1%}")
                    a3, a4 = st.columns(2)
                    mean_f0 = v_meas.get("mean_pitch_hz")
                    a3.caption(f"Pitch (F₀): {mean_f0:.1f} Hz" if mean_f0 else "Pitch: Unvoiced")
                    a4.caption(f"Fillers: {v_meas.get('filler_word_count', 0)}")
                else:
                    st.caption("No acoustic measurements recorded.")

