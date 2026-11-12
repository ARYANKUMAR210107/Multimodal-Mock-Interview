"""
Assessment History & Final Reports Page
=======================================
Inspect historical mock interview sessions, review question-by-question
assessments, and open saved final comprehensive reports.
"""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from frontend.components.ui import page_header, section_header, empty_state, tier_badge
from frontend.services.api_client import client


def _fmt_pct(value):
    if isinstance(value, bool) or value is None:
        return "—"
    try:
        v = float(value)
        if v != v:
            return "—"
        # accept 0-1 or 0-100
        if v <= 1.0:
            return f"{v:.1%}"
        return f"{v:.1f}%"
    except Exception:
        return "—"


def _fmt_num(value, suffix=""):
    if isinstance(value, bool) or value is None:
        return "—"
    try:
        v = float(value)
        if v != v:
            return "—"
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}{suffix}"
        return f"{v:.2f}{suffix}".rstrip("0").rstrip(".") + suffix
    except Exception:
        return "—"


def _render_voice_measurements(meas: dict):
    if not isinstance(meas, dict) or not meas:
        st.caption("No acoustic measurements recorded.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Speaking Rate", f"{_fmt_num(meas.get('speaking_rate_wpm'), ' WPM')}" if meas.get('speaking_rate_wpm') is not None else "—")
    c2.metric("Pause Ratio", _fmt_pct(meas.get('pause_ratio')))
    mean_f0 = meas.get("mean_pitch_hz")
    c3.metric("Pitch (F₀)", f"{float(mean_f0):.1f} Hz" if isinstance(mean_f0, (int, float)) and mean_f0 == mean_f0 else "Unvoiced")
    pitch_std = meas.get("pitch_std_hz")
    c4.metric("Pitch Variation", f"{float(pitch_std):.1f} Hz" if isinstance(pitch_std, (int, float)) and pitch_std == pitch_std else "—")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Duration", f"{_fmt_num(meas.get('duration_sec'), ' s')}")
    d2.metric("Speech Time", f"{_fmt_num(meas.get('speaking_duration_sec'), ' s')}")
    d3.metric("Pause Count", _fmt_num(meas.get('pause_count')))
    rms = meas.get('mean_energy_rms')
    d4.metric("RMS Energy", f"{float(rms):.4f}" if isinstance(rms, (int, float)) and rms == rms else "—")
    fillers = meas.get("filler_word_count")
    if fillers is not None:
        st.caption(f"Detected filler words (transcript): **{fillers}** · Transcript words: **{meas.get('transcript_word_count', '—')}**")


def _extract_voice_metrics(qr: dict):
    voice = qr.get("voice") or {}
    meas = voice.get("measurements") or {}
    # compatibility if measurements wrapped
    if isinstance(meas, dict) and "voice_assessment" in meas and isinstance(meas.get("voice_assessment"), dict):
        meas = meas.get("voice_assessment") or {}
    mm = qr.get("multimodal") or {}
    return voice, meas, mm


def render_assessment_history(nav_to_fn):
    page_header(
        kicker="Audit & Records",
        title="Assessment History & Reports",
        description="Review saved session assessments and open final comprehensive reports. Data shown is loaded from actual stored session/report records.",
    )

    sessions = client.get_sessions(limit=50)
    if not sessions:
        empty_state("📂", "No Assessment History", "No mock interview sessions have been recorded yet.", "Start New Assessment", on_click=lambda: nav_to_fn("New Assessment"))
        return

    f_col1, f_col2 = st.columns([1.5, 1])
    with f_col1:
        status_filter = st.selectbox("Filter Sessions by Status:", ["All Sessions", "Completed", "Ready for Report", "In Progress"], index=0)
    with f_col2:
        student_filter = st.text_input("Filter by Candidate Name:", placeholder="Type to filter…")

    filtered_sessions = sessions
    if status_filter == "Completed":
        filtered_sessions = [s for s in filtered_sessions if s.get("status") == "completed"]
    elif status_filter == "Ready for Report":
        filtered_sessions = [s for s in filtered_sessions if s.get("status") == "ready_for_report"]
    elif status_filter == "In Progress":
        filtered_sessions = [s for s in filtered_sessions if s.get("status") == "in_progress"]

    if student_filter:
        filtered_sessions = [
            s for s in filtered_sessions
            if student_filter.lower() in (s.get("student_name") or s.get("student", {}).get("full_name", "")).lower()
        ]

    table_data = []
    for s in filtered_sessions:
        st_info = s.get("student", {})
        cand_name = s.get("student_name") or st_info.get("full_name") or s.get("student_id", "Unknown")
        table_data.append({
            "Session ID": s.get("id"),
            "Candidate": cand_name,
            "Roll Number": st_info.get("roll_number") or "—",
            "Domain": s.get("domain"),
            "Difficulty": s.get("difficulty"),
            "Questions": s.get("question_count"),
            "Status": (s.get("status", "") or "").replace("_", " ").title(),
            "Text": s.get("text_status", "—"),
            "Voice": s.get("voice_status", "—"),
            "Date": (s.get("created_at") or "")[:10] or "—",
        })

    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.info("No sessions match the selected filter criteria.")
        return

    st.write("")
    section_header("Detailed Session Report Inspection", "Select a session to open its saved final report or inspect current assessment records")

    preselected_session = st.session_state.get("view_report_session_id")
    sess_map = {
        f"{(s.get('student_name') or s.get('student', {}).get('full_name') or 'Candidate')} · {s.get('domain')} ({(s.get('status') or '').replace('_',' ')}) [ID: {s.get('id')}]": s.get("id")
        for s in filtered_sessions
    }

    default_idx = 0
    if preselected_session:
        for idx, sid in enumerate(sess_map.values()):
            if sid == preselected_session:
                default_idx = idx
                break

    chosen_sess_label = st.selectbox("Select Session to Inspect:", list(sess_map.keys()), index=default_idx)
    target_session_id = sess_map[chosen_sess_label]

    final_report_obj = client.get_final_report(target_session_id)
    ok, session_data = client.get_session(target_session_id)
    if not ok:
        err_msg = str(session_data)
        st.error(f"Could not load session `{target_session_id}`: {err_msg}")
        return

    report = None
    if isinstance(final_report_obj, dict):
        report = final_report_obj.get("report") or final_report_obj

    stud = session_data.get("student", {})
    with st.container(border=True):
        rh1, rh2 = st.columns([2.5, 1])
        with rh1:
            st.markdown(f'<div class="page-eyebrow">SESSION REPORT VIEW · {target_session_id}</div>', unsafe_allow_html=True)
            st.markdown(f"## {stud.get('full_name', 'Candidate')} — {session_data.get('domain', '')} Assessment")
            st.markdown(
                f"**Roll Number:** `{stud.get('roll_number') or 'N/A'}` · **Target Role:** `{stud.get('target_job_role') or 'Software Engineer'}`\n\n"
                f"**Course & Dept:** {stud.get('course') or '—'} ({stud.get('department') or '—'}) · "
                f"**Difficulty:** `{session_data.get('difficulty', '—')}` · **Mode:** `{session_data.get('mode', '—')}`\n\n"
                f"**Session status:** `{session_data.get('status')}` · **Text:** `{session_data.get('text_status')}` · **Voice:** `{session_data.get('voice_status')}`"
            )
            if report and report.get("report_version"):
                st.caption(f"Saved final report version: v{report.get('report_version')} · Report ID: {final_report_obj.get('id') if isinstance(final_report_obj, dict) else '—'}")
            elif session_data.get("status") == "ready_for_report":
                st.info("Required assessment modes appear complete for this session. Click 'Generate / Open Final Report' to create or reopen the saved final report.")
        with rh2:
            st.write("")
            export_payload = report if report else session_data
            st.download_button(
                "📥 Export JSON",
                data=json.dumps(export_payload, indent=2, default=str),
                file_name=f"interviewai_report_{target_session_id.lower()}.json",
                mime="application/json",
                use_container_width=True,
            )
            if session_data.get("status") == "in_progress":
                if st.button("Resume In-Progress Session ➔", type="primary", use_container_width=True):
                    st.session_state["active_interview_session_id"] = target_session_id
                    nav_to_fn("Active Workspace")
            if session_data.get("status") in ("ready_for_report", "completed", "in_progress"):
                if st.button("Generate / Open Final Report", type="secondary", use_container_width=True):
                    ok_r, rep_r = client.create_final_report(target_session_id)
                    if ok_r:
                        st.session_state["view_report_session_id"] = target_session_id
                        st.success("Final report loaded/saved successfully.")
                        st.rerun()
                    else:
                        st.error(f"Could not generate/open final report: {rep_r}")

    # Prefer final report content when available
    if report:
        section_header("Final Comprehensive Report", "Saved report generated from actual stored assessment records")
        assess = report.get("assessment", {}) or {}
        ta = report.get("text_assessment", {}) or {}
        va = report.get("voice_assessment", {}) or {}
        mm = report.get("multimodal_summary", {}) or {}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Questions", f"{assess.get('answered_questions', 0)}/{assess.get('total_questions', 0)} answered")
        m2.metric("Avg Text Score", _fmt_pct(ta.get('average_text_assessment_score')))
        m3.metric("Avg Voice/Multimodal", _fmt_pct(va.get('average_multimodal_voice_score') or mm.get('voice_score')))
        m4.metric("Combined Score", _fmt_pct(mm.get('combined_numeric_score')))

        with st.container(border=True):
            st.markdown("**Assessment summary**")
            st.caption(
                f"Completion: {assess.get('completion_percentage', 0)}% · "
                f"Text: {assess.get('text_status', ta.get('status', '—'))} · "
                f"Voice: {assess.get('voice_status', va.get('status', '—'))} · "
                f"Interview status: {assess.get('interview_status', '—')}"
            )
            if mm.get("explanation"):
                st.info(mm.get("explanation"))
            if va.get("score_note"):
                st.caption(va.get("score_note"))

        s1, s2, s3 = st.columns(3)
        with s1, st.container(border=True):
            st.markdown("**Strengths**")
            for x in (mm.get("strengths") or [])[:10]:
                st.success(f"✓ {x}")
            if not mm.get("strengths"):
                st.caption("No strengths summarized.")
        with s2, st.container(border=True):
            st.markdown("**Weaknesses / Areas for Improvement**")
            for x in (mm.get("weaknesses") or [])[:10]:
                st.info(f"→ {x}")
            if not mm.get("weaknesses"):
                st.caption("No weaknesses summarized.")
        with s3, st.container(border=True):
            st.markdown("**Recommendations**")
            for x in (mm.get("recommendations") or [])[:10]:
                st.warning(f"• {x}")
            if not mm.get("recommendations"):
                st.caption("No recommendations summarized.")

        section_header("Question-by-Question Results", "Text, voice/audio, and multimodal details from the saved final report")
        for qr in (report.get("question_results") or []):
            pos = qr.get("position")
            q_text = qr.get("question", "Question")
            status = (qr.get("status") or "").upper()
            text_q = qr.get("text") or {}
            voice_q, meas_q, mm_q = _extract_voice_metrics(qr)
            with st.expander(f"Q{pos}: {str(q_text)[:80]}… [{status}]"):
                st.markdown(f"**Question:** {q_text}")
                c1, c2 = st.columns(2, gap="large")
                with c1, st.container(border=True):
                    st.markdown("**Text Assessment**")
                    ans = text_q.get("answer") or "No written answer recorded."
                    st.markdown(f"> {ans}")
                    t_eval = text_q.get("assessment")
                    if t_eval:
                        st.metric("Text Quality Score", _fmt_pct(t_eval.get("score") or t_eval.get("quality_score")))
                        a,b,c,d = st.columns(4)
                        a.caption(f"Reference: {_fmt_pct(t_eval.get('cosine_similarity') or t_eval.get('relevance_score'))}")
                        b.caption(f"Question: {_fmt_pct(t_eval.get('question_alignment'))}")
                        c.caption(f"Keyword: {_fmt_pct(t_eval.get('jaccard_similarity') or t_eval.get('keyword_overlap'))}")
                        d.caption(f"Structure: {_fmt_pct(t_eval.get('structure_score'))}")
                    else:
                        st.caption("No saved text assessment for this question.")
                with c2, st.container(border=True):
                    st.markdown("**Voice / Audio Assessment**")
                    transcript = voice_q.get("corrected_transcript") or voice_q.get("original_transcript") or "No audio transcript recorded."
                    st.markdown(f"> {transcript}")
                    _render_voice_measurements(meas_q)
                    vta = voice_q.get("voice_text_assessment")
                    if vta and isinstance(vta, dict):
                        st.caption(f"Transcript text assessment score: {_fmt_pct(vta.get('quality_score'))}")
                    for s in (voice_q.get("strengths") or [])[:5]:
                        st.success(f"✓ {s}")
                    for s in (voice_q.get("improvement_areas") or [])[:5]:
                        st.info(f"→ {s}")
                    for s in (voice_q.get("observations") or [])[:5]:
                        st.caption(f"• {s}")
                if mm_q:
                    st.markdown("**Multimodal (question-level, saved/practice formula)**")
                    m1c,m2c,m3c = st.columns(3)
                    m1c.metric("Text Score", _fmt_pct(mm_q.get("text_score")))
                    m2c.metric("Voice Composite", _fmt_pct(mm_q.get("voice_composite_score")))
                    m3c.metric("Status", str(mm_q.get("status", "—")))
                    if mm_q.get("formula"):
                        st.caption(f"Formula: {mm_q.get('formula')}")
        return

    # Fallback: show live session question records if no final report saved yet
    section_header("Current Session Assessment Records", "No saved final comprehensive report found yet; showing available per-question records")
    q_list = session_data.get("questions", [])
    for q_item in q_list:
        pos = q_item.get("position")
        q_obj = q_item.get("question", {})
        q_text = q_obj.get("question_text", "Question")
        status = (q_item.get("status", "pending") or "pending").upper()
        text_resp = q_item.get("text_response") or {}
        voice_resp = q_item.get("voice_response") or {}
        mm_resp = voice_resp.get("multimodal_report") or {}
        with st.expander(f"Question {pos}: {str(q_text)[:75]}… [{status}]"):
            st.markdown(f"**Full Question:** {q_text}")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Written/Text Response**")
                ans_text = text_resp.get("answer_text") or "No written answer recorded."
                st.markdown(f"> {ans_text}")
                t_eval = text_resp.get("text_assessment")
                if t_eval:
                    st.metric("Text Quality Score", _fmt_pct(t_eval.get('quality_score')))
                    st.caption(f"Reference: {_fmt_pct(t_eval.get('relevance_score'))} · Question: {_fmt_pct(t_eval.get('question_alignment_score'))} · Keyword: {_fmt_pct(t_eval.get('keyword_overlap'))} · Structure: {_fmt_pct(t_eval.get('structure_score'))}")
                else:
                    st.caption("No semantic assessment recorded.")
            with c2:
                st.markdown("**Voice / Spoken Response**")
                transcript = voice_resp.get("corrected_transcript") or voice_resp.get("original_transcript") or "No audio transcript recorded."
                st.markdown(f"> {transcript}")
                v_meas = voice_resp.get("audio_measurements") or (mm_resp.get("voice_assessment") if isinstance(mm_resp, dict) else None)
                _render_voice_measurements(v_meas or {})
                if isinstance(mm_resp, dict) and mm_resp.get("text_assessment"):
                    ta2 = mm_resp.get("text_assessment") or {}
                    st.caption(f"Multimodal text score: {_fmt_pct(ta2.get('quality_score'))}")
                fb = (mm_resp.get("feedback") if isinstance(mm_resp, dict) else None) or {}
                for s in (fb.get("strengths") or [])[:4]:
                    st.success(f"✓ {s}")
                for s in (fb.get("areas_for_improvement") or [])[:4]:
                    st.info(f"→ {s}")
