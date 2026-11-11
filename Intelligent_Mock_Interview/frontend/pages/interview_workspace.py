"""
Unified Connected Interview Workspace Page
==========================================
The core interactive assessment environment connecting Text Interview,
Voice Recording, Whisper Transcription, Librosa Audio Extraction,
and Multimodal Evaluation for one persistent interview session.
"""
from __future__ import annotations

import hashlib
import json
import pandas as pd
import streamlit as st

from frontend.components.ui import page_header, section_header, empty_state, tier_badge, status_pill
from frontend.services.api_client import client


def render_interview_workspace(nav_to_fn):
    session_id = st.session_state.get("active_interview_session_id")

    # If no active session, prompt user to select or create one
    if not session_id:
        page_header(
            kicker="Active Assessment",
            title="Active Workspace",
            description="Execute interactive mock technical interviews with real-time text scoring and acoustic vocal analysis.",
        )
        sessions = client.get_sessions(status="in_progress", limit=20)
        if sessions:
            st.info("Select an active session to resume or launch a new assessment:")
            s_map = {f"{s.get('student', {}).get('full_name', s.get('student_id'))} — {s.get('domain')} ({s.get('difficulty')}) [ID: {s.get('id')}]": s.get("id") for s in sessions}
            chosen = st.selectbox("In-Progress Sessions:", list(s_map.keys()))
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Resume Selected Session ➔", type="primary", use_container_width=True):
                    st.session_state["active_interview_session_id"] = s_map[chosen]
                    st.rerun()
            with c2:
                if st.button("Configure New Assessment", use_container_width=True):
                    nav_to_fn("New Assessment")
        else:
            empty_state(
                "🎙️",
                "No Active Interview Session",
                "There is currently no interview session loaded. Start a new assessment to begin.",
                "Start New Assessment",
                on_click=lambda: nav_to_fn("New Assessment"),
            )
        return

    # Fetch active session details from SQLite
    ok, session = client.get_session(session_id)
    if not ok:
        err_msg = str(session)
        if "own sessions" in err_msg:
            st.error(
                f"⚠️ **Access denied.** You do not own interview session `{session_id}`. "
                f"Please sign in as the session owner or start a new assessment."
            )
        elif "not found" in err_msg.lower():
            st.error(f"⚠️ **Session not found.** Interview session `{session_id}` does not exist.")
        else:
            st.error(f"⚠️ **Could not load interview session** `{session_id}`: {err_msg}")
        if st.button("Return to Dashboard"):
            st.session_state.pop("active_interview_session_id", None)
            nav_to_fn("Dashboard")
        return

    student = session.get("student", {})
    questions = session.get("questions", [])
    total_q = len(questions)
    current_pos = st.session_state.get("workspace_current_pos", session.get("current_position", 1))

    # Keep position in bounds
    if current_pos < 1: current_pos = 1
    if current_pos > total_q: current_pos = total_q

    # 1. Header Banner with Candidate & Session Metadata
    st.markdown(
        f"""
        <div class="page-header-wrapper" style="margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.75rem;">
                <div>
                    <div class="page-eyebrow">ACTIVE INTERVIEW WORKSPACE · {session.get('domain').upper()}</div>
                    <h2 class="page-title" style="font-size: 1.85rem; margin-bottom: 0.25rem;">
                        {student.get('full_name')} · {session.get('title')}
                    </h2>
                    <div style="color: var(--iai-muted); font-size: 0.88rem;">
                        <strong>Candidate:</strong> {student.get('full_name')} ({student.get('roll_number') or 'ID: ' + student.get('id')}) · 
                        <strong>Domain:</strong> {session.get('domain')} · 
                        <strong>Difficulty:</strong> {session.get('difficulty')} · 
                        <strong>Session ID:</strong> <code>{session_id}</code>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: var(--iai-muted); margin-bottom: 0.25rem;">SESSION PROGRESS</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: var(--iai-secondary);">
                        Question {current_pos} <span style="font-size: 0.95rem; color: var(--iai-subtext); font-weight: 500;">/ {total_q}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar
    progress_val = float(current_pos) / float(total_q) if total_q > 0 else 0.0
    st.progress(progress_val)

    # 2. Question Navigator (Previous, Dropdown, Next)
    n_col1, n_col2, n_col3 = st.columns([1, 2.5, 1], gap="small")
    with n_col1:
        if st.button("⬅ Previous", disabled=(current_pos <= 1), use_container_width=True):
            st.session_state["workspace_current_pos"] = current_pos - 1
            st.rerun()
    with n_col2:
        q_options = {f"Q{q['position']}: {q['question']['question_text'][:55]}…": q['position'] for q in questions}
        sel_label = st.selectbox(
            "Jump to Question:",
            list(q_options.keys()),
            index=(current_pos - 1),
            label_visibility="collapsed",
            key=f"q_selector_{current_pos}",
        )
        if q_options[sel_label] != current_pos:
            st.session_state["workspace_current_pos"] = q_options[sel_label]
            st.rerun()
    with n_col3:
        if st.button("Next ➡", disabled=(current_pos >= total_q), use_container_width=True):
            st.session_state["workspace_current_pos"] = current_pos + 1
            st.rerun()

    # Active Question Data
    curr_q_item = questions[current_pos - 1] if questions else {}
    q_data = curr_q_item.get("question", {})
    q_text = q_data.get("question_text", "Question unavailable")
    q_ref = q_data.get("reference_answer", "")
    q_concepts = q_data.get("expected_concepts", [])

    # Existing responses for this question
    text_resp = curr_q_item.get("text_response") or {}
    voice_resp = curr_q_item.get("voice_response") or {}

    st.write("")

    # 3. Assessed Question Card
    with st.container(border=True):
        q_top_col1, q_top_col2 = st.columns([3, 1])
        with q_top_col1:
            st.markdown(f'<div class="page-eyebrow">QUESTION {current_pos} OF {total_q} · TOPIC: {q_data.get("topic", "General")}</div>', unsafe_allow_html=True)
            st.markdown(f"### {q_text}")
        with q_top_col2:
            st.markdown(f"**Difficulty:** `{q_data.get('difficulty', 'Intermediate')}`")
            status_lbl = curr_q_item.get("status", "pending")
            status_pill("online" if status_lbl == "assessed" else "warning", status_lbl.capitalize())

        if q_concepts:
            st.markdown(f"<div style='font-size: 0.82rem; color: var(--iai-muted); margin-top: 0.4rem;'><strong>Key Concepts:</strong> {' · '.join(f'<code>{c}</code>' for c in q_concepts)}</div>", unsafe_allow_html=True)

        # ── Text-to-Speech (TTS) Question Reader ─────────────────────────
        tts_q_key = f"tts_audio_q_{current_pos}"
        col_tts1, col_tts2 = st.columns([1, 2])
        with col_tts1:
            if st.button("🔊 Read Question Aloud", key=f"btn_read_q_{current_pos}", help="Listen to the question read aloud by the AI interviewer"):
                with st.spinner("🔊 Synthesizing question audio…"):
                    ok_tts, tts_data = client.get_tts_audio(q_text)
                    if ok_tts and isinstance(tts_data, bytes):
                        st.session_state[tts_q_key] = tts_data
                        st.session_state[f"play_tts_{current_pos}"] = True
                    else:
                        st.session_state[f"use_browser_tts_{current_pos}"] = True
                st.rerun()

        with col_tts2:
            if st.session_state.get(tts_q_key):
                st.audio(st.session_state[tts_q_key], format="audio/wav", autoplay=st.session_state.get(f"play_tts_{current_pos}", False))
                st.session_state[f"play_tts_{current_pos}"] = False
            elif st.session_state.get(f"use_browser_tts_{current_pos}"):
                safe_text = json.dumps(q_text)
                st.components.v1.html(
                    f"""
                    <script>
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            var msg = new SpeechSynthesisUtterance({safe_text});
                            msg.rate = 0.95;
                            window.speechSynthesis.speak(msg);
                        }}
                    </script>
                    <div style="font-size: 0.8rem; color: #06b6d4;">🔊 Speaking question via browser voice engine…</div>
                    """,
                    height=35,
                )
                st.session_state[f"use_browser_tts_{current_pos}"] = False

        if q_ref:
            with st.expander("Benchmark Reference Answer (Evaluator Reference)"):
                st.write(q_ref)

    st.write("")

    # 4. Tabbed Response Interface: Text vs Voice vs Evaluation
    t_text, t_voice, t_eval = st.tabs(["✍️ 1. Written / Text Response", "🎙️ 2. Voice & Audio Response", "📊 3. Question Evaluation Report"])

    # ── TAB 1: TEXT RESPONSE ─────────────────────────────────────────────────
    with t_text:
        section_header("Written Technical Answer", "Type your response to evaluate semantic depth and structure")

        # Key for this specific question's answer
        text_draft_key = f"draft_text_{session_id}_{current_pos}"
        existing_answer = text_resp.get("answer_text") or st.session_state.get(text_draft_key, "")

        candidate_answer = st.text_area(
            "Candidate Answer:",
            value=existing_answer,
            height=200,
            placeholder="Type your detailed technical explanation here. Include core concepts, syntax, and relevant examples…",
            key=f"input_text_area_{current_pos}",
        )
        words_count = len(candidate_answer.strip().split()) if candidate_answer.strip() else 0
        st.caption(f"Word Count: {words_count} words (Minimum 3 words required for semantic NLP analysis)")

        c_save, c_eval_btn = st.columns([1, 1], gap="medium")
        with c_save:
            if st.button("💾 Save Answer Draft", key=f"btn_save_draft_{current_pos}", use_container_width=True):
                if candidate_answer.strip():
                    payload = {"answer_text": candidate_answer.strip(), "response_mode": "text"}
                    saved = client.save_response_draft(session_id, current_pos, payload)
                    if saved:
                        st.success("Draft saved successfully to session database.")
                    else:
                        st.error("Failed to save draft.")
                else:
                    st.warning("Type an answer before saving draft.")
        with c_eval_btn:
            if st.button("⚡ Evaluate Text Answer", key=f"btn_eval_text_{current_pos}", type="primary", use_container_width=True):
                if words_count < 3:
                    st.error("Answer is too short to evaluate. Please provide at least 3 words.")
                else:
                    with st.spinner("Analyzing semantic relevance against technical benchmark…"):
                        success, eval_res = client.assess_text(
                            question=q_text,
                            candidate_answer=candidate_answer.strip(),
                            reference_answer=q_ref,
                            session_id=session_id,
                            position=current_pos,
                        )
                        if success:
                            st.success("Text assessment completed and saved to session!")
                            st.session_state[f"last_text_eval_{current_pos}"] = eval_res
                            st.rerun()
                        else:
                            st.error(f"Evaluation error: {eval_res}")

        # Show existing text evaluation if available
        current_text_eval = text_resp.get("text_assessment") or st.session_state.get(f"last_text_eval_{current_pos}")
        if current_text_eval:
            st.write("")
            with st.container(border=True):
                section_header("Semantic NLP Assessment Results", "TF-IDF sublinear vectorization & heuristics")
                q_score = current_text_eval.get("quality_score", 0.0)
                st.metric("Overall Text Quality Score", f"{q_score:.1%}")
                st.progress(float(q_score))
                st.markdown(tier_badge(q_score), unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Reference Cosine", f"{current_text_eval.get('relevance_score', 0):.1%}")
                m2.metric("Question Alignment", f"{current_text_eval.get('question_alignment_score', 0):.1%}")
                m3.metric("Keyword Overlap", f"{current_text_eval.get('keyword_overlap', 0):.1%}")
                m4.metric("Structure Coherence", f"{current_text_eval.get('structure_score', 0):.1%}")

                feedback = current_text_eval.get("feedback", {})
                if feedback.get("strengths"):
                    st.markdown("**Demonstrated Strengths:**")
                    for s_item in feedback["strengths"]:
                        st.success(f"✓ {s_item}")
                if feedback.get("improvements"):
                    st.markdown("**Areas for Coaching:**")
                    for imp in feedback["improvements"]:
                        st.info(f"→ {imp}")

    # ── TAB 2: VOICE RESPONSE ────────────────────────────────────────────────
    with t_voice:
        section_header("Spoken Technical Answer", "Record or upload audio to analyze pacing, pitch, and fluency")

        # ── AI Interviewer Audio Prompt (TTS) ──────────────────────────────
        with st.container(border=True):
            st.markdown('<div class="page-eyebrow">AI INTERVIEWER AUDIO PROMPT</div>', unsafe_allow_html=True)
            v_tts1, v_tts2 = st.columns([1, 2])
            with v_tts1:
                if st.button("🎧 Listen to Interviewer Question", key=f"btn_v_listen_{current_pos}", use_container_width=True):
                    with st.spinner("Loading interviewer audio…"):
                        ok_tts, tts_data = client.get_tts_audio(q_text)
                        if ok_tts and isinstance(tts_data, bytes):
                            st.session_state[tts_q_key] = tts_data
                            st.session_state[f"play_tts_{current_pos}"] = True
                        else:
                            st.session_state[f"use_browser_tts_{current_pos}"] = True
                    st.rerun()
            with v_tts2:
                if st.session_state.get(tts_q_key):
                    st.audio(st.session_state[tts_q_key], format="audio/wav")
                else:
                    st.caption("Click the button to listen to the AI interviewer read this question before recording.")

        # ── STEP 1: Audio Capture ──────────────────────────────────────────
        with st.container(border=True):
            st.markdown('<div class="page-eyebrow">STEP 1 · CAPTURE AUDIO</div>', unsafe_allow_html=True)
            audio_src = st.radio(
                "Audio Input Method:",
                ["🎙️ Microphone Recording", "📁 Audio File Upload"],
                horizontal=True,
                key=f"src_radio_{current_pos}",
            )

            cached_audio_key = f"cached_audio_bytes_{session_id}_{current_pos}"
            cached_name_key = f"cached_audio_name_{session_id}_{current_pos}"

            if "Microphone" in audio_src:
                st.caption("💡 Tip: Speak clearly at a normal pace. Recording starts when you click the microphone.")
                mic_recording = st.audio_input(
                    "Record your spoken answer:",
                    key=f"mic_input_{current_pos}",
                )
                if mic_recording is not None:
                    raw_bytes = mic_recording.getvalue() if hasattr(mic_recording, "getvalue") else mic_recording.read()
                    if raw_bytes:
                        st.session_state[cached_audio_key] = raw_bytes
                        st.session_state[cached_name_key] = getattr(mic_recording, "name", None) or f"mic_q{current_pos}.webm"
            else:
                st.caption("Supported formats: WAV, MP3, M4A, WebM — Maximum recommended: 10 MB")
                file_upload = st.file_uploader(
                    "Upload audio file:",
                    type=["wav", "mp3", "m4a", "webm"],
                    key=f"file_uploader_{current_pos}",
                )
                if file_upload is not None:
                    raw_bytes = file_upload.getvalue() if hasattr(file_upload, "getvalue") else file_upload.read()
                    if raw_bytes:
                        st.session_state[cached_audio_key] = raw_bytes
                        st.session_state[cached_name_key] = file_upload.name

            captured_audio_bytes = st.session_state.get(cached_audio_key)
            audio_name = st.session_state.get(cached_name_key, f"recording_q{current_pos}.wav")

        # ── STEP 2: Playback + Status Banner ──────────────────────────────
        if captured_audio_bytes:
            audio_kb = len(captured_audio_bytes) / 1024
            audio_duration_est = audio_kb / 16  # rough estimate: 16 KB/s for compressed audio
            st.markdown(
                f"""
                <div style="background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.3);
                    border-radius: 8px; padding: 0.65rem 1rem; margin: 0.5rem 0 0.75rem;
                    display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.3rem;">🎵</span>
                        <div>
                            <div style="font-weight: 700; color: var(--iai-primary); font-size: 0.88rem;">
                                Audio Ready — {audio_name}
                            </div>
                            <div style="color: var(--iai-muted); font-size: 0.78rem;">
                                {audio_kb:.1f} KB · ~{audio_duration_est:.0f}s est. duration · Ready for transcription
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_audio_play, col_audio_reset = st.columns([4, 1])
            with col_audio_play:
                st.audio(captured_audio_bytes)
            with col_audio_reset:
                if st.button("🗑️ Reset Audio", key=f"btn_reset_audio_{current_pos}", help="Clear recording"):
                    st.session_state.pop(cached_audio_key, None)
                    st.session_state.pop(cached_name_key, None)
                    st.rerun()

        # ── STEP 3: Transcription ──────────────────────────────────────────
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="page-eyebrow">STEP 2 · TRANSCRIBE WITH WHISPER</div>', unsafe_allow_html=True)
            st.caption("faster-whisper converts your speech to text. You can review and correct the transcript before scoring.")

            tr_key = f"voice_tr_editor_{session_id}_{current_pos}"
            ta_key = f"ta_transcript_{current_pos}"
            existing_transcript = (
                voice_resp.get("corrected_transcript")
                or voice_resp.get("original_transcript")
                or st.session_state.get(tr_key, "")
            )

            # Transcription action button (full width, prominent)
            is_transcribing = st.session_state.get(f"is_transcribing_{current_pos}", False)
            transcribe_btn = st.button(
                "✨ Transcribe Audio with Whisper",
                key=f"btn_transcribe_{current_pos}",
                type="secondary",
                use_container_width=True,
                disabled=(captured_audio_bytes is None or is_transcribing),
            )

            if captured_audio_bytes is None:
                st.info("⬆️ Capture or upload audio in Step 1 to enable transcription.")

            if transcribe_btn and captured_audio_bytes:
                st.session_state[f"is_transcribing_{current_pos}"] = True
                try:
                    import logging as _logging
                    import os as _os
                    _diag_logger = _logging.getLogger("interviewai.transcribe_diag")
                    _diag_logger.setLevel(_logging.INFO)
                    if not _diag_logger.handlers:
                        _diag_handler = _logging.FileHandler(
                            _os.path.join(_os.path.expanduser("~"), "interviewai_transcribe_diag.log")
                        )
                        _diag_handler.setFormatter(_logging.Formatter("%(asctime)s - %(message)s"))
                        _diag_logger.addHandler(_diag_handler)
                    _diag_logger.info(
                        "BUTTON_CLICKED audio_bytes=%d audio_name=%s session_id=%s position=%d",
                        len(captured_audio_bytes), audio_name, session_id, current_pos
                    )
                    with st.spinner("🔄 Transcribing audio with faster-whisper…"):
                        # Prime Whisper decoder with question and domain keywords
                        q_prompt = f"Technical mock interview. Question: {q_text}. Expected concepts: {', '.join(q_concepts)}."
                        ok, t_res = client.transcribe_audio(audio_name, captured_audio_bytes, prompt=q_prompt)
                        _diag_logger.info(
                            "API_RESPONSE ok=%s type=%s keys=%s transcript_len=%d",
                            ok, type(t_res).__name__,
                            list(t_res.keys()) if isinstance(t_res, dict) else str(t_res)[:100],
                            len(t_res.get("transcript", "")) if isinstance(t_res, dict) else 0,
                        )
                        if ok:
                            raw_transcript = t_res.get("transcript", "")
                            detected_lang = t_res.get("detected_language", "unknown")
                            st.session_state[tr_key] = raw_transcript
                            st.session_state[ta_key] = raw_transcript
                            st.session_state[f"orig_tr_{current_pos}"] = raw_transcript
                            st.session_state[f"det_lang_{current_pos}"] = detected_lang
                            _diag_logger.info(
                                "TRANSCRIPT_STORED key=%s transcript=%s lang=%s",
                                tr_key, raw_transcript[:120], detected_lang
                            )
                            st.success(f"✅ Transcription complete (language: {detected_lang})! Review the text below and correct any errors before scoring.")
                            st.rerun()
                        else:
                            err_str = str(t_res)
                            if "429" in err_str or "quota" in err_str.lower() or "rate limit" in err_str.lower():
                                st.warning(
                                    "⚠️ **Transcription rate limit exceeded.** Please wait a moment and retry."
                                )
                            elif "404" in err_str or "model" in err_str.lower() and "not found" in err_str.lower():
                                st.error(
                                    "⚠️ **Transcription model not found.** Please check the server configuration and retry."
                                )
                            elif "400" in err_str or "invalid request" in err_str.lower() or "bad request" in err_str.lower():
                                st.error("⚠️ **Invalid transcription request.** Please check the audio format and retry.")
                            elif "415" in err_str or "unsupported" in err_str.lower():
                                st.error("⚠️ **Audio format not supported.** Please re-record or convert to WAV/MP3 and retry.")
                            elif "503" in err_str or "not configured" in err_str.lower() or "offline" in err_str.lower():
                                st.error(
                                    "⚠️ **Transcription service unavailable.** Please check the backend server and retry."
                                )
                            elif "401" in err_str or "invalid key" in err_str.lower() or "api key" in err_str.lower():
                                st.error("⚠️ **Authentication error.** Please check the server configuration.")
                            else:
                                st.error(f"⚠️ Transcription failed: {t_res}")
                except Exception as exc:
                    st.session_state[f"is_transcribing_{current_pos}"] = False
                    st.error(f"⚠️ Transcription error: {str(exc)}")
                finally:
                    st.session_state[f"is_transcribing_{current_pos}"] = False

            # Transcript text area - always visible for manual entry fallback
            st.write("")
            if ta_key not in st.session_state:
                st.session_state[ta_key] = st.session_state.get(tr_key, existing_transcript)

            editable_transcript = st.text_area(
                "Reviewed Transcript:",
                height=130,
                key=ta_key,
                placeholder="Transcribed speech appears here after Whisper processing, or type a manual transcript to proceed…",
            )
            # Sync session state
            st.session_state[tr_key] = editable_transcript

            word_count_v = len(editable_transcript.strip().split()) if editable_transcript.strip() else 0
            st.caption(f"Transcript word count: **{word_count_v}** words (minimum 3 required for multimodal scoring)")

            # Transcript draft action buttons
            col_save_draft, col_draft_info = st.columns([1, 2])
            with col_save_draft:
                if st.button("💾 Save Transcript Draft", key=f"btn_save_tr_{current_pos}", use_container_width=True):
                    if editable_transcript.strip():
                        saved = client.save_response_draft(session_id, current_pos, {
                            "answer_text": editable_transcript.strip(),
                            "response_mode": "voice",
                        })
                        if saved:
                            st.success("Draft saved to session database.")
                        else:
                            st.error("Could not save draft.")
                    else:
                        st.warning("Type or transcribe an answer before saving draft.")
            with col_draft_info:
                if voice_resp.get("corrected_transcript") or voice_resp.get("original_transcript"):
                    st.caption("✅ A saved transcript exists for this question from a previous session.")

        # ── STEP 4: Multimodal Assessment ─────────────────────────────────
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="page-eyebrow">STEP 3 · MULTIMODAL SCORING</div>', unsafe_allow_html=True)
            st.caption("Combines semantic NLP scoring (text) with Librosa acoustic feature analysis (audio) for a composite practice score.")

            assess_voice_btn = st.button(
                "🚀 Assess Voice & Generate Multimodal Score",
                key=f"btn_assess_voice_{current_pos}",
                type="primary",
                use_container_width=True,
                disabled=(captured_audio_bytes is None or word_count_v < 3),
            )

            if captured_audio_bytes is None:
                st.info("Audio required. Capture audio in Step 1 first.")
            elif word_count_v < 3:
                st.info("Transcript required. Transcribe audio or type at least 3 words in Step 2 first.")

            if assess_voice_btn and captured_audio_bytes and word_count_v >= 3:
                with st.spinner("⚙️ Extracting Librosa acoustic features and scoring multimodal response…"):
                    orig_tr = st.session_state.get(f"orig_tr_{current_pos}", editable_transcript)
                    ok, m_res = client.assess_multimodal(
                        question=q_text,
                        corrected_transcript=editable_transcript.strip(),
                        audio_bytes=captured_audio_bytes,
                        audio_filename=audio_name,
                        reference_answer=q_ref,
                        original_transcript=orig_tr,
                        session_id=session_id,
                        position=current_pos,
                    )
                    if ok:
                        st.success("✅ Multimodal analysis complete! Score and acoustic measurements saved to session.")
                        st.session_state[f"last_multi_eval_{current_pos}"] = m_res
                        st.rerun()
                    else:
                        st.error(f"Multimodal assessment error: {m_res}")

    # ── TAB 3: QUESTION MULTIMODAL EVALUATION REPORT ─────────────────────────
    with t_eval:
        section_header("Question Multimodal Assessment Report", f"Synchronized evaluation for Question {current_pos}")

        multi_report = voice_resp.get("multimodal_report") or st.session_state.get(f"last_multi_eval_{current_pos}")
        audio_meas = voice_resp.get("audio_measurements") or {}

        if multi_report:
            # 1. Composite Practice Score
            text_part = multi_report.get("text_assessment", {})
            voice_part = multi_report.get("voice_assessment", {})
            t_qual = text_part.get("quality_score", 0.0)
            
            # Rule-based transparent score fusion
            # Normalize fluency (1-5 -> 0.2-1.0) and prosody (1-5 -> 0.2-1.0)
            pause_r = voice_part.get("pause_ratio", 0.15) or 0.15
            fluency_norm = max(0.0, min(1.0, 1.0 - pause_r))
            pitch_std = voice_part.get("pitch_std_hz", 15.0) or 15.0
            prosody_norm = max(0.0, min(1.0, pitch_std / 50.0))
            
            composite_val = 0.50 * t_qual + 0.25 * fluency_norm + 0.25 * prosody_norm

            r_col1, r_col2 = st.columns([1.2, 1.8], gap="large")
            with r_col1, st.container(border=True):
                st.markdown('<div class="page-eyebrow">PRACTICE COMPOSITE SCORE</div>', unsafe_allow_html=True)
                st.metric("Multimodal Practice Score", f"{composite_val:.1%}")
                st.progress(composite_val)
                st.markdown(tier_badge(composite_val), unsafe_allow_html=True)
                st.caption("Formula: `0.50 × TextQuality + 0.25 × Fluency + 0.25 × Prosody`")
                with st.expander("Formula Transparency & Limitations"):
                    st.caption("This is a deterministic heuristic composite designed exclusively for mock interview practice feedback. It does NOT represent an automated hiring recommendation or a psychometric judgment.")

            with r_col2, st.container(border=True):
                st.markdown('<div class="page-eyebrow">ACOUSTIC & VOCAL MEASUREMENTS</div>', unsafe_allow_html=True)
                a1, a2, a3 = st.columns(3)
                a1.metric("Speaking Rate", f"{voice_part.get('speaking_rate_wpm', '—')} WPM")
                a2.metric("Pause Ratio", f"{voice_part.get('pause_ratio', 0):.1%}")
                a3.metric("Fillers Detected", f"{voice_part.get('filler_word_count', 0)}")

                a4, a5, a6 = st.columns(3)
                mean_pitch = voice_part.get("mean_pitch_hz")
                a4.metric("Pitch (Mean F₀)", f"{mean_pitch:.1f} Hz" if mean_pitch else "Unvoiced")
                pitch_std_val = voice_part.get("pitch_std_hz")
                a5.metric("Pitch Variation", f"{pitch_std_val:.1f} Hz" if pitch_std_val else "Unvoiced")
                rms_val = voice_part.get("mean_energy_rms")
                a6.metric("RMS Energy", f"{rms_val:.4f}" if rms_val else "N/A")

            # Spectral MFCC
            mfcc_means = voice_part.get("mfcc_mean", [])
            if mfcc_means and any(v is not None for v in mfcc_means):
                with st.container(border=True):
                    section_header("MFCC Spectral Timbre Profile", "13 Mel-Frequency Cepstral Coefficients extracted with Librosa")
                    clean_mfcc = [float(v) if v is not None else 0.0 for v in mfcc_means]
                    st.bar_chart(pd.DataFrame({"Coefficient Mean": clean_mfcc}, index=[f"MFCC {i+1}" for i in range(len(clean_mfcc))]), height=200)

            # Qualitative Feedback
            fb = multi_report.get("feedback", {})
            st.write("")
            f_col1, f_col2 = st.columns(2, gap="medium")
            with f_col1, st.container(border=True):
                st.markdown("**Demonstrated Strengths:**")
                for st_item in fb.get("strengths", []):
                    st.success(f"✓ {st_item}")
            with f_col2, st.container(border=True):
                st.markdown("**Areas for Coaching:**")
                for imp_item in fb.get("areas_for_improvement", []):
                    st.info(f"→ {imp_item}")
        else:
            empty_state("📊", "No Multimodal Evaluation Yet", "Record or upload audio and click 'Assess Voice & Generate Multimodal Score' to view combined diagnostics.")

    # 5. Bottom Completion & Final Report Action
    st.write("")
    st.write("")
    with st.container(border=True):
        f1, f2 = st.columns([2.5, 1], gap="medium")
        with f1:
            assessed_cnt = sum(1 for q in questions if q.get("status") == "assessed")
            st.markdown(f"**Interview Completion Status:** `{assessed_cnt} of {total_q}` questions assessed.")
            st.caption("When you have evaluated all questions, generate the final comprehensive report.")
        with f2:
            if st.button("🏁 Finalize & View Report ➔", type="primary", use_container_width=True):
                with st.spinner("Compiling full session report across all interview responses…"):
                    ok, rep = client.create_final_report(session_id)
                    if ok:
                        st.session_state["view_report_session_id"] = session_id
                        nav_to_fn("Assessment History")
                    else:
                        st.error(f"Failed to generate final report: {rep}")

