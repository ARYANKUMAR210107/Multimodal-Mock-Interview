"""Session-state helpers for keeping voice interview responses isolated."""


def reset_for_question_change(session_state, question: str) -> bool:
    """Clear response-specific data when the selected prompt changes.

    Call before rendering audio and transcript widgets so their keyed values can
    be safely cleared before Streamlit instantiates those widgets for the run.
    Returns True when a previous/different question context was reset.
    """
    if session_state.get("voice_question_context") == question:
        return False

    session_state["voice_question_context"] = question
    for key in (
        "voice_report", "voice_report_context", "voice_audio_digest",
        "voice_audio_bytes", "voice_audio_name", "voice_audio_mime",
        "voice_recording", "voice_upload",
    ):
        session_state.pop(key, None)
    session_state["pending_voice_transcript"] = ""
    session_state["voice_transcript_editor"] = ""
    return True
