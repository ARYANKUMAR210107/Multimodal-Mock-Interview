# Voice Interview Manual Browser Test

Use a current desktop browser and allow microphone access when prompted. Start FastAPI at `http://127.0.0.1:8000` and Streamlit at `http://127.0.0.1:8501` before testing.

## Microphone workflow

1. Open the Streamlit page and select **Voice Interview**.
2. Select an interview question and leave the **Record with microphone** source selected.
3. Click the audio recorder, speak for at least three seconds, then stop the recording.
4. Confirm the captured audio player appears and plays the recording. Confirm the UI shows a captured file and size.
5. Click **Transcribe with Gemini**. Confirm a transcript appears in the editable transcript box. This step sends audio to Gemini and may incur API usage.
6. Edit a few words in the transcript. Confirm the edits remain visible after the Streamlit rerun.
7. Click **Assess this response**. Confirm the report shows the selected question, one response ID, text metrics, audio duration, speaking rate, pitch when measurable, RMS, MFCCs, and pause measurements. The report should explain that it has no combined numeric score.
8. Open **Multimodal Report** and confirm it shows the same response ID and reviewed transcript.

## Upload workflow and validation

1. Select **Upload audio** and upload a valid WAV recording. Confirm the player appears and the assessment completes.
2. Repeat with a valid MP3, M4A, or WebM recording where available.
3. Try an unsupported file type, an empty file, a file larger than 25 MB, and a corrupted audio file. Confirm the app displays a useful error and remains usable.
4. Try a recording shorter than 0.25 seconds. The backend should reject it with a minimum-duration error.

## Response isolation

1. Complete or partially complete a response, then change the selected interview question.
2. Confirm the old recording, transcript, and report are cleared before assessing the new question.
3. Record or upload a new answer and confirm its report contains the new question and response ID.

Microphone permission and actual playback depend on browser/device hardware and must be confirmed in the browser. Automated tests cover upload validation, mocked transcription, session reset behavior, acoustic feature extraction, and report generation; they do not simulate a physical microphone.
