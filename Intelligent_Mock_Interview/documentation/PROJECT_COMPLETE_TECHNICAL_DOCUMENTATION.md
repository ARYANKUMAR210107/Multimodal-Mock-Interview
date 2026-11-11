# Complete Technical Documentation

## Audit scope and evidence

This document was originally written from a 2 November 2026 checkout. The implementation has since added Gemini transcription and persistent student/interview management. Current workflow and verification status are maintained in `README.md` and `IMPLEMENTATION_STATUS.md`; historical audit observations below should be read with that update in mind. Evaluation figures are still read from saved model metadata and reports, not from a fresh training run.

Paths in this document are relative to `Intelligent_Mock_Interview/` unless explicitly called repository-root files.

## 1. Purpose and overview

The project is a local academic mock-interview practice application. A candidate can practice technical question answering, compare an answer with a reference solution, inspect simple acoustic measurements from a voice recording, and view a separate prediction based on soft-skill ratings. The intended audience is students preparing for interviews, project demonstrators, and instructors reviewing a supervised-learning prototype.

The objectives represented in code are:

1. Predict the dataset's `CLASS` label from interview-assessor soft-skill ratings.
2. Provide categorized technical questions and compare candidate text with a stored reference answer.
3. Extract acoustic measurements from an uploaded or browser-recorded answer.
4. Transcribe a finished voice recording through Google's Gemini API and let the candidate correct it before text scoring.
5. Produce a transparent report containing text scores and voice measurements.

The system is called multimodal because its current Voice Interview flow takes **two different input modalities** from one answer: audio and the text transcript. It evaluates the transcript with text retrieval heuristics and the audio with signal processing. The two sets of results are placed in one report; no model has been trained on aligned audio/text interview examples. The employability classifier is a separate model with a separate dataset and target.

## 2. Current capabilities

- Streamlit dashboard with employability prediction, technical text interview, Voice Interview, a legacy Multimodal Report page, and model/EDA displays (`frontend/app.py`).
- FastAPI service that loads the saved Gradient Boosting classifier, fitted TF-IDF vectorizer, audio analyzer, and reference Q&A records at import/startup (`backend/main.py`).
- Six trained classical scikit-learn classifiers, stratified 60/20/20 data splits, five-fold training-set cross-validation, validation comparison, and held-out test reporting (`src/employability_pipeline.py`).
- Text scoring using TF-IDF cosine similarity, Jaccard keyword overlap, fixed structural heuristics, and a weighted composite (`src/text_pipeline.py`).
- Librosa extraction of duration, energy, pause ratio, pitch estimates, and 13 MFCC means (`src/audio_feature_extractor.py`).
- Microphone record-then-transcribe and file-upload flows, edited-transcript scoring, approximate transcript WPM, filler-pattern count, and a report (`frontend/app.py`, `backend/main.py`).
- Dataset EDA and precomputed figures/reports (`src/eda_runner.py`, `reports/`).
- Persistent students, authored questions, linked interview sessions, responses, assessments, and versioned reports in SQLite (`src/database.py`, `backend/management_api.py`).

## 3. Technology stack actually used

| Technology | Where used | Role |
|---|---|---|
| Python | All application and analysis modules | Main implementation language. |
| Streamlit | `frontend/app.py` | Browser UI, forms, microphone/audio controls, charts, and session state. |
| FastAPI + Uvicorn | `backend/main.py` | REST service and local ASGI server. |
| Pydantic | `backend/main.py` | Validation and response schemas for existing JSON endpoints. |
| Google Gen AI Python SDK | `backend/main.py` | Uploads a recording and requests verbatim transcription through Gemini Interactions. |
| SQLAlchemy | `src/database.py`, `backend/management_api.py` | Persists students, question bank records, sessions, responses, assessments, and report versions in SQLite. |
| python-dotenv | `backend/main.py` | Loads the repository-root `.env`; the key is not in frontend code. |
| Requests | `frontend/app.py` | Streamlit-to-local-FastAPI multipart HTTP requests. |
| pandas, NumPy | data pipelines, UI, API | Tabular data, numeric arrays, feature construction, and metrics support. |
| scikit-learn | `src/employability_pipeline.py`, `src/text_pipeline.py` | Classifiers, preprocessing pipelines, TF-IDF, cosine similarity, splits, CV, and metrics. |
| joblib | `models/`, UI, API, pipelines | Persist and reload estimators/vectorizer. The `.joblib` files are serialized Python artifacts, not source. |
| Librosa | `src/audio_feature_extractor.py` | Audio decoding/resampling, RMS, non-silent intervals, YIN pitch, and MFCC calculations. |
| SoundFile | audio test and extractor imports | Audio file I/O support; `sf` is imported but not used in the extractor's main methods. |
| Matplotlib + Seaborn | `src/eda_runner.py`, `src/employability_pipeline.py`, UI | EDA plots, model comparison/confusion matrices, and feature importance figures. |
| SciPy | `src/employability_pipeline.py` | `scipy.stats.pointbiserialr` for feature/target correlation investigation. |
| Hugging Face Hub | `download_dataset.py` | Downloads the AI interview-question Parquet shards. |
| Excel/OpenPyXL, Parquet/PyArrow | dataset readers and dependencies | Read the spreadsheet and Parquet corpus. |
| Jupyter + ipykernel | `notebooks/*.ipynb` | Interactive experiment notebooks; notebooks contain no saved cell outputs in this checkout. |
| unittest + FastAPI TestClient | `src/test_voice_interview_api.py` | Mock-based automated API checks. |

No deep-learning framework, transformer library, or embedding service is used for candidate scoring. The project does include a SQLite database for student and interview records. It does not implement authentication, a container, or a deployment configuration. NLTK appears in requirements but is not imported by the current source. Package presence alone is not evidence of runtime use.

## 4. Architecture and working workflow

### Application entry points

- Backend: from the project directory, `uvicorn backend.main:app --reload --port 8000`.
- Frontend: `streamlit run frontend/app.py`.
- EDA: `python src/eda_runner.py` (script executes when run; it writes the report and figures).
- Employability training: `python src/employability_pipeline.py` (script executes a complete training/reporting pipeline and overwrites model/report outputs; the audit did not run it).
- Text vectorizer/report pipeline: `python src/text_pipeline.py`.
- Dataset download: `python download_dataset.py` (downloads files from Hugging Face; its output path is relative to the working directory).

### Voice Interview data flow

1. The candidate chooses a question from `src/interview_repository.py` and selects microphone or upload in Streamlit.
2. Streamlit retains the selected recording bytes in its session state and displays playback.
3. A button posts multipart audio to `POST /api/audio/transcribe`.
4. FastAPI checks extension, signature, MIME compatibility, empty input, and the 25 MiB limit. It uploads the recording through Google's Gen AI SDK and requests verbatim transcription from `gemini-3.5-transcribe`; the response transcript is returned to the Streamlit editor.
5. The candidate edits the transcript.
6. A second button sends question, reference answer, corrected transcript, and the same audio to `POST /api/interview/multimodal-report`.
7. FastAPI runs `InterviewTextAssessor.assess_answer`, writes the recording bytes briefly to a suffixed temporary file for `CandidateAudioAnalyzer`, calculates WPM/filler count from the corrected text, removes the temporary file, and returns both modality results.
8. Streamlit displays the resulting report. Employability prediction is not part of this voice report.

Google's API is called only for the transcription stage. Text and audio feature calculations happen in this project. The microphone function is record-then-transcribe; it is not a live caption stream.

## 5. Core source modules

| File | Responsibility |
|---|---|
| `backend/main.py` | One FastAPI app, environment loading, model initialization, upload validation, transcription, report creation, and existing v1 prediction/text/audio endpoints. |
| `backend/management_api.py`, `src/database.py` | Persistent management endpoints, SQLAlchemy models, SQLite lifecycle, and idempotent question-bank seeding. |
| `frontend/app.py` | One Streamlit app, cached resource loading, navigation, scoring forms, audio capture/upload, local API requests, and visual reporting. |
| `src/employability_pipeline.py` | Spreadsheet preprocessing, feature selection/engineering, stratified splitting, classifier fitting, CV/evaluation, plots, and serialization. |
| `src/text_pipeline.py` | Parquet chat-template parsing, text cleanup, TF-IDF fitting, answer scoring, and report generation. |
| `src/audio_feature_extractor.py` | `CandidateAudioAnalyzer` file/array loading and acoustic feature extraction. |
| `src/interview_repository.py` | Hard-coded categorized questions and reference answers consumed by the Streamlit interview pages. |
| `src/eda_runner.py` | Dataset summaries and plots for employability, interview Q&A, and OpenEnded annotations. |
| `download_dataset.py` | Hugging Face shard downloader. |
| `src/test_database_management.py`, `src/test_end_to_end_audit.py`, `src/test_voice_interview_api.py` | Persistent management, linked UI/API workflow, transcription, and report tests. |
| `src/test_fixes.py`, `src/test_ui_and_features.py` | Older scripts with hard-coded local Windows project paths; they may generate a WAV artifact when run. |

## 6. Models and outputs

Six classical estimators are trained on the employability spreadsheet. The selected model is Gradient Boosting, chosen by **validation macro F1**, not test score. All six final estimators are saved in `models/`; `employability_model_meta.json` records their validation/test metrics and feature list. The API and current UI load the Gradient Boosting artifact.

The TF-IDF vectorizer is a separately serialized retrieval vocabulary/weighting model. It is not a trained answer-quality classifier. The audio extractor has no saved model artifact and no learned weights.

Current saved held-out test results for Gradient Boosting are accuracy 0.7688, positive-class precision 0.7419, positive-class recall 0.9220, positive-class F1 0.8222, macro F1 0.7460, and ROC-AUC 0.8248. These values describe the provided rating dataset only, not real job placement outcomes.

## 7. Important boundaries and limitations

- Dataset A trains only employability classification. Dataset B supplies Q&A references and vocabulary; it has no human answer-quality labels. Dataset C supplies speech-label/metadata EDA only; there is no OpenEnded WAV archive in the checkout and no acoustic model training pipeline.
- Nothing merges A, B, and C into a multimodal training set. The custom interview assessment CSV is an empty template.
- The reference-answer corpus is LLM-generated according to its dataset documentation and is not human-verified. TF-IDF is lexical weighting, not semantic understanding.
- Text scores and audio's original estimated fluency/prosody formulas are heuristics, not human-validated measurements. The new report omits those heuristic scores and emphasizes measurable indicators.
- Filler words are counted with a short fixed English regex; WPM uses the corrected transcript's entire word count divided by total audio duration, not a speech-only interval or ASR timestamps.
- F0 and pause measurements depend on codec decoding, waveform quality, voice range, and the extractor's thresholds. M4A/WebM feature extraction depends on local decoder support.
- `reports/` contains generated results. Dataset documentation and README have historic statements that conflict with current artifacts/code; see `PROJECT_FINAL_AUDIT.md`.
- The project is a local prototype, not a production hiring decision system. Candidate and interview records persist in SQLite, but the app does not provide account authentication or production deployment hardening.

## 8. Detailed companion documents

- [Architecture and file map](PROJECT_ARCHITECTURE.md)
- [Algorithms and mathematics](ALGORITHMS_AND_MATHEMATICS.md)
- [Dataset and EDA analysis](DATASET_AND_EDA_ANALYSIS.md)
- [Viva questions and answers](VIVA_QUESTIONS_AND_ANSWERS.md)
- [Final technical audit](PROJECT_FINAL_AUDIT.md)
