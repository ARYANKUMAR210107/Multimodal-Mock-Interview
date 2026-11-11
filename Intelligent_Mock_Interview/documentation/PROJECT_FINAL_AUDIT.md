# Project Final Technical Audit

> **Historical snapshot:** this audit describes the checkout as inspected on
> 2 November 2026 and predates the Gemini transcription and persistent SQLite
> management implementation. Its OpenAI and no-database statements are not
> current. Use `README.md` for current system behavior and
> `IMPLEMENTATION_STATUS.md` for the latest verification record.

**Audit basis:** Current local checkout inspected 2 November 2026. No application code, dataset, model, or existing report was modified for this audit. The six files in this audit set are the only created project files. Metrics come from checked-in metadata/reports/figures; test status comes from the latest recorded implementation checks, not a rerun during this read-only audit.

## Executive summary

This is a functional academic prototype with three distinct tracks:

1. A supervised employability-label classifier trained on 2,982 spreadsheet rows.
2. A technical answer evaluator based on TF-IDF cosine similarity, Jaccard overlap, and structural heuristics compared with reference answers.
3. A voice interview flow that submits completed recordings to OpenAI `gpt-transcribe`, lets the candidate edit the transcript, extracts Librosa features, and returns a rule-based report.

The word **multimodal** describes the combination of audio-derived and text-derived results in the new report. There is no learned joint model. There is no acoustic model trained on OpenEnded audio labels; the archive is missing. The employability classifier does not participate in the voice report. Text and voice values are not human-calibrated candidate-quality predictions.

## Component status

| Component | Status | Actual implementation | Pending work / audit concern |
|---|---|---|---|
| Frontend | Partially implemented | Streamlit dashboard, employability page, technical text page, new Voice Interview page, benchmark page. | Browser workflow was startup-tested, not manually exercised with camera/mic controls. Old audio code is still present but its “Audio Assessment” navigation option was replaced; legacy Multimodal Report remains selectable and can use neutral defaults for missing modality. |
| Backend | Partially implemented | One FastAPI application; model loading, question, employability, text, audio, old multimodal, new transcription and report routes. | No auth/rate limiting; broad CORS; different validation across old/new routes; synchronous OpenAI SDK call inside an async route can block the event loop during the external call. |
| Employability ML | Fully implemented for the supplied dataset | Six scikit-learn pipelines, train-only 5-fold CV, validation selection by macro F1, held-out test evaluation, six serialized estimators and metadata. | External validation is absent; labels/inputs are subjective ratings; target is not an actual employment outcome. Avoid interpreting model probabilities as calibrated placement chances. |
| NLP evaluation | Partially implemented | TF-IDF (1–2 grams, sublinear TF, up to 10,000 terms), cosine similarity, Jaccard overlap, fixed length/connector structure rule, weighted composite. | No human candidate-quality labels; heuristic tiers are not validated. Vectorizer fit includes the nominal eval Q&A split. No BERT, embeddings, or semantic model. |
| Audio analysis | Partially implemented | Runtime Librosa extraction of duration, thresholded nonsilent intervals/pause ratio, RMS, YIN F0 stats, 13 MFCC means; new endpoint derives WPM/filler count from corrected transcript. | No trained audio estimator; thresholds/pitch depend on recording/codecs; filler regex and WPM are approximate. OpenEnded WAV archive is unavailable. |
| OpenAI API | Fully implemented in code; live service unverified | `POST /api/audio/transcribe` uses SDK model `gpt-transcribe`, key from root `.env`, validates file, size, extension/signature/MIME, and maps API/network errors to HTTP responses. | No real API call was made in checks; API key/account/billing/network behavior is unverified. |
| Multimodal report | Partially implemented | `POST /api/interview/multimodal-report` evaluates corrected transcript and analyzes same uploaded audio; feedback uses text scores, pause thresholds and detected fillers. UI labels report “Rule-Based Multimodal Assessment”. | It deliberately returns no combined numeric score; report is unvalidated against a human rubric. UI can retain a report if question selection changes, risking stale report context. |
| Testing | Partially implemented | Six new unittest API tests use mocked OpenAI/audio and cover missing key, gpt-transcribe call, validation, report transcript routing, legacy text and employability endpoints. Python compile, FastAPI startup, and Streamlit startup were recorded as successful in the preceding implementation turn. | No live microphone/browser interaction or real OpenAI transcription. Older scripts use fixed Windows paths; some older test scripts create/overwrite a sample WAV. This audit did not rerun tests. |
| Documentation | Partially implemented before this audit; improved by this audit set | README has app setup and voice instructions; changelog and reports describe prior work. Six detailed audit documents are created here. | Existing dataset guide still says no model training has begun; some EDA “Next” sections describe future work as if project status. Requirement versions are ranges without a lockfile. |

## Model performance from saved outputs

The training script selects Gradient Boosting by **validation macro F1 = 0.7470**. The best-validation model is then evaluated on the held-out 597 rows. `Precision`, `Recall`, and `Binary F1` below are for positive label `Employable`; `Macro F1` averages both class F1 values.

| Model | Validation accuracy | Validation macro F1 | Validation ROC-AUC | Test accuracy | Test precision | Test recall | Test binary F1 | Test macro F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5796 | 0.5764 | 0.6001 | 0.5930 | 0.6604 | 0.6127 | 0.6357 | 0.5873 | 0.6316 |
| Decision Tree | 0.6432 | 0.6431 | 0.7069 | 0.6985 | 0.7139 | 0.8006 | 0.7548 | 0.6817 | 0.7766 |
| Random Forest | 0.7052 | 0.7034 | 0.8075 | 0.7420 | 0.7697 | 0.7919 | 0.7806 | 0.7338 | 0.8132 |
| SVM | 0.7303 | 0.7215 | 0.7700 | 0.7588 | 0.7440 | 0.8902 | 0.8105 | 0.7394 | 0.7926 |
| Gradient Boosting | **0.7638** | **0.7470** | **0.8347** | **0.7688** | 0.7419 | **0.9220** | **0.8222** | **0.7460** | **0.8248** |
| KNN | 0.6918 | 0.6841 | 0.7898 | 0.7119 | 0.7574 | 0.7399 | 0.7485 | 0.7056 | 0.7849 |

The recorded five-fold CV on the training split has Gradient Boosting at accuracy `0.7634 ± 0.0125` and macro F1 `0.7425 ± 0.0161`, highest among listed models. Logistic Regression is lowest by CV accuracy/macro F1 and by test accuracy, macro F1, and AUC. “Worst” depends on metric: Decision Tree has lower test binary F1 than KNN, for example.

### Test confusion matrices

Rows are actual `[Employable, LessEmployable]`; columns are predicted in the same order. These exact counts are read from `reports/figures/E8_confusion_matrices_test.png`.

| Model | Confusion matrix | Main observation |
|---|---|---|
| Logistic Regression | `[[212,134],[109,142]]` | Weakest overall test accuracy; misses many Employable cases. |
| Decision Tree | `[[277,69],[111,140]]` | Better recall than LR but many LessEmployable false positives. |
| Random Forest | `[[274,72],[82,169]]` | More balanced separation than single tree in this split. |
| SVM | `[[308,38],[106,145]]` | High Employable recall, moderate LessEmployable errors. |
| Gradient Boosting | `[[319,27],[111,140]]` | Highest Employable recall; 111 false Employable predictions among 251 LessEmployable cases. |
| KNN | `[[256,90],[82,169]]` | Lower Employable recall than GB/SVM, fewer LessEmployable false positives than GB. |

These matrices clarify that the selected model's recall (0.922) comes with a nontrivial false-positive count. The intended threshold/cost tradeoff is not optimized in source.

## Confirmed issues and inconsistencies

### Accuracy of claims

- `documentation/DATASET_DOCUMENTATION.md` is explicitly stale: it says “No model training has begun,” but six trained model artifacts, model metadata, and evaluation reports are present.
- OpenEnded corpus documentation mentions intended audio quality training, but current code only analyzes CSV annotation metadata; `wav.zip` is absent and no model training code joins these labels to features.
- `reports/eda_summary_report.txt` includes “Next” plans for clustering and speech regression; those plans are not implemented by the present source.
- The project has no ground-truth labels for candidate text scores, so words such as “grade,” “accuracy,” “proficient,” and “strong placement candidate” should be presented as heuristic UI tiers rather than validated judgments.

### API/UI correctness and security

- The older `/api/v1/assess/multimodal` predicts employability along with text and optional audio, while the new report route does not. Its name can therefore be confused with the interview-specific multimodal report. It lacks the newer route's comprehensive model guards/file validation and can fail when a resource is unavailable.
- The separate Streamlit “Multimodal Report” page substitutes `0.50` for an unevaluated modality and computes a composite. That can represent a partial session as a complete score. The Voice Interview page is the appropriate integrated path.
- The Voice Interview report is stored in session state and reset when the audio hash changes, but changing the selected question alone does not invalidate it. A stale report can remain visible beside a newly selected question.
- CORS is configured as `allow_origins=["*"]` and `allow_credentials=True`; no API authentication, per-user limits, or rate limiting exists. This is not suitable for unrestricted public deployment.
- New routes cap audio at 25 MiB and inspect signatures, but legacy audio routes are less constrained. A reverse proxy/body limit would still be prudent for public exposure because multipart parsing happens before endpoint logic.
- The transcription route uses a synchronous OpenAI client from an `async` endpoint. Under concurrent requests this can block the event loop; use an async client or move the call off-loop for a deployed service.
- The new code does not log the API key and has no route that returns it. `.env` is ignored by the root `.gitignore`; `.env.example` is safe placeholder content. Ensure a real key is never pasted into source, docs, or frontend state.
- Audio in Streamlit session state is transient application memory, but is transmitted to the backend and then OpenAI for transcription. The app has no consent workflow or persistent deletion/audit controls. Candidate audio should be treated as sensitive.
- `joblib.load()` assumes trusted local model artifacts. Do not load artifacts obtained from untrusted sources.

### Runtime and reproducibility

- The exact dependency environment is not pinned: requirements use lower bounds and there is no lockfile. Scikit-learn/joblib serialization can be version-sensitive.
- Model-training and EDA scripts have top-level side effects; importing them can create directories and overwrite saved outputs.
- Old test files hard-code `C:\Users\Asus\OneDrive\Desktop\intelligent_supervised\Intelligent_Mock_Interview`; this reduces portability.
- The Streamlit page loads model metadata, Parquet questions, and a direct model even where pages use API/local paths differently. This duplicates resource loading and increases startup coupling.
- The test report claims historical script checks but those two older scripts were not run in this audit. New focused API tests were run during the prior implementation work.
- Librosa codec support is installation-dependent for M4A/WebM. Upload acceptance/transcription support does not guarantee local acoustic decoding support.

## Test/evidence summary

| Check | Evidence/status |
|---|---|
| New voice API unit tests | Six tests passed in the preceding implementation turn; OpenAI and audio analyzer are mocked. |
| Python syntax compilation | `compileall` passed in the preceding implementation turn. |
| FastAPI boot | Uvicorn logged “Application startup complete” and bound to local test port 18500; then stopped intentionally. |
| Streamlit boot | Streamlit printed local URL on test port 18501; then stopped intentionally. |
| Missing OpenAI key | Covered by test: endpoint returns 503 without including credential text. |
| Real OpenAI call | Not tested; requires user's API key/account/network and may incur usage. |
| Actual browser microphone and editing | Not manually tested in browser. |
| Actual M4A/WebM Librosa decoding | Not confirmed; environment codec availability varies. |
| Fresh full employability training | Not run during this audit; saved outputs inspected instead. |

## Practical viva/demo framing

Use the saved result honestly: “On the project’s 597-row held-out test split, Gradient Boosting reached 76.88% accuracy and 0.8248 AUC while being selected by validation macro F1. The voice demo transcribes a completed recording with OpenAI and reports a transcript-based TF-IDF score plus Librosa measurements. The latter are rule-based, not a jointly trained model, and require human validation before being treated as assessment.”

## Companion files

- [Complete technical documentation](PROJECT_COMPLETE_TECHNICAL_DOCUMENTATION.md)
- [Architecture and file map](PROJECT_ARCHITECTURE.md)
- [Algorithms and mathematics](ALGORITHMS_AND_MATHEMATICS.md)
- [Dataset and EDA analysis](DATASET_AND_EDA_ANALYSIS.md)
- [Viva questions and answers](VIVA_QUESTIONS_AND_ANSWERS.md)
