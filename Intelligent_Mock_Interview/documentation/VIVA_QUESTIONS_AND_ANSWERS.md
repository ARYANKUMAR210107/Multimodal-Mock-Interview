# Viva Questions and Answers

Answers reflect this repository's implementation. For questions about limitations, state the limitation openly rather than describing a planned capability as complete.

## A. 30 basic viva questions

1. **What is the project?** — A local mock-interview practice system with a soft-skill classifier, technical-answer scoring, voice feature extraction, and a voice transcription/report workflow.
2. **What problem does it address?** — It gives students a repeatable way to practice technical answers and inspect simple text and voice-delivery indicators.
3. **Who are the intended users?** — Students practicing interviews and instructors demonstrating the project; it is not a production hiring service.
4. **What does “multimodal” mean here?** — The new Voice Interview report combines text-derived scores with measurements from the audio recording. There is no trained multimodal model.
5. **What are the main languages and frameworks?** — Python, Streamlit for the UI, and FastAPI/Uvicorn for the API.
6. **Where does the project start?** — Streamlit starts with `streamlit run frontend/app.py`; FastAPI starts with `uvicorn backend.main:app --reload --port 8000` from the project directory.
7. **Does the project have a database?** — Yes. SQLAlchemy persists students, questions, interview sessions, responses, assessments, and versioned reports in a local SQLite database. Audio bytes are processed temporarily and are not stored in that database.
8. **What is the supervised ML target?** — `CLASS` in the Student Employability spreadsheet: `Employable` or `LessEmployable`.
9. **How many rows are in that dataset?** — The existing EDA report records 2,982 rows and 10 columns.
10. **How many classes and what are their counts?** — Employable 1,729 (58%); LessEmployable 1,253 (42%).
11. **Which inputs does the classifier use?** — Five soft-skill ratings plus their row mean and minimum.
12. **Why is the student name omitted?** — It is an identifier and has no intended predictive meaning.
13. **Which model was selected?** — Gradient Boosting, selected by highest validation macro F1.
14. **What is the reported test accuracy for that model?** — 0.7688, or about 76.88%, on the 597-row held-out test split.
15. **What does accuracy mean?** — The fraction of test predictions that are correct: `(TP+TN)/N`.
16. **What is precision?** — Among examples predicted Employable, the fraction actually Employable.
17. **What is recall?** — Among actually Employable examples, the fraction predicted Employable.
18. **What is F1?** — The harmonic mean of precision and recall.
19. **What does ROC-AUC measure?** — How well score rankings separate positive and negative labels over possible thresholds; it does not measure probability calibration.
20. **What is the question dataset?** — 5,817 question-answer records stored in train/eval Parquet shards as one chat-formatted text field.
21. **Are those questions human graded?** — No. The corpus is used as reference Q&A, not as candidate answer-quality labels.
22. **What is TF-IDF?** — A method that weights a term by its frequency in a document and how uncommon it is across the corpus.
23. **What does cosine similarity do?** — Compares the angle between two TF-IDF vectors; a larger value indicates more shared weighted terms.
24. **What audio library is used?** — Librosa.
25. **What does MFCC stand for?** — Mel-Frequency Cepstral Coefficients, a compact summary of the audio spectrum's shape.
26. **Does pitch prove confidence?** — No. Pitch variation is an acoustic measurement and does not establish confidence, personality, honesty, intelligence, or emotion.
27. **Does the project train an audio model?** — No. It extracts features and applies fixed heuristic formulas in the extractor; the new report displays measurements instead of relying on those estimated scores.
28. **What does Google Gemini do?** — The backend sends a completed audio file to `gemini-3.5-transcribe` for verbatim transcription and returns the transcript for candidate review.
29. **Is microphone speech transcribed live?** — No. Streamlit records first, then a button submits the finished recording for transcription.
30. **Where is the Gemini key stored?** — In repository-root `.env`, loaded by the backend through python-dotenv; `.env.example` contains only a placeholder.

## B. 30 intermediate technical questions

1. **Why use stratified splitting?** — It keeps the approximately 58/42 target proportions similar across train, validation, and test partitions.
2. **How are the split sizes produced?** — Twenty percent is first reserved for test, then 25% of the remaining 80% is validation, resulting in 60/20/20 overall.
3. **Why use a validation set and a test set?** — Validation is used for model selection; the test set is kept for final comparison after that selection.
4. **How is the model selected?** — `best_name` maximizes validation macro F1 in `employability_pipeline.py`.
5. **What does macro F1 add?** — It averages the two per-class F1 scores equally, rather than letting the larger class dominate.
6. **What does `StandardScaler` do?** — It centers each feature and scales by the training standard deviation; it matters for distances and margins.
7. **Which estimators are scaled in the source?** — Logistic Regression, SVM, Gradient Boosting, and KNN. Decision Tree and Random Forest are not scaled.
8. **Why is scaling useful for KNN?** — Euclidean distance otherwise lets larger-unit features dominate the neighbor calculation.
9. **What is the Logistic Regression equation?** — `p(y=1|x)=1/(1+exp(-(wᵀx+b)))`.
10. **What criterion does the Decision Tree use by default?** — scikit-learn's default Gini impurity; the code does not set another criterion.
11. **How does Random Forest differ from one tree?** — It fits many randomized trees and aggregates their predictions, typically reducing variance.
12. **What kernel does the SVM use?** — RBF with `C=1.0` and `gamma="scale"`; probability estimates are enabled.
13. **How does Gradient Boosting learn sequentially?** — Each added tree fits the current ensemble's loss gradient/residual direction; learning rate controls each stage's contribution.
14. **What KNN configuration is used?** — Nine neighbors, Euclidean distance, distance weighting, and standardized features.
15. **What does `class_weight="balanced"` do?** — It adjusts class penalties inversely to class frequencies in Logistic Regression, Decision Tree, Random Forest, and SVM. The Gradient Boosting and KNN definitions do not set it.
16. **Why is Student Performance Rating excluded?** — The report's point-biserial correlation is about −0.0008 and its single-feature tree reaches only majority baseline; the project also treats its meaning as ambiguous.
17. **What is the exact text composite?** — `Q=0.50*reference cosine + 0.20*question cosine + 0.15*Jaccard + 0.15*structure`.
18. **What is Jaccard overlap?** — `|A∩B|/|A∪B|` over cleaned, stopword-filtered token sets.
19. **How is the structure score calculated?** — A fixed length component plus 0.25 per detected connector up to 0.5; very short answers get a special low score. It does not parse grammar.
20. **Does the score measure semantic correctness?** — Not reliably. TF-IDF and Jaccard are mainly lexical overlap; no human answer labels calibrate the weights.
21. **What preprocessing does `clean_text` do?** — Lowercase, remove instruction tags and ASCII punctuation, and normalize whitespace; no stemming or lemmatization.
22. **What n-grams and vectorizer options are used?** — Unigrams and bigrams, sublinear term frequency, English stop words, `max_features=10000`.
23. **Does the TF-IDF fit use only train data?** — No. `run_text_pipeline()` concatenates question and answer text from train and eval before fitting; these are unlabeled references.
24. **How is pause ratio estimated?** — Librosa splits intervals using `top_db=25`; the rest of clip duration is treated as pause/silence.
25. **How is pitch estimated?** — `librosa.yin` searches 65–400 Hz, then the extractor summarizes values between 70 and 380 Hz.
26. **How is WPM estimated in the new report?** — Corrected transcript word count divided by audio duration in minutes; it includes the whole duration, including pauses.
27. **Which filler patterns are counted?** — A fixed case-insensitive regex for `um`, `uh`, `erm`, `ah`, `like`, `you know`, and `i mean`.
28. **How is upload validation performed on the new endpoints?** — Extension, empty payload, 25 MiB limit, container/frame signature, and MIME are checked.
29. **How are temporary audio files cleaned up?** — `_analyze_audio_bytes()` uses `try/finally` and removes its temporary file after Librosa processing.
30. **What makes the new report rule-based?** — It calls the existing text heuristics and calculates basic audio features/derived counts; there is no fitted fusion model or combined numeric score.

## C. 20 advanced ML/NLP/audio questions

1. **Why report both binary and macro F1?** — Binary F1 treats Employable as positive; macro F1 gives both class-specific F1 values equal weight and is the selection metric.
2. **What does the Gradient Boosting test confusion matrix show?** — With rows/columns `[Employable, LessEmployable]`, it is `[[319,27],[111,140]]`; high positive recall comes with 111 false Employable predictions among the less-employable class.
3. **Can the model's predicted probability be read as calibrated placement likelihood?** — No. It is a classifier probability on subjective labels, with no calibration study or real hiring outcome target.
4. **Why may the train/test metrics not generalize?** — The data source, interviewer rating process, and student population may differ from future candidates; features are subjective and limited.
5. **Does five-fold CV guarantee robustness?** — No. It measures variation across folds of this dataset and split procedure; it does not establish external validity.
6. **What does the SVM RBF kernel enable?** — A nonlinear boundary in an implicit feature space, controlled in part by gamma and C.
7. **Why might Gradient Boosting beat linear Logistic Regression here?** — It can model feature interactions/nonlinear thresholds; the recorded metrics support comparative performance, but do not identify a causal reason.
8. **What is the limitation of impurity-based feature importance?** — It measures split-based contribution within that fitted forest and can be biased by correlation/cardinality; it is not causal importance.
9. **Is including the eval Q&A split in TF-IDF fit supervised leakage?** — There are no response labels, so it is not target leakage; it is still transductive use of the nominal eval text for vocabulary and IDF.
10. **Why doesn't cosine similarity understand paraphrases?** — It compares weighted dimensions from the same token vocabulary; synonyms with no token overlap can score low.
11. **Can the structural heuristic distinguish a correct long answer from a fluent but incorrect one?** — No. It rewards length/connectors and has no correctness or grammar model.
12. **Why is fixed-score feedback not calibrated?** — There is no held-out human-labeled candidate-response set to choose/validate thresholds or weights.
13. **What affects F0/YIN reliability?** — Voicing, noise, silence, sample rate, pitch range assumptions, clip duration, and algorithm settings.
14. **What does a 25 dB `effects.split` threshold mean in practice?** — Segments sufficiently below the recording peak are classified as nonsilent boundaries; it is a relative amplitude threshold, not a linguistic pause detector.
15. **Why are MFCC means not a pronunciation score?** — MFCCs summarize spectral envelope characteristics. Without a trained mapping or aligned phonetic reference, they do not score word correctness.
16. **What is the bias in transcript-based WPM?** — ASR/editing omissions or additions affect the numerator; using total duration includes silence and does not estimate syllable rate or articulation rate.
17. **Why might the filler regex overcount?** — “Like” can be a content word; phrase boundaries/language variants and punctuation/ASR formatting affect matching.
18. **What would be required to train an acoustic model?** — Obtain aligned audio, join labels by `wavname`, define speaker-safe splits, extract features consistently, train against Gemini labels, and evaluate against independent human ratings.
19. **Why should human_test remain untouched during training?** — It is the only human-annotated gold set described; tuning on it would bias the evaluation. Current code does not train a model on it.
20. **What would make multimodal fusion scientifically defensible?** — A representative dataset with matched question, recording, transcript, and human rubric labels; a predeclared split/metric and calibration/ablation evaluation.

## D. 20 questions about this code, datasets, and architecture

1. **Which function creates the employability features?** — In the script, `soft_skill_mean` and `soft_skill_min` are added around `src/employability_pipeline.py:182-185`; API prediction reconstructs them in `backend/main.py`.
2. **Which file contains all model definitions?** — `src/employability_pipeline.py`, dictionary `MODELS_DEF` around line 331.
3. **Where are model benchmark metrics stored?** — `models/employability_model_meta.json`, with `val_results` and `test_results`; confusion matrices are in E4/E8 image files and omitted from JSON.
4. **What does the API load on startup?** — Gradient Boosting joblib, TF-IDF joblib, Librosa analyzer, and up to 50 eval Parquet Q&A rows.
5. **What does `/api/v1/questions` return?** — Cached parsed eval Q&A records, limited by a query parameter (default 15).
6. **What does `/api/v1/predict/employability` accept?** — Five JSON integer soft-skill ratings from 1 to 5.
7. **What response does employability inference return?** — Predicted label, both class probabilities, maximum probability as a confidence field, feature mean/minimum, and rule-generated recommendations.
8. **What is the key text class?** — `InterviewTextAssessor` in `src/text_pipeline.py`.
9. **What does `parse_instruction_parquet()` expect?** — A `text` column using `[INST] question [/INST] answer` markers and optional `</s>` ending.
10. **Why is `interview_repository.py` still needed?** — Current UI question selection uses its categorized, hand-authored reference items; it does not pull categories from Parquet.
11. **How many technical categories does the repository describe?** — Eight: Python, C++, Java, DSA, DBMS, OS, Computer Networks, and Machine Learning.
12. **What does the new transcription endpoint return?** — JSON with `transcript`, the configured Gemini `model`, provider, and response ID.
13. **What is the multipart field name for new voice calls?** — `file`; report additionally has form fields `question`, `reference_answer`, and `corrected_transcript`.
14. **Where does the backend get its secret?** — The backend loads the parent repository root `.env`; `_get_gemini_api_key()` reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
15. **How does the frontend call the backend?** — `requests.post()` to `API_BASE_URL` (default `http://127.0.0.1:8000`) with multipart fields/files.
16. **Which OpenEnded split is human annotated?** — `human_test.csv`, 1,000 rows and three annotators, plus averaged score columns.
17. **Why are OpenEnded audio labels not in an acoustic model?** — The WAV archive is absent and there is no feature/label training loop in source.
18. **What is the custom template's state?** — Header-only with zero records; its labels are not training targets yet.
19. **Which older test path is portable?** — The new `src/test_voice_interview_api.py` resolves project location with `Path(__file__)`; the two older scripts hard-code an absolute Windows path.
20. **Where can an examiner see the saved test confusion counts?** — `reports/figures/E8_confusion_matrices_test.png`; exact model matrices are transcribed in `ALGORITHMS_AND_MATHEMATICS.md`.

## E. 15 challenging external-examiner questions

1. **You call it employability prediction. Does the model predict actual employment?** — No. It predicts the dataset's assessor-assigned `CLASS` label from five subjective interview ratings and aggregates. Actual hiring/job outcomes are absent, so “placement probability” is stronger wording than the evidence supports.
2. **Why is 76.88% accuracy not enough to claim the system is reliable?** — The test set is a single random split of one source, ratings are subjective, class prevalence is 58/42, and no external institution/time cohort was evaluated. Macro F1 and errors must also be considered.
3. **Gradient Boosting has 92.2% Employable recall. Is that good?** — It catches many Employable-labeled cases but predicts Employable for 111 of 251 LessEmployable test cases; precision is 74.2%. Whether that tradeoff is good depends on the cost of each error.
4. **Why did you use a separate test set if every model's test result is shown?** — The model is selected by validation macro F1 before test evaluation. Reporting all models on the test set still exposes comparisons; future tuning based on those test results would erode its holdout status.
5. **Your vectorizer uses eval records. Is your evaluation fully isolated?** — No supervised text evaluation exists, but the nominal eval Q&A text contributes to vocabulary/IDF. A stricter future experiment should fit the vectorizer on train only and evaluate on untouched reference data.
6. **How can cosine similarity call an answer correct?** — It cannot establish correctness. It measures lexical direction similarity and is a baseline retrieval heuristic; synonym-heavy correct answers may be under-scored, while copied incorrect phrases may be over-scored.
7. **Why is the project multimodal if no fusion model is trained?** — It is multimodal in input/report design: an audio recording and corrected text both contribute outputs. It is not multimodal machine learning; the report presents separate heuristic text scores and signal measurements.
8. **Why aren't OpenEnded labels used to train the audio scorer?** — The referenced audio files are not available locally, and code contains no feature join/training workflow. Using labels without their recordings would not teach an acoustic predictor.
9. **What exactly does the pitch score mean?** — The report exposes average/std F0 estimates in Hz when available. They describe pitch statistics under the extractor's assumptions; they do not establish vocal quality or personal traits.
10. **Could noise inflate the pause ratio?** — Yes. `effects.split(top_db=25)` is a thresholded signal-level segmentation, so quiet speech/noise and microphone gain affect the intervals. It is not a semantic hesitation detector.
11. **Is the text score reproducible and valid?** — The code/formula is reproducible for the same vectorizer and input, but validity against human judgments is not established; the Q&A dataset has no answer-quality labels.
12. **What are the privacy implications of transcription?** — The browser audio is sent to the backend and then to Google Gemini for transcription. Audio is not persisted in the local database, while transcripts and assessment results are. Consider service-level data handling and consent before recording sensitive speech.
13. **Is this API safe for public internet deployment?** — Not as-is: wildcard CORS, no authentication/rate limiting, uneven protections on legacy upload routes, no broader operational controls, and joblib trust assumptions need addressing.
14. **What is the biggest scientific next step?** — Collect representative interview answers with aligned audio and independent human rubric labels, split by speaker/candidate, and validate text/audio components and fusion on held-out groups.
15. **What would you honestly claim in a demo?** — It is a working academic prototype that trains and compares six classifiers on one employability dataset, scores text against a Q&A reference with transparent heuristics, transcribes submitted recordings using Gemini, reports Librosa measurements, and stores interview progress and reports in SQLite. It does not objectively decide who is employable or provide a trained multimodal quality score.
