# DATASET DOCUMENTATION
## Intelligent Multimodal Mock Interview Assessment System

**Generated:** 2026-10-31  
**Status:** Setup Phase — No model training has begun

---

## Table of Contents

1. [Dataset A — Student Employability](#dataset-a--student-employability)
2. [Dataset B — AI Interview Questions](#dataset-b--ai-interview-questions)
3. [Dataset C — OpenEnded Speech Corpus](#dataset-c--openended-speech-corpus)
4. [Dataset D — Mozilla Common Voice](#dataset-d--mozilla-common-voice)
5. [Dataset E — Custom Interview Dataset Template](#dataset-e--custom-interview-dataset-template)

---

## Dataset A — Student Employability

| Field | Value |
|---|---|
| **Dataset Name** | Student Employability Dataset |
| **Original Source** | Provided by user (Excel file) |
| **HuggingFace / URL** | N/A (user-provided) |
| **File Location** | `dataset/employability/Student-Employability-Datasets.xlsx` |
| **File Format** | Microsoft Excel (.xlsx) |
| **File Size** | ~135 KB |
| **Sheet Name** | `Data` |

### Schema

| Column | Type | Description |
|---|---|---|
| `Name of Student` | string | Anonymized student identifier |
| `GENERAL APPEARANCE` | int (2–5) | Interviewer rating for general appearance |
| `MANNER OF SPEAKING` | int (2–5) | Rating for speaking manner |
| `PHYSICAL CONDITION` | int (2–5) | Rating for physical condition |
| `MENTAL ALERTNESS` | int (2–5) | Rating for alertness and responsiveness |
| `SELF-CONFIDENCE` | int (2–5) | Rating for confidence |
| `ABILITY TO PRESENT IDEAS` | int (2–5) | Rating for idea presentation |
| `COMMUNICATION SKILLS` | int (2–5) | Rating for overall communication |
| `Student Performance Rating` | int (3–5) | Aggregated performance score |
| `CLASS` | string | **Target variable** |

### Target Variable

- **Column:** `CLASS`
- **Type:** Binary classification
- **Values:**
  - `Employable` → 1729 records (58.0%)
  - `LessEmployable` → 1253 records (42.0%)
- **Class imbalance:** Moderate (58/42 split — manageable without oversampling)

### Data Quality

| Metric | Value |
|---|---|
| Total Records | 2,982 |
| Total Columns | 10 |
| Missing Values | **0** (all columns complete) |
| Duplicate Rows | **0** |
| Feature scale | All numeric features rated 2–5 |

### Statistical Summary (Numeric Features)

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| GENERAL APPEARANCE | 4.25 | 0.68 | 2 | 5 |
| MANNER OF SPEAKING | 3.88 | 0.76 | 2 | 5 |
| PHYSICAL CONDITION | 3.97 | 0.74 | 2 | 5 |
| MENTAL ALERTNESS | 3.96 | 0.78 | 2 | 5 |
| SELF-CONFIDENCE | 3.91 | 0.81 | 2 | 5 |
| ABILITY TO PRESENT IDEAS | 3.81 | 0.74 | 2 | 5 |
| COMMUNICATION SKILLS | 3.53 | 0.74 | 2 | 5 |
| Student Performance Rating | 4.61 | 0.69 | 3 | 5 |

### Intended Project Module

- **Module 1:** Employability Prediction (Supervised Classification)
- Algorithms: Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Dataset Limitations

- Ratings are subjective (interviewer-assigned) — potential bias
- `Name of Student` is an identifier with no predictive value (should be dropped before training)
- All numeric features are on the same Likert scale (2–5), so scaling may not be strictly required but is recommended
- No demographic or academic background data present

### License and Usage

- User-provided dataset. Usage is for academic purposes only.
- Original file must not be modified. Use only processed copies for training.

---

## Dataset B — AI Interview Questions

| Field | Value |
|---|---|
| **Dataset Name** | AI Interview Questions |
| **HuggingFace Repository** | [K-areem/AI-Interview-Questions](https://huggingface.co/datasets/K-areem/AI-Interview-Questions) |
| **Files Downloaded** | `train-00000-of-00001.parquet`, `eval-00000-of-00001.parquet` |
| **File Location** | `dataset/interview_questions/` |
| **File Format** | Parquet |
| **File Sizes** | Train: ~769 KB, Eval: ~195 KB |

### Schema

| Column | Type | Description |
|---|---|---|
| `text` | string | Combined instruction–response pair in chat template format |

### Data Format

Each record follows the Llama/Mistral instruction format:

```
<s>[INST] {question} [/INST] {answer}</s>
```

- Train split: 4,653 records
- Eval split: 1,164 records
- **Total:** 5,817 records

### Data Quality

| Metric | Train | Eval |
|---|---|---|
| Records | 4,653 | 1,164 |
| Missing Values | 0 | 0 |
| Duplicates | 0 | 0 |

### Intended Project Module

- **Module 2:** Interview Q&A Reference System
- Use for: question classification, answer quality benchmark, NLP fine-tuning reference
- The Q&A pairs serve as reference answers to compare against candidate responses

> [!NOTE]
> This dataset was originally formatted for LLM fine-tuning. The `[INST]` / `[/INST]` tags must be parsed out during preprocessing to extract raw question and answer text.

### Dataset Limitations

- Only one column (`text`) — question and answer must be parsed with regex
- No topic/category labels — topic clustering will be needed
- No difficulty labels
- Answers are generated by an LLM, not human-verified

### License and Usage

- HuggingFace: [K-areem/AI-Interview-Questions](https://huggingface.co/datasets/K-areem/AI-Interview-Questions)
- Check repository license before publication

---

## Dataset C — OpenEnded Speech Corpus

| Field | Value |
|---|---|
| **Dataset Name** | OpenEnded Speech Scoring Corpus |
| **HuggingFace Repository** | [yuwchen/OpenEnded](https://huggingface.co/datasets/yuwchen/OpenEnded) |
| **Files Downloaded** | 4 annotation CSVs (see below) |
| **File Location** | `dataset/openended/` |
| **Audio Archive** | `wav.zip` — **NOT downloaded** (see note below) |

### Downloaded Files

| File | Records | Columns | Description |
|---|---|---|---|
| `gemini2_train.csv` | 6,109 | 4 | Gemini-2 AI labels for train split |
| `gemini2_dev.csv` | 2,673 | 4 | Gemini-2 AI labels for dev split |
| `human_test.csv` | 1,000 | 13 | 3-annotator human labels for test split |
| `meta_data.csv` | 9,782 | 4 | Recording metadata (wavname, user, question, split) |

### Schema

#### gemini2_train.csv / gemini2_dev.csv

| Column | Type | Description |
|---|---|---|
| `wavname` | string | Audio filename reference |
| `Accuracy` | int (1–5) | Pronunciation accuracy score (AI-labeled) |
| `Fluency` | int (1–5) | Speech fluency score (AI-labeled) |
| `Prosody` | int (1–5) | Prosody/intonation score (AI-labeled) |

#### human_test.csv

| Column | Type | Description |
|---|---|---|
| `wavname` | string | Audio filename reference |
| `A1_Accuracy`, `A1_Fluency`, `A1_Prosody` | int | Annotator 1 scores |
| `A2_Accuracy`, `A2_Fluency`, `A2_Prosody` | int | Annotator 2 scores |
| `A3_Accuracy`, `A3_Fluency`, `A3_Prosody` | int | Annotator 3 scores |
| `Accuracy`, `Fluency`, `Prosody` | float | **Averaged human scores (target)** |

#### meta_data.csv

| Column | Type | Description |
|---|---|---|
| `wavname` | string | Audio filename (links to annotation files) |
| `user_id` | string | Anonymized speaker ID |
| `question` | string | Interview question asked (1 missing value) |
| `split` | string | train / dev / test partition |

### Data Quality

| Metric | Value |
|---|---|
| Total unique recordings | 9,782 |
| Missing values in meta_data | 1 (in `question` column) |
| Duplicates | 0 across all files |

### Audio Archive Decision

> [!CAUTION]
> **wav.zip is ~6.71 GB — NOT downloaded.**
>
> **Reason:** The annotation CSV files are sufficient for the setup phase and initial EDA. The audio files are only required when extracting acoustic features (MFCCs, pitch, energy) for speech scoring models.
>
> **Storage requirement:** You need at least **8–10 GB free disk space** before downloading wav.zip.
>
> **Decision needed from user:** Confirm available storage and need before downloading.

### Intended Project Module

- **Module 3:** Speech Quality Scoring (Pronunciation Accuracy, Fluency, Prosody)
- Labels from gemini2_train/dev serve as training signal
- human_test serves as gold standard evaluation
- Requires audio feature extraction (MFCC, spectral features via librosa) — **only when wav.zip is downloaded**

### Dataset Limitations

- Audio files not downloaded — speech feature pipeline is blocked until wav.zip is available
- AI-generated labels (Gemini-2) may not match human judgment perfectly
- Only 1,000 recordings have 3-way human annotation (test set only)
- One missing question value in meta_data.csv

### License and Usage

- HuggingFace: [yuwchen/OpenEnded](https://huggingface.co/datasets/yuwchen/OpenEnded)
- Check repository license before publication

---

## Dataset D — Mozilla Common Voice

| Field | Value |
|---|---|
| **Dataset Name** | Mozilla Common Voice |
| **Source** | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets) |
| **Files Downloaded** | **None — folder reserved** |
| **File Location** | `dataset/common_voice/` (empty placeholder) |
| **Status** | ⚠️ NOT DOWNLOADED |

### Purpose

Mozilla Common Voice provides crowd-sourced speech recordings with validated transcriptions. In this project, it would be used for:

- Training / fine-tuning an ASR (Automatic Speech Recognition) model
- Transcription of candidate audio responses for text-based NLP analysis

### Recommended Subset

> [!IMPORTANT]
> Do **NOT** download the full dataset. The full Common Voice corpus is hundreds of GB.
>
> **Recommended subset:** English (`en`) validated clips only.
>
> - Common Voice 17.0 English validated set: ~2,400 hours (~5–7 GB compressed)
> - For a college project: download via `datasets` library from HuggingFace:
>   ```python
>   from datasets import load_dataset
>   ds = load_dataset("mozilla-foundation/common_voice_17_0", "en",
>                     split="train", streaming=True)
>   ```
>   Use `streaming=True` to avoid downloading the full archive.

### Decision Required

This dataset requires user decision on:
1. Whether a custom ASR model is needed (or a pre-trained Whisper model will be used instead)
2. Available local storage for audio files

### License

- Mozilla Public License 2.0
- Community contributions are CC0 or CC-BY licensed
- Academic use permitted

---

## Dataset E — Custom Interview Dataset Template

| Field | Value |
|---|---|
| **Dataset Name** | Custom Interview Answer Assessment Template |
| **Source** | Created for this project |
| **File Location** | `dataset/custom/interview_answer_assessment_template.csv` |
| **File Format** | CSV |
| **Status** | Template only — no records yet |

### Schema

| Column | Type | Description |
|---|---|---|
| `candidate_id` | string | Unique candidate identifier |
| `question_id` | string | Question identifier |
| `question` | string | Interview question text |
| `candidate_answer` | string | Candidate's spoken/typed response |
| `reference_answer` | string | Expert reference answer |
| `relevance_score` | float (0–1) | How relevant the answer is to the question |
| `structure_score` | float (0–1) | Answer structure and organization |
| `quality_score` | float (0–1) | Overall answer quality |
| `human_evaluator_label` | string/int | Human-assigned evaluation label |

> [!WARNING]
> **Labels must be left empty until genuine human evaluation is collected.**
> Do NOT generate synthetic labels or fill columns with dummy scores.
> This dataset will be used for supervised training only after actual interviews are conducted.

### Intended Project Module

- **Module 4:** Answer Quality Assessment
- Used as ground truth for training a text-based answer scoring model
- Will be merged with parsed questions from Dataset B

### License and Usage

- Original dataset created for this academic project
- Not suitable for distribution until human evaluation is complete

