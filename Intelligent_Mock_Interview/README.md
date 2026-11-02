# Intelligent Multimodal Mock Interview Assessment System

## Project Introduction

This is a college-level Supervised Machine Learning project that builds an intelligent, multimodal mock interview assessment system. The system evaluates candidates across multiple dimensions:

- **Text dimension**: Answer relevance, structure, and quality scoring using NLP models
- **Speech dimension**: Pronunciation accuracy, fluency, and prosody scoring using audio ML
- **Employability prediction**: Predicting student employability from academic and soft-skill features

The project is organized as a modular pipeline covering data preparation → EDA → feature engineering → model training → evaluation → deployment (FastAPI backend + Streamlit frontend).

---

## Folder Structure

```
intelligent_supervised/
│
├── venv/                         ← Python 3.13 virtual environment (DO NOT DELETE)
│
└── Intelligent_Mock_Interview/   ← Main project root
    │
    ├── dataset/
    │   ├── employability/        ← Student-Employability-Datasets.xlsx
    │   ├── interview_questions/  ← train & eval Parquet files (AI Interview Qs)
    │   ├── openended/            ← OpenEnded Speech Corpus annotation CSVs
    │   ├── common_voice/         ← Placeholder (Mozilla Common Voice – not downloaded yet)
    │   └── custom/               ← interview_answer_assessment_template.csv
    │
    ├── notebooks/                ← Jupyter notebooks for EDA and experiments
    ├── models/                   ← Saved trained models (.pkl, .joblib, .h5)
    ├── src/                      ← Reusable Python source modules
    ├── backend/                  ← FastAPI backend application
    ├── frontend/                 ← Streamlit frontend application
    ├── reports/                  ← Dataset inventory, evaluation reports
    ├── documentation/            ← Dataset documentation, design docs
    │
    ├── requirements.txt          ← Python dependency list
    ├── README.md                 ← This file
    └── .gitignore                ← Git ignore rules
```

---

## Python Environment Setup

The virtual environment is located at:

```
c:\Users\Asus\OneDrive\Desktop\intelligent_supervised\venv\
```

**Python version:** 3.13.14  
**pip version:** 26.2.2

### Activate the virtual environment (PowerShell):

```powershell
cd c:\Users\Asus\OneDrive\Desktop\intelligent_supervised
.\venv\Scripts\Activate.ps1
```

> If you see an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Library Installation Commands

After activating the virtual environment:

```powershell
cd Intelligent_Mock_Interview
pip install -r requirements.txt
```

To install a specific missing library:

```powershell
pip install openpyxl
pip install librosa soundfile
pip install fastapi uvicorn
pip install streamlit
pip install nltk
```

---

## Dataset Locations

| Dataset | Location | Format |
|---|---|---|
| Student Employability | `dataset/employability/Student-Employability-Datasets.xlsx` | Excel (.xlsx) |
| AI Interview Questions (Train) | `dataset/interview_questions/train-00000-of-00001.parquet` | Parquet |
| AI Interview Questions (Eval) | `dataset/interview_questions/eval-00000-of-00001.parquet` | Parquet |
| OpenEnded – Gemini Train Labels | `dataset/openended/gemini2_train.csv` | CSV |
| OpenEnded – Gemini Dev Labels | `dataset/openended/gemini2_dev.csv` | CSV |
| OpenEnded – Human Test Labels | `dataset/openended/human_test.csv` | CSV |
| OpenEnded – Metadata | `dataset/openended/meta_data.csv` | CSV |
| Custom Template | `dataset/custom/interview_answer_assessment_template.csv` | CSV |
| Common Voice (not downloaded) | `dataset/common_voice/` | TBD |

---

## How to Run Jupyter Notebook

```powershell
# Activate venv first
cd c:\Users\Asus\OneDrive\Desktop\intelligent_supervised
.\venv\Scripts\Activate.ps1

# Change to project folder
cd Intelligent_Mock_Interview

# Launch JupyterLab
jupyter lab

# Or classic notebook
jupyter notebook
```

Then open any `.ipynb` file from the `notebooks/` folder in your browser.

---

## How to Inspect Datasets

### Excel (Employability Dataset):

```python
import pandas as pd

df = pd.read_excel(
    "dataset/employability/Student-Employability-Datasets.xlsx",
    sheet_name="Data"
)
print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum())
```

### Parquet (Interview Questions):

```python
import pandas as pd

df_train = pd.read_parquet("dataset/interview_questions/train-00000-of-00001.parquet")
df_eval  = pd.read_parquet("dataset/interview_questions/eval-00000-of-00001.parquet")
print(df_train.shape, df_eval.shape)
print(df_train.head())
```

### CSV (OpenEnded Corpus):

```python
import pandas as pd

meta    = pd.read_csv("dataset/openended/meta_data.csv")
train   = pd.read_csv("dataset/openended/gemini2_train.csv")
dev     = pd.read_csv("dataset/openended/gemini2_dev.csv")
test    = pd.read_csv("dataset/openended/human_test.csv")
print(meta.shape, train.shape, dev.shape, test.shape)
```

---

## How to Run the Applications

### 1. Launch FastAPI Backend Service
```powershell
# Activate environment
cd c:\Users\Asus\OneDrive\Desktop\intelligent_supervised
.\venv\Scripts\Activate.ps1

cd Intelligent_Mock_Interview
uvicorn backend.main:app --reload --port 8000
```
Interactive Swagger API docs will be available at: `http://127.0.0.1:8000/docs`

### 2. Launch Streamlit Web UI
```powershell
# In a new terminal (with venv activated)
cd c:\Users\Asus\OneDrive\Desktop\intelligent_supervised
.\venv\Scripts\Activate.ps1

cd Intelligent_Mock_Interview
streamlit run frontend/app.py
```
The interactive application will open automatically in your browser at: `http://localhost:8501`

---

## Notes

- **Best Model:** Gradient Boosting Classifier (`accuracy=76.88%`, `ROC-AUC=0.8248`) saved at `models/gradient_boosting_employability.joblib`.
- **Text Engine:** Sublinear TF-IDF Vectorizer with Cosine & Jaccard similarity scoring saved at `models/interview_tfidf_vectorizer.joblib`.
- **Speech Engine:** Librosa acoustic extractor analyzing Pitch, RMS Energy, and Pause Ratios directly from candidate responses.
- The `wav.zip` archive (~6.71 GB) for the OpenEnded corpus was intentionally not downloaded to preserve disk space; candidate audio is evaluated via live runtime extraction.

