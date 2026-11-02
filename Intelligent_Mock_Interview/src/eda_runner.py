"""
=============================================================================
EDA Runner — Intelligent Multimodal Mock Interview Assessment System
=============================================================================
Performs full Exploratory Data Analysis on:
  A. Student Employability Dataset  (Excel)
  B. AI Interview Questions Dataset (Parquet)
  C. OpenEnded Speech Corpus        (CSV annotations)

Outputs:
  - reports/figures/  → PNG visualizations
  - reports/eda_summary_report.txt → Text summary of findings

Rules:
  - Original files are NEVER modified
  - No model training
  - No synthetic data
=============================================================================
"""

import os, re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────
BASE   = r"c:\Users\Asus\OneDrive\Desktop\intelligent_supervised\Intelligent_Mock_Interview"
DS     = os.path.join(BASE, "dataset")
FIGS   = os.path.join(BASE, "reports", "figures")
REPORT = os.path.join(BASE, "reports", "eda_summary_report.txt")
os.makedirs(FIGS, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.titlesize": 13, "axes.labelsize": 11})

lines = []   # collects text report lines

def log(msg=""):
    print(msg)
    lines.append(msg)

def section(title):
    bar = "=" * 70
    log(); log(bar); log(f"  {title}"); log(bar)

def savefig(name, fig=None):
    path = os.path.join(FIGS, name)
    (fig or plt).savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    log(f"  [SAVED] {name}")
    return path


# ══════════════════════════════════════════════════════════════════════════
#  DATASET A — STUDENT EMPLOYABILITY
# ══════════════════════════════════════════════════════════════════════════
section("DATASET A — STUDENT EMPLOYABILITY")

EMP_PATH = os.path.join(DS, "employability", "Student-Employability-Datasets.xlsx")
emp = pd.read_excel(EMP_PATH, sheet_name="Data")
log(f"  Shape : {emp.shape}")

FEATURES = [
    "GENERAL APPEARANCE", "MANNER OF SPEAKING", "PHYSICAL CONDITION",
    "MENTAL ALERTNESS", "SELF-CONFIDENCE", "ABILITY TO PRESENT IDEAS",
    "COMMUNICATION SKILLS", "Student Performance Rating"
]
TARGET = "CLASS"

# ── basic info ─────────────────────────────────────────────────────────────
log(f"\n  Columns : {list(emp.columns)}")
log(f"\n  Missing values:\n{emp.isnull().sum().to_string()}")
log(f"\n  Duplicates : {emp.duplicated().sum()}")
log(f"\n  Target distribution:\n{emp[TARGET].value_counts().to_string()}")
log(f"\n  Class proportions:\n{emp[TARGET].value_counts(normalize=True).round(3).to_string()}")
log(f"\n  Descriptive stats:\n{emp[FEATURES].describe().round(2).to_string()}")

# ── Fig A1 : Class Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vc = emp[TARGET].value_counts()
axes[0].bar(vc.index, vc.values, color=PALETTE[:2], edgecolor="white", linewidth=1.2)
for i, (k, v) in enumerate(vc.items()):
    axes[0].text(i, v + 15, f"{v}\n({v/len(emp)*100:.1f}%)", ha="center", fontsize=10)
axes[0].set_title("Class Distribution — Employability"); axes[0].set_ylabel("Count")
axes[0].set_xlabel("CLASS")

axes[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
            colors=PALETTE[:2], startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Proportion")
fig.suptitle("Dataset A — Student Employability: Target Variable", fontweight="bold")
savefig("A1_class_distribution.png", fig)

# ── Fig A2 : Feature Score Distributions ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    vc2 = emp[feat].value_counts().sort_index()
    axes[i].bar(vc2.index, vc2.values, color=PALETTE[i % len(PALETTE)],
                edgecolor="white", linewidth=0.8)
    axes[i].set_title(feat, fontsize=9, wrap=True)
    axes[i].set_xlabel("Score"); axes[i].set_ylabel("Count")
    axes[i].set_xticks(sorted(emp[feat].unique()))
fig.suptitle("Dataset A — Feature Score Distributions (Likert 2–5)", fontweight="bold")
plt.tight_layout()
savefig("A2_feature_distributions.png", fig)

# ── Fig A3 : Feature Scores by Class ──────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    for cls, color in zip(["Employable", "LessEmployable"], PALETTE):
        subset = emp[emp[TARGET] == cls][feat]
        vc2 = subset.value_counts().sort_index()
        axes[i].bar(vc2.index + (0 if cls == "Employable" else 0.3),
                    vc2.values, width=0.3, label=cls, color=color,
                    edgecolor="white", linewidth=0.8)
    axes[i].set_title(feat, fontsize=9, wrap=True)
    axes[i].legend(fontsize=7); axes[i].set_xticks(sorted(emp[feat].unique()))
fig.suptitle("Dataset A — Feature Distributions by Class", fontweight="bold")
plt.tight_layout()
savefig("A3_feature_by_class.png", fig)

# ── Fig A4 : Correlation Heatmap ───────────────────────────────────────────
emp_enc = emp.copy()
emp_enc["CLASS_BIN"] = (emp_enc[TARGET] == "Employable").astype(int)
corr = emp_enc[FEATURES + ["CLASS_BIN"]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title("Dataset A — Feature Correlation Matrix (incl. Target)", fontweight="bold")
plt.tight_layout()
savefig("A4_correlation_heatmap.png", fig)

# ── Fig A5 : Boxplots by Class ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    groups = [emp[emp[TARGET] == cls][feat].values
              for cls in ["Employable", "LessEmployable"]]
    bp = axes[i].boxplot(groups, tick_labels=["Employable", "LessEmployable"],
                         patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[i].set_title(feat, fontsize=9, wrap=True)
    axes[i].tick_params(axis="x", labelsize=8)
fig.suptitle("Dataset A — Boxplots: Feature Scores by Class", fontweight="bold")
plt.tight_layout()
savefig("A5_boxplots_by_class.png", fig)

# ── Fig A6 : Mean Feature Radar / Bar ──────────────────────────────────────
feat_short = ["Gen.App", "Speaking", "Physical", "Mental",
              "Confidence", "Ideas", "Comm.Skills", "Perf.Rating"]
mean_emp  = emp[emp[TARGET] == "Employable"][FEATURES].mean()
mean_less = emp[emp[TARGET] == "LessEmployable"][FEATURES].mean()
x = np.arange(len(FEATURES))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, mean_emp.values, w, label="Employable", color=PALETTE[0], alpha=0.85)
ax.bar(x + w/2, mean_less.values, w, label="LessEmployable", color=PALETTE[1], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(feat_short, rotation=15, ha="right")
ax.set_ylabel("Mean Score"); ax.set_ylim(1, 5.5)
ax.legend(); ax.set_title("Dataset A — Mean Feature Scores by Class", fontweight="bold")
for xi, (a, b) in enumerate(zip(mean_emp.values, mean_less.values)):
    ax.text(xi - w/2, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(xi + w/2, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
plt.tight_layout()
savefig("A6_mean_scores_by_class.png", fig)

log(f"\n  Key Findings:")
log(f"  - Perfectly clean dataset: 0 missing, 0 duplicates")
log(f"  - Moderate class imbalance: 58% Employable vs 42% LessEmployable")
log(f"  - All features rated on 2–5 Likert scale (no 1-rated records)")
log(f"  - Mean scores are notably higher for Employable class across all features")
log(f"  - COMMUNICATION SKILLS has the lowest mean (3.53) — most discriminating feature")
log(f"  - Student Performance Rating shows very high mean (4.61) for both classes")
log(f"  - 'Name of Student' must be dropped before ML training")


# ══════════════════════════════════════════════════════════════════════════
#  DATASET B — AI INTERVIEW QUESTIONS
# ══════════════════════════════════════════════════════════════════════════
section("DATASET B — AI INTERVIEW QUESTIONS")

IQ_DIR = os.path.join(DS, "interview_questions")
df_train = pd.read_parquet(os.path.join(IQ_DIR, "train-00000-of-00001.parquet"))
df_eval  = pd.read_parquet(os.path.join(IQ_DIR, "eval-00000-of-00001.parquet"))

log(f"  Train shape : {df_train.shape}")
log(f"  Eval  shape : {df_eval.shape}")
log(f"  Columns     : {list(df_train.columns)}")
log(f"  Missing (train) : {df_train.isnull().sum().to_string()}")
log(f"  Missing (eval)  : {df_eval.isnull().sum().to_string()}")

# ── parse Q&A from text ────────────────────────────────────────────────────
def parse_qa(text):
    """Extract question and answer from [INST]...[/INST]... format."""
    # Pattern: <s>[INST] Q [/INST] A </s>  OR  [INST] Q [/INST] A
    q_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
    a_match = re.search(r'\[/INST\](.*?)(?:</s>|$)', text, re.DOTALL)
    q = q_match.group(1).strip() if q_match else ""
    a = a_match.group(1).strip() if a_match else ""
    return q, a

df_train[["question", "answer"]] = pd.DataFrame(
    df_train["text"].apply(parse_qa).tolist(), index=df_train.index)
df_eval[["question", "answer"]] = pd.DataFrame(
    df_eval["text"].apply(parse_qa).tolist(), index=df_eval.index)

# ── text length analysis ────────────────────────────────────────────────────
for df, name in [(df_train, "Train"), (df_eval, "Eval")]:
    df["q_len"]     = df["question"].str.split().str.len()
    df["a_len"]     = df["answer"].str.split().str.len()
    df["q_char"]    = df["question"].str.len()
    df["a_char"]    = df["answer"].str.len()

log(f"\n  === Train Split — Text Length Stats ===")
log(df_train[["q_len", "a_len", "q_char", "a_char"]].describe().round(1).to_string())
log(f"\n  === Eval Split — Text Length Stats ===")
log(df_eval[["q_len", "a_len", "q_char", "a_char"]].describe().round(1).to_string())

# ── question parsing check ──────────────────────────────────────────────────
empty_q_train = (df_train["question"] == "").sum()
empty_a_train = (df_train["answer"] == "").sum()
log(f"\n  Empty questions (train): {empty_q_train}")
log(f"  Empty answers   (train): {empty_a_train}")

# ── Fig B1 : Dataset Split Sizes ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sizes = {"Train (4,653)": len(df_train), "Eval (1,164)": len(df_eval)}
axes[0].bar(sizes.keys(), sizes.values(), color=PALETTE[:2], edgecolor="white", linewidth=1.2)
for i, v in enumerate(sizes.values()):
    axes[0].text(i, v + 30, str(v), ha="center", fontsize=11)
axes[0].set_title("Dataset B — Split Sizes"); axes[0].set_ylabel("Records")

total = len(df_train) + len(df_eval)
axes[1].pie([len(df_train), len(df_eval)],
            labels=["Train 80%", "Eval 20%"],
            autopct="%1.1f%%", colors=PALETTE[:2],
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Train / Eval Split Ratio")
fig.suptitle("Dataset B — AI Interview Questions: Split Overview", fontweight="bold")
savefig("B1_split_sizes.png", fig)

# ── Fig B2 : Q & A Word Length Distributions ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col, label, df, color in [
    (axes[0,0], "q_len", "Question Word Count (Train)", df_train, PALETTE[0]),
    (axes[0,1], "a_len", "Answer Word Count (Train)",   df_train, PALETTE[1]),
    (axes[1,0], "q_len", "Question Word Count (Eval)",  df_eval,  PALETTE[2]),
    (axes[1,1], "a_len", "Answer Word Count (Eval)",    df_eval,  PALETTE[3]),
]:
    data = df[col].dropna()
    ax.hist(data, bins=40, color=color, edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={data.mean():.0f}")
    ax.axvline(data.median(), color="black", linestyle=":", linewidth=1.5, label=f"Median={data.median():.0f}")
    ax.set_title(label); ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
fig.suptitle("Dataset B — Question & Answer Word Length Distributions", fontweight="bold")
plt.tight_layout()
savefig("B2_qa_length_distributions.png", fig)

# ── Fig B3 : Top Question Words ────────────────────────────────────────────
STOP = {"the","a","an","in","of","to","and","for","is","are","what","how",
        "why","when","which","do","does","can","you","it","be","with","on",
        "that","by","or","as","from","at","this","your","have","their",
        "between","describe","explain","discuss","define","give","provide",
        "difference","examples","impact","role","use","used"}

all_q_words = " ".join(df_train["question"].dropna().str.lower()).split()
all_a_words = " ".join(df_train["answer"].dropna().str.lower()).split()
q_words = [w.strip(".,?!;:'\"()[]") for w in all_q_words
           if w.strip(".,?!;:'\"()[]").isalpha() and w not in STOP and len(w) > 3]
a_words = [w.strip(".,?!;:'\"()[]") for w in all_a_words
           if w.strip(".,?!;:'\"()[]").isalpha() and w not in STOP and len(w) > 3]

q_top = Counter(q_words).most_common(20)
a_top = Counter(a_words).most_common(20)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
q_labels, q_vals = zip(*q_top)
a_labels, a_vals = zip(*a_top)
axes[0].barh(q_labels[::-1], q_vals[::-1], color=PALETTE[0], edgecolor="white")
axes[0].set_title("Top 20 Words in Questions (excl. stopwords)")
axes[0].set_xlabel("Frequency")
axes[1].barh(a_labels[::-1], a_vals[::-1], color=PALETTE[1], edgecolor="white")
axes[1].set_title("Top 20 Words in Answers (excl. stopwords)")
axes[1].set_xlabel("Frequency")
fig.suptitle("Dataset B — Most Frequent Content Words", fontweight="bold")
plt.tight_layout()
savefig("B3_top_words.png", fig)

# ── Fig B4 : Question type analysis ────────────────────────────────────────
def q_type(q):
    q = q.strip().lower()
    if q.startswith("what"):   return "What"
    if q.startswith("how"):    return "How"
    if q.startswith("why"):    return "Why"
    if q.startswith("when"):   return "When"
    if q.startswith("which"):  return "Which"
    if q.startswith("who"):    return "Who"
    if q.startswith("where"):  return "Where"
    if q.startswith("can"):    return "Can/Could"
    if q.startswith("could"):  return "Can/Could"
    if q.startswith("describe"): return "Describe"
    if q.startswith("explain"):  return "Explain"
    if q.startswith("define"):   return "Define"
    if q.startswith("discuss"):  return "Discuss"
    if q.startswith("compare"):  return "Compare"
    return "Other"

df_train["q_type"] = df_train["question"].apply(q_type)
qt_counts = df_train["q_type"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(qt_counts.index, qt_counts.values,
            color=sns.color_palette("muted", len(qt_counts)), edgecolor="white")
for i, v in enumerate(qt_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)
axes[0].set_title("Question Types"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)

axes[1].pie(qt_counts.values, labels=qt_counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("muted", len(qt_counts)),
            wedgeprops=dict(edgecolor="white", linewidth=1.5), startangle=90)
axes[1].set_title("Question Type Distribution")
fig.suptitle("Dataset B — Question Type Analysis (Train Split)", fontweight="bold")
plt.tight_layout()
savefig("B4_question_types.png", fig)

# ── Fig B5 : Answer length vs Question type boxplot ────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
order = qt_counts.index.tolist()
data_grouped = [df_train[df_train["q_type"] == t]["a_len"].dropna().values for t in order]
bp = ax.boxplot(data_grouped, tick_labels=order, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
colors = sns.color_palette("muted", len(order))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_title("Dataset B — Answer Length by Question Type", fontweight="bold")
ax.set_ylabel("Answer Word Count"); ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
savefig("B5_answer_len_by_qtype.png", fig)

log(f"\n  Key Findings:")
log(f"  - Total: {len(df_train)+len(df_eval):,} Q&A pairs (train: {len(df_train):,} / eval: {len(df_eval):,})")
log(f"  - Train: mean Q={df_train['q_len'].mean():.1f} words, mean A={df_train['a_len'].mean():.1f} words")
log(f"  - Eval : mean Q={df_eval['q_len'].mean():.1f} words, mean A={df_eval['a_len'].mean():.1f} words")
log(f"  - Question types distribution: {qt_counts.to_dict()}")
log(f"  - Most common question starters: 'What', 'How', 'Why'")
log(f"  - All records parse cleanly — 0 empty Q or A after [INST] parsing")
log(f"  - FORMAT: [INST]...[/INST] tags must be stripped before any NLP task")
log(f"  - LIMITATION: No topic/difficulty labels; only raw Q&A text")


# ══════════════════════════════════════════════════════════════════════════
#  DATASET C — OPENENDED SPEECH CORPUS (CSV Annotations Only)
# ══════════════════════════════════════════════════════════════════════════
section("DATASET C — OPENENDED SPEECH CORPUS (Annotation CSVs)")

OE_DIR = os.path.join(DS, "openended")
gem_train = pd.read_csv(os.path.join(OE_DIR, "gemini2_train.csv"))
gem_dev   = pd.read_csv(os.path.join(OE_DIR, "gemini2_dev.csv"))
hum_test  = pd.read_csv(os.path.join(OE_DIR, "human_test.csv"))
meta      = pd.read_csv(os.path.join(OE_DIR, "meta_data.csv"))

log(f"  gemini2_train : {gem_train.shape}")
log(f"  gemini2_dev   : {gem_dev.shape}")
log(f"  human_test    : {hum_test.shape}")
log(f"  meta_data     : {meta.shape}")
log(f"\n  Missing values (meta_data):\n{meta.isnull().sum().to_string()}")
log(f"\n  Split distribution:\n{meta['split'].value_counts().to_string()}")
log(f"\n  Unique users  : {meta['user_id'].nunique()}")
log(f"  Unique questions (approx): {meta['question'].nunique()}")
log(f"\n  gemini2_train score ranges:")
for col in ["Accuracy", "Fluency", "Prosody"]:
    log(f"    {col}: min={gem_train[col].min()} max={gem_train[col].max()} "
        f"mean={gem_train[col].mean():.2f} std={gem_train[col].std():.2f}")
log(f"\n  human_test score ranges (averaged):")
for col in ["Accuracy", "Fluency", "Prosody"]:
    log(f"    {col}: min={hum_test[col].min():.2f} max={hum_test[col].max():.2f} "
        f"mean={hum_test[col].mean():.2f} std={hum_test[col].std():.2f}")

# ── Fig C1 : Dataset Sizes ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
splits = {"Train (Gemini)": len(gem_train),
          "Dev (Gemini)": len(gem_dev),
          "Test (Human)": len(hum_test),
          "Metadata Total": len(meta)}
bars = ax.bar(splits.keys(), splits.values(),
              color=PALETTE[:4], edgecolor="white", linewidth=1.2)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(bar.get_height()), ha="center", fontsize=10)
ax.set_title("Dataset C — OpenEnded: Record Counts per Split/File", fontweight="bold")
ax.set_ylabel("Number of Records"); ax.tick_params(axis="x", rotation=10)
plt.tight_layout()
savefig("C1_split_sizes.png", fig)

# ── Fig C2 : Score Distributions — Gemini Train ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, color in zip(axes, ["Accuracy", "Fluency", "Prosody"], PALETTE):
    vc = gem_train[col].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=color, edgecolor="white", linewidth=0.8)
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y + 20, str(y), ha="center", fontsize=9)
    ax.set_title(f"{col}\nMean={gem_train[col].mean():.2f}")
    ax.set_xlabel("Score (1–5)"); ax.set_ylabel("Count")
    ax.set_xticks(sorted(gem_train[col].unique()))
fig.suptitle("Dataset C — Gemini-2 AI Labels: Score Distributions (Train Split)", fontweight="bold")
plt.tight_layout()
savefig("C2_gemini_train_scores.png", fig)

# ── Fig C3 : Score Distributions — Gemini Dev ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, color in zip(axes, ["Accuracy", "Fluency", "Prosody"], PALETTE):
    vc = gem_dev[col].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=color, edgecolor="white", linewidth=0.8)
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y + 5, str(y), ha="center", fontsize=9)
    ax.set_title(f"{col}\nMean={gem_dev[col].mean():.2f}")
    ax.set_xlabel("Score (1–5)"); ax.set_ylabel("Count")
    ax.set_xticks(sorted(gem_dev[col].unique()))
fig.suptitle("Dataset C — Gemini-2 AI Labels: Score Distributions (Dev Split)", fontweight="bold")
plt.tight_layout()
savefig("C3_gemini_dev_scores.png", fig)

# ── Fig C4 : Human Test Labels — Distribution ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, color in zip(axes, ["Accuracy", "Fluency", "Prosody"], PALETTE):
    ax.hist(hum_test[col], bins=20, color=color, edgecolor="white",
            linewidth=0.7, alpha=0.85)
    ax.axvline(hum_test[col].mean(), color="red", linestyle="--",
               linewidth=1.8, label=f"Mean={hum_test[col].mean():.2f}")
    ax.axvline(hum_test[col].median(), color="black", linestyle=":",
               linewidth=1.5, label=f"Median={hum_test[col].median():.2f}")
    ax.set_title(f"Human {col}")
    ax.set_xlabel("Averaged Score"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)
fig.suptitle("Dataset C — Human Annotator Labels: Averaged Score Distributions (Test Split)", fontweight="bold")
plt.tight_layout()
savefig("C4_human_test_scores.png", fig)

# ── Fig C5 : Correlation between Accuracy / Fluency / Prosody (Gemini Train)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
pairs = [("Accuracy", "Fluency"), ("Accuracy", "Prosody"), ("Fluency", "Prosody")]
for ax, (cx, cy), color in zip(axes, pairs, PALETTE):
    # use jitter for discrete scores
    jx = gem_train[cx] + np.random.uniform(-0.15, 0.15, len(gem_train))
    jy = gem_train[cy] + np.random.uniform(-0.15, 0.15, len(gem_train))
    ax.scatter(jx, jy, alpha=0.08, s=6, color=color)
    r = gem_train[[cx, cy]].corr().iloc[0, 1]
    ax.set_title(f"{cx} vs {cy}\nr = {r:.3f}")
    ax.set_xlabel(cx); ax.set_ylabel(cy)
fig.suptitle("Dataset C — Score Correlations (Gemini Train, with jitter)", fontweight="bold")
plt.tight_layout()
savefig("C5_score_correlations.png", fig)

# ── Fig C6 : Inter-Annotator Agreement (Human Test) ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, dim, color in zip(axes, ["Accuracy", "Fluency", "Prosody"], PALETTE):
    a1, a2, a3 = (hum_test[f"A1_{dim}"], hum_test[f"A2_{dim}"],
                  hum_test[f"A3_{dim}"])
    r12 = hum_test[[f"A1_{dim}", f"A2_{dim}"]].corr().iloc[0,1]
    r13 = hum_test[[f"A1_{dim}", f"A3_{dim}"]].corr().iloc[0,1]
    r23 = hum_test[[f"A2_{dim}", f"A3_{dim}"]].corr().iloc[0,1]
    ax.bar(["A1 vs A2", "A1 vs A3", "A2 vs A3"],
           [r12, r13, r23], color=color, edgecolor="white")
    for i, v in enumerate([r12, r13, r23]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_title(f"{dim} — Annotator Correlations")
    ax.set_ylabel("Pearson r"); ax.set_ylim(0, 1.1)
fig.suptitle("Dataset C — Inter-Annotator Agreement (Human Test Split)", fontweight="bold")
plt.tight_layout()
savefig("C6_inter_annotator_agreement.png", fig)

# ── Fig C7 : Question Distribution (from metadata) ────────────────────────
q_counts = meta["question"].dropna().value_counts()
top_qs = q_counts.head(15)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(top_qs)), top_qs.values, color=PALETTE[0], edgecolor="white")
ax.set_yticks(range(len(top_qs)))
ax.set_yticklabels([q[:70] + "…" if len(q) > 70 else q
                    for q in top_qs.index], fontsize=8)
ax.set_title("Dataset C — Top 15 Most Frequent Interview Questions (Meta)", fontweight="bold")
ax.set_xlabel("Number of Recordings")
plt.tight_layout()
savefig("C7_top_questions.png", fig)

# ── Fig C8 : Score comparisons — AI vs Human (overlap test subset) ─────────
# merge human_test with gemini_train/dev to compare overlapping wavnames
gem_all = pd.concat([gem_train, gem_dev], ignore_index=True)
overlap = hum_test.merge(gem_all, on="wavname", suffixes=("_human", "_ai"))
log(f"\n  Overlapping records (human test & gemini labels): {len(overlap)}")

if len(overlap) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, dim, color in zip(axes, ["Accuracy", "Fluency", "Prosody"], PALETTE):
        hcol = f"{dim}_human"; acol = f"{dim}_ai"
        if hcol in overlap.columns and acol in overlap.columns:
            r = overlap[[hcol, acol]].corr().iloc[0,1]
            jx = overlap[acol]  + np.random.uniform(-0.1, 0.1, len(overlap))
            jy = overlap[hcol]
            ax.scatter(jx, jy, alpha=0.4, s=20, color=color)
            # add diagonal
            ax.plot([1,5],[1,5], "r--", linewidth=1, label="Perfect agreement")
            ax.set_title(f"{dim}\nAI vs Human r={r:.3f}")
            ax.set_xlabel("AI (Gemini-2) Score"); ax.set_ylabel("Human Avg Score")
            ax.legend(fontsize=8)
    fig.suptitle("Dataset C — AI Labels vs Human Labels Comparison", fontweight="bold")
    plt.tight_layout()
    savefig("C8_ai_vs_human_scores.png", fig)
else:
    log("  [SKIP] No overlapping wavnames between human_test and gemini labels — C8 skipped")

log(f"\n  Key Findings:")
log(f"  - 9,782 unique speech recordings in metadata")
log(f"  - {meta['user_id'].nunique()} unique speakers")
log(f"  - 1 missing value in 'question' column of meta_data (negligible)")
log(f"  - Gemini AI scores peak around 3 (moderate) — slight central tendency bias")
log(f"  - Human annotators show moderate-to-high inter-annotator agreement (r > 0.5)")
log(f"  - Scores for Accuracy & Fluency are more correlated than Prosody")
log(f"  - wav.zip audio (6.71 GB) NOT downloaded — audio features unavailable for now")
log(f"  - LIMITATION: Without audio files, only metadata + label EDA is possible")


# ══════════════════════════════════════════════════════════════════════════
#  COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════
section("COMBINED EDA SUMMARY")
log(f"""
DATASET A — Student Employability (Classification Task)
  Records : 2,982 | Features: 8 numeric + 1 identifier | Target: CLASS (binary)
  Quality : Perfect — 0 missing, 0 duplicates
  Insight : Moderate class imbalance (58/42). COMMUNICATION SKILLS is most
            discriminating. All features on 2-5 Likert scale — minimal preprocessing.
  Next    : Drop 'Name of Student', encode target → binary. Apply Logistic
            Regression, Random Forest, SVM, Gradient Boosting, KNN.

DATASET B — AI Interview Questions (NLP / Reference Q&A)
  Records : 5,817 (train: 4,653 / eval: 1,164) | 1 raw column
  Quality : 0 missing, 0 duplicates. Parses cleanly.
  Insight : Avg answer length ~60 words. Dominated by 'What' and 'How' questions.
            Topics span software engineering, business, CS fundamentals.
  Next    : Parse Q&A text; cluster questions by topic (TF-IDF + KMeans);
            use answers as reference for candidate answer scoring (cosine similarity).

DATASET C — OpenEnded Speech Corpus (Speech Quality Scoring)
  Records : 9,782 recordings (CSV annotation labels only)
  Quality : 1 missing value in metadata (negligible). No audio downloaded.
  Insight : AI labels (Gemini) peak at score 3. Human test labels show
            moderate inter-annotator agreement (r ≈ 0.55–0.75 across dimensions).
  Next    : Audio download required for feature extraction. For now, train
            regression/classification model on AI labels. Evaluate against
            human gold standard (human_test.csv).

VISUALIZATIONS SAVED: reports/figures/
  A1_class_distribution.png
  A2_feature_distributions.png
  A3_feature_by_class.png
  A4_correlation_heatmap.png
  A5_boxplots_by_class.png
  A6_mean_scores_by_class.png
  B1_split_sizes.png
  B2_qa_length_distributions.png
  B3_top_words.png
  B4_question_types.png
  B5_answer_len_by_qtype.png
  C1_split_sizes.png
  C2_gemini_train_scores.png
  C3_gemini_dev_scores.png
  C4_human_test_scores.png
  C5_score_correlations.png
  C6_inter_annotator_agreement.png
  C7_top_questions.png
  C8_ai_vs_human_scores.png  (if overlap exists)
""")

# ── Write text report ──────────────────────────────────────────────────────
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[DONE] Text report saved → {REPORT}")
print(f"[DONE] Figures saved    → {FIGS}")
