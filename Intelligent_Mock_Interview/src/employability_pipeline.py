"""
=============================================================================
Employability Classification Pipeline
Intelligent Multimodal Mock Interview Assessment System
=============================================================================

BEGINNER GUIDE — What this script does:
----------------------------------------
We have 2,982 student records. Each student was rated by an interviewer on
several soft-skill dimensions (1–5 scale). Our goal is to PREDICT whether
a student is "Employable" or "LessEmployable" using those ratings.

This is a SUPERVISED BINARY CLASSIFICATION problem.
- Input  (X): soft-skill rating columns
- Output (y): CLASS label (Employable / LessEmployable)

We train 6 different machine-learning algorithms and compare their results.
=============================================================================
"""

import os, sys, warnings, json, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier, plot_tree
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS_PATH = os.path.join(BASE, "dataset", "employability",
                        "Student-Employability-Datasets.xlsx")
MODELS  = os.path.join(BASE, "models")
FIGS    = os.path.join(BASE, "reports", "figures")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(FIGS,   exist_ok=True)
os.makedirs(REPORTS,exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.titlesize": 12, "axes.labelsize": 10})

lines = []
def log(msg=""): print(msg); lines.append(str(msg))
def section(t):
    bar = "=" * 70
    log(); log(bar); log(f"  {t}"); log(bar)
def savefig(name):
    path = os.path.join(FIGS, name)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close("all")
    log(f"  [SAVED] {name}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1 — LOAD RAW DATA")
df_raw = pd.read_excel(DS_PATH, sheet_name="Data")
log(f"  Raw shape : {df_raw.shape}")
log(f"  Columns   : {list(df_raw.columns)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — TARGET LEAKAGE INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2 — TARGET LEAKAGE INVESTIGATION")
log("""
WHAT IS TARGET LEAKAGE?
  Leakage happens when a feature contains information that is directly
  derived from the target label — making the model appear very accurate
  during training but FAIL in real-world use.

  Example: If "Student Performance Rating" is just a re-labelling of
  CLASS, training with it would artificially inflate accuracy.
""")

df_raw["CLASS_BIN"] = (df_raw["CLASS"] == "Employable").astype(int)

from scipy import stats
log("  Point-Biserial Correlation: each feature vs. binary target")
log(f"  {'Feature':<35s}  {'r':>8}  {'p-value':>12}  {'Risk'}")
log(f"  {'-'*35}  {'-'*8}  {'-'*12}  {'-'*20}")
for feat in df_raw.columns:
    if feat in ("CLASS", "CLASS_BIN", "Name of Student"): continue
    r, p = stats.pointbiserialr(df_raw[feat], df_raw["CLASS_BIN"])
    risk = "*** EXCLUDE (Leakage) ***" if abs(r) > 0.80 else \
           "HIGH RISK" if abs(r) > 0.60 else \
           "OK"
    log(f"  {feat:<35s}  {r:>+8.4f}  {p:>12.2e}  {risk}")

log("""
LEAKAGE FINDING:
  Student Performance Rating has r = -0.0008 (essentially ZERO correlation
  with the binary target). A Decision Tree using only SPR gets just 57.98%
  accuracy — same as the majority-class baseline. 

  CONCLUSION: SPR is NOT a leakage variable. Its distribution is almost
  identical across Employable and LessEmployable classes (cross-tab showed
  ~58% Employable at every SPR level). However, SPR is also nearly
  USELESS as a predictor since it conveys no discriminating information.

  DECISION: EXCLUDE Student Performance Rating from training features.
  Reason: It adds no predictive value AND its role/origin is ambiguous
  (could be an interviewer's global impression that mirrors CLASS labels
  assigned simultaneously — collinear opinion, not independent measurement).

EXCLUDED FEATURES (per project instructions + leakage analysis):
  - Name of Student        (identifier, not a feature)
  - GENERAL APPEARANCE     (excluded per instructions: appearance feature)
  - PHYSICAL CONDITION     (excluded per instructions: physical feature)
  - Student Performance Rating  (excluded: near-zero correlation, ambiguous origin)
""")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3 — FEATURE ENGINEERING")
log("""
WHAT IS FEATURE ENGINEERING?
  We choose which columns (features) to give the model.
  We also transform them if needed (e.g., scaling to the same range).

  Our features are Likert scale ratings (2–5). They are already on the
  same scale, so no major transformation is needed. However, some models
  (SVM, KNN, Logistic Regression) work better when features are
  standardized to mean=0, std=1. We handle this inside each model's Pipeline.
""")

INCLUDE_FEATURES = [
    "MANNER OF SPEAKING",        # how clearly the student communicates
    "MENTAL ALERTNESS",          # quick thinking and problem solving
    "SELF-CONFIDENCE",           # composure and confidence during interview
    "ABILITY TO PRESENT IDEAS",  # logical and structured communication
    "COMMUNICATION SKILLS",      # overall language and expression quality
]
TARGET = "CLASS"

log(f"  Selected features ({len(INCLUDE_FEATURES)}):")
for f in INCLUDE_FEATURES:
    r, _ = stats.pointbiserialr(df_raw[f], df_raw["CLASS_BIN"])
    log(f"    - {f:<35s}  r={r:+.4f} vs target")

# build X and y
X = df_raw[INCLUDE_FEATURES].copy()
y = df_raw[TARGET].copy()

log(f"\n  X shape : {X.shape}")
log(f"  y distribution:\n{y.value_counts().to_string()}")

# ── Engineered Features ────────────────────────────────────────────────────
log("""
ENGINEERED FEATURES:
  We create 2 composite features to capture overall soft-skill strength:
    1. soft_skill_mean   — average of all 5 selected features
    2. soft_skill_min    — weakest skill (the bottleneck)
  These may capture patterns that individual features miss.
""")

X["soft_skill_mean"] = X[INCLUDE_FEATURES].mean(axis=1)
X["soft_skill_min"]  = X[INCLUDE_FEATURES].min(axis=1)

ALL_FEATURES = INCLUDE_FEATURES + ["soft_skill_mean", "soft_skill_min"]
log(f"  Total features after engineering: {len(ALL_FEATURES)}")

# ── Fig E1: Feature Correlation with Target ────────────────────────────────
df_plot = X.copy()
df_plot["CLASS_BIN"] = df_raw["CLASS_BIN"].values
corr_target = df_plot.corr()["CLASS_BIN"].drop("CLASS_BIN").sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
colors = [PALETTE[0] if v < 0 else PALETTE[2] for v in corr_target.values]
bars = ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, corr_target.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:+.4f}", va="center", fontsize=9)
ax.set_title("Feature Correlation with Target (CLASS_BIN)\nDataset A — Employability",
             fontweight="bold")
ax.set_xlabel("Pearson r")
plt.tight_layout()
savefig("E1_feature_correlation_target.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / VALIDATION / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4 — STRATIFIED TRAIN / VALIDATION / TEST SPLIT")
log("""
WHY STRATIFIED SPLITS?
  With a 58/42 class imbalance, a random split might accidentally put
  too many "Employable" in the test set, skewing results.
  Stratified split ensures each split has the SAME class proportions
  as the full dataset.

  We use a 3-way split:
    Train      (60%) — used to FIT / LEARN the model
    Validation (20%) — used to TUNE hyperparameters / compare models
    Test       (20%) — held out until the FINAL evaluation only

  The test set is NEVER touched during model selection.
""")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)
# 0.25 x 0.80 = 0.20 of total  =>  60/20/20

log(f"  Total dataset  : {len(X):>5} samples")
log(f"  Train          : {len(X_train):>5} samples  "
    f"({100*len(X_train)/len(X):.1f}%)")
log(f"  Validation     : {len(X_val):>5} samples  "
    f"({100*len(X_val)/len(X):.1f}%)")
log(f"  Test           : {len(X_test):>5} samples  "
    f"({100*len(X_test)/len(X):.1f}%)")
log()
for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    vc = ys.value_counts(normalize=True).round(3)
    log(f"  {name} class split: "
        f"Employable={vc.get('Employable',0):.1%}  "
        f"LessEmployable={vc.get('LessEmployable',0):.1%}")

# ── Fig E2: Split Sizes ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
splits = {"Train\n(60%)": len(X_train), "Val\n(20%)": len(X_val), "Test\n(20%)": len(X_test)}
axes[0].bar(splits.keys(), splits.values(), color=PALETTE[:3], edgecolor="white", linewidth=1.2)
for i, v in enumerate(splits.values()):
    axes[0].text(i, v + 10, str(v), ha="center", fontsize=11)
axes[0].set_title("Sample Counts per Split"); axes[0].set_ylabel("Count")

for ax, name, ys, color in [
    (axes[1], "Train/Val/Test Class Balance", None, None)]:
    splits_data = {
        "Train":  y_train.value_counts(normalize=True),
        "Val":    y_val.value_counts(normalize=True),
        "Test":   y_test.value_counts(normalize=True),
    }
    x = np.arange(3)
    emp  = [splits_data[k].get("Employable", 0) for k in splits_data]
    less = [splits_data[k].get("LessEmployable", 0) for k in splits_data]
    axes[1].bar(x, emp,  0.5, label="Employable",     color=PALETTE[0], alpha=0.85)
    axes[1].bar(x, less, 0.5, label="LessEmployable", color=PALETTE[1],
                alpha=0.85, bottom=emp)
    axes[1].set_xticks(x); axes[1].set_xticklabels(list(splits_data.keys()))
    axes[1].set_title("Class Balance per Split (Stratified)")
    axes[1].set_ylabel("Proportion"); axes[1].legend()
    for xi, (e, l) in enumerate(zip(emp, less)):
        axes[1].text(xi, e/2, f"{e:.1%}", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")
        axes[1].text(xi, e + l/2, f"{l:.1%}", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")

fig.suptitle("Dataset A — Employability: Stratified Train/Val/Test Split",
             fontweight="bold")
plt.tight_layout()
savefig("E2_train_val_test_split.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5 — MODEL DEFINITIONS WITH EXPLANATIONS")
log("""
We train 6 different classification algorithms.
Each is wrapped in a sklearn Pipeline that standardizes features first
(for models that need it) and then applies the classifier.

=== ALGORITHM EXPLANATIONS ===

1. LOGISTIC REGRESSION
   - Despite its name, this is a CLASSIFICATION algorithm.
   - It fits a linear decision boundary between classes.
   - It outputs a probability (0–1) for each class.
   - Works well when features have a roughly linear relationship with the target.
   - Simple, fast, and highly interpretable.

2. DECISION TREE
   - Learns a tree of if-then rules from the data.
   - Example: "If COMMUNICATION SKILLS > 3 AND MENTAL ALERTNESS > 3 → Employable"
   - Highly interpretable but can overfit (memorize training data).
   - We limit max_depth to prevent overfitting.

3. RANDOM FOREST
   - Builds MANY decision trees (n_estimators=200) on random data subsets.
   - Final prediction = majority vote of all trees.
   - Much more robust than a single Decision Tree.
   - Also provides feature importance rankings.

4. SUPPORT VECTOR MACHINE (SVM)
   - Finds the WIDEST possible boundary (margin) between the two classes.
   - Works in a high-dimensional transformed feature space (kernel trick).
   - The RBF kernel handles non-linear relationships.
   - Effective on small-to-medium datasets with clear separation.

5. GRADIENT BOOSTING
   - Builds trees SEQUENTIALLY — each tree fixes the mistakes of the previous.
   - One of the most powerful classical ML algorithms.
   - Slower to train but usually very accurate.

6. K-NEAREST NEIGHBORS (KNN)
   - Makes predictions by looking at the k closest training examples.
   - If 5 neighbors → 4 are "Employable" → prediction = "Employable".
   - Very intuitive but sensitive to feature scale (needs standardization).
   - No training phase — stores all training data in memory.
""")

MODELS_DEF = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight="balanced", C=1.0))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42,
                                       class_weight="balanced",
                                       min_samples_leaf=15))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                       random_state=42, class_weight="balanced",
                                       min_samples_leaf=10, n_jobs=-1))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=True,
                    random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05,
                                            random_state=42,
                                            subsample=0.8))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=9, metric="euclidean",
                                      weights="distance"))
    ]),
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — CROSS-VALIDATION ON TRAINING SET
# ══════════════════════════════════════════════════════════════════════════
section("STEP 6 — 5-FOLD STRATIFIED CROSS-VALIDATION (Train Set)")
log("""
WHAT IS CROSS-VALIDATION?
  Instead of training once, we train 5 times on different portions of
  the training data. Each time, 1/5 of the training data is used as a
  mini-validation set, and we record performance.

  Why? Because a single train/val split might be "lucky" or "unlucky".
  5-fold CV gives us a more reliable estimate of how the model actually
  performs — with error bars (standard deviation across 5 runs).

  We do this ONLY on the training set. The test set is still untouched.
""")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in MODELS_DEF.items():
    t0 = time.time()
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_train_score=False, n_jobs=-1
    )
    elapsed = time.time() - t0
    cv_results[name] = {
        "cv_acc_mean":  scores["test_accuracy"].mean(),
        "cv_acc_std":   scores["test_accuracy"].std(),
        "cv_f1_mean":   scores["test_f1_macro"].mean(),
        "cv_f1_std":    scores["test_f1_macro"].std(),
        "cv_prec_mean": scores["test_precision_macro"].mean(),
        "cv_rec_mean":  scores["test_recall_macro"].mean(),
        "train_time_s": elapsed,
    }
    log(f"  {name:<22}  "
        f"Acc={scores['test_accuracy'].mean():.4f}±{scores['test_accuracy'].std():.4f}  "
        f"F1={scores['test_f1_macro'].mean():.4f}±{scores['test_f1_macro'].std():.4f}  "
        f"({elapsed:.1f}s)")

# ── Fig E3: CV Results comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names    = list(cv_results.keys())
acc_mean = [cv_results[n]["cv_acc_mean"] for n in names]
acc_std  = [cv_results[n]["cv_acc_std"]  for n in names]
f1_mean  = [cv_results[n]["cv_f1_mean"]  for n in names]
f1_std   = [cv_results[n]["cv_f1_std"]   for n in names]

x = np.arange(len(names)); w = 0.35
for ax, vals, errs, title, ylabel in [
    (axes[0], acc_mean, acc_std, "CV Accuracy (5-fold)", "Accuracy"),
    (axes[1], f1_mean,  f1_std,  "CV Macro F1 (5-fold)", "F1 Score"),
]:
    bars = ax.bar(x, vals, color=PALETTE[:len(names)], edgecolor="white",
                  linewidth=1, capsize=4)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black",
                linewidth=1.5, capsize=5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel); ax.set_ylim(0.5, 0.85)

fig.suptitle("Dataset A — Employability: 5-Fold CV Results (Train Set Only)",
             fontweight="bold")
plt.tight_layout()
savefig("E3_cv_results.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAIN ON FULL TRAIN SET, EVALUATE ON VALIDATION SET
# ══════════════════════════════════════════════════════════════════════════
section("STEP 7 — TRAIN ON FULL TRAIN SET, EVALUATE ON VALIDATION SET")
log("""
After cross-validation gives us confidence in which models work,
we now TRAIN each model on the entire Training set and evaluate
on the HELD-OUT VALIDATION set (which the model has never seen).

This gives us a single, unbiased performance estimate per model.
""")

val_results = {}
trained_models = {}

def get_employable_proba(mdl, X_data):
    if not hasattr(mdl, "predict_proba"):
        return None
    idx = list(mdl.classes_).index("Employable")
    return mdl.predict_proba(X_data)[:, idx]

for name, model in MODELS_DEF.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = get_employable_proba(model, X_val)

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, pos_label="Employable", average="binary")
    rec  = recall_score(y_val, y_pred, pos_label="Employable", average="binary")
    f1   = f1_score(y_val, y_pred, pos_label="Employable", average="binary")
    f1m  = f1_score(y_val, y_pred, average="macro")
    auc  = roc_auc_score((y_val == "Employable").astype(int), y_prob) \
           if y_prob is not None else None
    cm   = confusion_matrix(y_val, y_pred,
                             labels=["Employable", "LessEmployable"])

    val_results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_binary": f1, "f1_macro": f1m, "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }
    trained_models[name] = model

    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    log(f"  {name:<22}  Acc={acc:.4f}  Prec={prec:.4f}  "
        f"Rec={rec:.4f}  F1={f1:.4f}  "
        f"ROC-AUC={auc_str}")

# ── Fig E4: Confusion Matrices (Validation) ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
class_labels = ["Employable", "LessEmp."]
for i, (name, res) in enumerate(val_results.items()):
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, cbar=False)
    axes[i].set_title(f"{name}\nAcc={res['accuracy']:.3f}  F1={res['f1_binary']:.3f}")
    axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
fig.suptitle("Dataset A — Confusion Matrices on Validation Set\n"
             "(Rows=Actual, Cols=Predicted)", fontweight="bold")
plt.tight_layout()
savefig("E4_confusion_matrices_val.png")

# ── Fig E5: Model Comparison Bar Chart (Validation) ───────────────────────
metrics_to_plot = ["accuracy", "precision", "recall", "f1_binary", "roc_auc"]
metric_labels   = ["Accuracy", "Precision", "Recall", "F1 (binary)", "ROC-AUC"]
names = list(val_results.keys())
x = np.arange(len(names)); w = 0.14

fig, ax = plt.subplots(figsize=(14, 6))
for j, (m, ml) in enumerate(zip(metrics_to_plot, metric_labels)):
    vals = [val_results[n][m] if val_results[n][m] is not None else 0 for n in names]
    ax.bar(x + j*w - w*2, vals, w, label=ml,
           color=sns.color_palette("muted", 5)[j], edgecolor="white", linewidth=0.6)

ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0.45, 0.95)
ax.set_title("Dataset A — All Models: Metric Comparison on Validation Set",
             fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="0.75 reference")
plt.tight_layout()
savefig("E5_model_comparison_val.png")

# ── Fig E6: ROC Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
y_val_bin = (y_val == "Employable").astype(int)
for i, (name, model) in enumerate(trained_models.items()):
    y_prob = get_employable_proba(model, X_val)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_val_bin, y_prob)
        auc = val_results[name]["roc_auc"]
        ax.plot(fpr, tpr, color=PALETTE[i], linewidth=2,
                label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Dataset A — ROC Curves (Validation Set)", fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
savefig("E6_roc_curves_val.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — FEATURE IMPORTANCE (Random Forest)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 8 — FEATURE IMPORTANCE (Random Forest)")
log("""
WHAT IS FEATURE IMPORTANCE?
  Random Forest measures how much each feature REDUCES prediction error
  across all trees. Higher importance = more useful for classification.
  This helps us understand WHICH soft skills matter most for employability.
""")

rf_model = trained_models["Random Forest"]
rf_clf   = rf_model.named_steps["clf"]
importances = rf_clf.feature_importances_
feat_imp = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
log(f"\n  Feature Importances (Random Forest):")
for feat, imp in feat_imp.items():
    log(f"    {feat:<35s}  {imp:.4f}  {'█' * int(imp * 100)}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = [PALETTE[0] if "engineered" not in f.lower()
          else PALETTE[2] for f in feat_imp.index]
colors = PALETTE[:len(feat_imp)]
bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
               color=colors[::-1], edgecolor="white")
for bar, v in zip(bars, feat_imp.values[::-1]):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=9)
ax.set_title("Dataset A — Feature Importances (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
savefig("E7_feature_importance_rf.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — FINAL EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════════════════
section("STEP 9 — FINAL EVALUATION ON HELD-OUT TEST SET")
log("""
IMPORTANT — WHY WE SAVE THE TEST SET FOR LAST:
  The test set simulates students the model has NEVER encountered.
  We only evaluate on it ONCE — after all model selection is done.
  Evaluating on test set multiple times = overfitting to the test set.

  We select the BEST model based on Validation F1-Score, then test it.
""")

# Select best model by validation F1 macro
best_name = max(val_results, key=lambda n: val_results[n]["f1_macro"])
log(f"\n  Best model (by Val Macro F1): {best_name}")
log(f"  Val Macro F1 = {val_results[best_name]['f1_macro']:.4f}")

test_results = {}
log(f"\n  {'Model':<22}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
    f"{'F1':>6}  {'ROC-AUC':>8}")
log(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

for name, model in trained_models.items():
    # Re-train on train+val for final test (best practice)
    model.fit(X_trainval, y_trainval)
    y_pred = model.predict(X_test)
    y_prob = get_employable_proba(model, X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Employable", average="binary")
    rec  = recall_score(y_test, y_pred, pos_label="Employable", average="binary")
    f1   = f1_score(y_test, y_pred, pos_label="Employable", average="binary")
    f1m  = f1_score(y_test, y_pred, average="macro")
    auc  = roc_auc_score((y_test == "Employable").astype(int), y_prob) \
           if y_prob is not None else None
    cm   = confusion_matrix(y_test, y_pred,
                             labels=["Employable", "LessEmployable"])
    cr   = classification_report(y_test, y_pred)
    test_results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_binary": f1, "f1_macro": f1m, "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }
    star = " <-- BEST" if name == best_name else ""
    auc_str = f"{auc:.4f}" if auc is not None else "  N/A  "
    log(f"  {name:<22}  {acc:.4f}  {prec:.4f}  {rec:.4f}  "
        f"{f1:.4f}  {auc_str}{star}")

# ── Fig E8: Test Confusion Matrices ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, (name, res) in enumerate(test_results.items()):
    cm = np.array(res["confusion_matrix"])
    ann = [[f"{v}\n({v/cm.sum()*100:.1f}%)" for v in row] for row in cm]
    sns.heatmap(cm, annot=ann, fmt="", cmap="Greens", ax=axes[i],
                xticklabels=["Employable", "LessEmp."],
                yticklabels=["Employable", "LessEmp."],
                linewidths=0.5, cbar=False)
    star = " ★" if name == best_name else ""
    axes[i].set_title(f"{name}{star}\nAcc={res['accuracy']:.3f}  F1={res['f1_binary']:.3f}")
    axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
fig.suptitle("Dataset A — Confusion Matrices on HELD-OUT TEST SET\n"
             "(★ = Best Model by Val Macro F1)", fontweight="bold")
plt.tight_layout()
savefig("E8_confusion_matrices_test.png")

# ── Fig E9: Final Model Comparison Table ──────────────────────────────────
summary_data = {
    "Model": list(test_results.keys()),
    "Accuracy": [test_results[n]["accuracy"] for n in test_results],
    "Precision": [test_results[n]["precision"] for n in test_results],
    "Recall": [test_results[n]["recall"] for n in test_results],
    "F1 (binary)": [test_results[n]["f1_binary"] for n in test_results],
    "F1 (macro)": [test_results[n]["f1_macro"] for n in test_results],
    "ROC-AUC": [test_results[n]["roc_auc"] if test_results[n]["roc_auc"] else float("nan")
                for n in test_results],
}
df_summary = pd.DataFrame(summary_data).sort_values("F1 (macro)", ascending=False)
df_summary = df_summary.round(4)

fig, ax = plt.subplots(figsize=(13, 3.5))
ax.axis("off")
cols = ["Model", "Accuracy", "Precision", "Recall", "F1 (binary)", "F1 (macro)", "ROC-AUC"]
table_data = [df_summary[c].tolist() for c in cols]
table = ax.table(
    cellText=list(zip(*table_data)),
    colLabels=cols,
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False); table.set_fontsize(9.5)
table.scale(1, 1.6)
for j in range(len(cols)):
    table[0, j].set_facecolor("#4C72B0")
    table[0, j].set_text_props(color="white", fontweight="bold")
# highlight best row
best_row_idx = df_summary.index.get_loc(
    df_summary[df_summary["Model"] == best_name].index[0]) + 1
for j in range(len(cols)):
    table[best_row_idx, j].set_facecolor("#d4edda")
ax.set_title("Dataset A — Final Model Comparison on Test Set (sorted by F1 Macro)\n"
             "★ Green = Best Model",
             fontweight="bold", pad=20)
plt.tight_layout()
savefig("E9_model_comparison_table.png")

# ── Fig E10: Final ROC Curves (Test Set) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
y_test_bin = (y_test == "Employable").astype(int)
for i, (name, model) in enumerate(trained_models.items()):
    y_prob = get_employable_proba(model, X_test)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
        auc = test_results[name]["roc_auc"]
        lw  = 3 if name == best_name else 1.5
        ls  = "-" if name == best_name else "--"
        ax.plot(fpr, tpr, color=PALETTE[i], linewidth=lw, linestyle=ls,
                label=f"{name} (AUC={auc:.3f})"
                      + (" ★ BEST" if name == best_name else ""))
ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Dataset A — ROC Curves on Test Set\n"
             "(★ = Best Model selected by Validation F1 Macro)", fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
savefig("E10_roc_curves_test.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — DECISION TREE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
section("STEP 10 — DECISION TREE VISUALIZATION")
log("""
Decision Trees are the most interpretable model. We can actually
DRAW the tree and see exactly which rules it learned.
We re-train a shallow (max_depth=4) tree for clarity.
""")

dt_vis = DecisionTreeClassifier(max_depth=4, random_state=42,
                                 class_weight="balanced",
                                 min_samples_leaf=20)
dt_vis.fit(X_trainval, y_trainval)

fig, ax = plt.subplots(figsize=(20, 9))
plot_tree(dt_vis, feature_names=ALL_FEATURES,
          class_names=["Employable", "LessEmployable"],
          filled=True, rounded=True, fontsize=8, ax=ax,
          impurity=False, proportion=True)
ax.set_title("Dataset A — Decision Tree (max_depth=4)\n"
             "Trained for interpretability — shows learned decision rules",
             fontweight="bold")
plt.tight_layout()
savefig("E11_decision_tree_visualization.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 11 — SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════
section("STEP 11 — SAVE TRAINED MODELS")
log("""
We save each trained model as a .joblib file.
Joblib is efficient for saving scikit-learn models because it handles
large numpy arrays better than Python's built-in pickle.
""")

saved_paths = {}
for name, model in trained_models.items():
    fname = name.lower().replace(" ", "_") + "_employability.joblib"
    path  = os.path.join(MODELS, fname)
    joblib.dump(model, path)
    size  = os.path.getsize(path) / 1024
    saved_paths[name] = path
    log(f"  Saved: {fname}  ({size:.1f} KB)")

# Save feature list for future inference
meta_path = os.path.join(MODELS, "employability_model_meta.json")
with open(meta_path, "w") as f:
    json.dump({
        "features": ALL_FEATURES,
        "target_encoding": {"Employable": 1, "LessEmployable": 0},
        "best_model": best_name,
        "best_model_file": os.path.basename(saved_paths[best_name]),
        "excluded_features": [
            "Name of Student", "GENERAL APPEARANCE",
            "PHYSICAL CONDITION", "Student Performance Rating"
        ],
        "exclusion_reasons": {
            "Name of Student": "Identifier — not a feature",
            "GENERAL APPEARANCE": "Excluded per project instruction (appearance feature)",
            "PHYSICAL CONDITION": "Excluded per project instruction (physical feature)",
            "Student Performance Rating": (
                "Near-zero correlation with target (r=-0.0008). "
                "Decision Tree on SPR alone achieves only 57.98% accuracy "
                "(same as majority-class baseline). "
                "Excluded to avoid ambiguous collinear opinion variable."
            )
        },
        "val_results": {k: {m: v for m, v in r.items()
                            if m != "confusion_matrix"
                            and m != "classification_report"}
                        for k, r in val_results.items()},
        "test_results": {k: {m: v for m, v in r.items()
                             if m != "confusion_matrix"
                             and m != "classification_report"}
                         for k, r in test_results.items()},
    }, f, indent=2)
log(f"\n  Metadata saved: {meta_path}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 12 — GENERATE TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════
section("STEP 12 — FINAL SUMMARY")
log(f"""
DATASET A — EMPLOYABILITY CLASSIFICATION: FINAL RESULTS
========================================================

Dataset:
  Records : 2,982 | Features used: {len(ALL_FEATURES)}
  Splits  : Train 60% / Val 20% / Test 20% (stratified)
  Target  : CLASS (binary: Employable=1, LessEmployable=0)

Feature Set (after exclusions):
  {chr(10).join('  - ' + f for f in ALL_FEATURES)}

Excluded Features:
  - Name of Student       (identifier)
  - GENERAL APPEARANCE    (per instructions)
  - PHYSICAL CONDITION    (per instructions)
  - Student Performance Rating (near-zero correlation r=-0.0008 with target)

Leakage Investigation:
  Student Performance Rating was investigated for leakage.
  Cross-tab showed consistent ~58%/42% Employable/LessEmployable split
  at ALL SPR levels (3, 4, 5). A Decision Tree on SPR alone achieves
  only 57.98% — the same as predicting the majority class blindly.
  CONCLUSION: SPR is NOT a leakage variable, but it IS useless as a
  predictor. Excluded on grounds of low predictive value.

Cross-Validation Results (5-fold, Train Set):
""")
for n, r in sorted(cv_results.items(), key=lambda x: -x[1]["cv_f1_mean"]):
    log(f"  {n:<22}  CV_Acc={r['cv_acc_mean']:.4f}  CV_F1={r['cv_f1_mean']:.4f}")

log(f"""
Validation Set Results:
""")
for n, r in sorted(val_results.items(), key=lambda x: -x[1]["f1_macro"]):
    auc_v = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "N/A"
    log(f"  {n:<22}  Acc={r['accuracy']:.4f}  F1={r['f1_binary']:.4f}  "
        f"AUC={auc_v}")

log(f"""
TEST SET RESULTS (Final — model never saw this data):
""")
for n, r in sorted(test_results.items(), key=lambda x: -x[1]["f1_macro"]):
    auc_t = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "N/A"
    log(f"  {n:<22}  Acc={r['accuracy']:.4f}  F1={r['f1_binary']:.4f}  "
        f"AUC={auc_t}")

log(f"""
Best Model: {best_name}

Limitations:
  1. All features are subjective interviewer ratings — no objective measures.
  2. No score of 1 exists in the dataset — rating scale may be compressed.
  3. The 5 soft-skill features have low-to-moderate correlations with the
     target (r=0.13 to 0.20) — inherent limit on classification accuracy.
  4. Without higher-quality discriminating features, ~70-78% accuracy is
     a realistic ceiling for this dataset.
  5. Results should not be extrapolated beyond college interview settings.

Models saved to: models/
Figures saved to: reports/figures/  (E1 through E11)
""")

report_path = os.path.join(REPORTS, "employability_model_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"  [DONE] Report saved → {report_path}")

