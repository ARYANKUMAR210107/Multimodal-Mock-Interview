# Algorithms and Mathematics Used in the Project

This reference is tied to the code in `src/employability_pipeline.py`, `src/text_pipeline.py`, and `src/audio_feature_extractor.py`. It explains implemented methods; it does not imply that the interview text/audio systems were trained as supervised models.

## 1. Employability classification data and features

The classification target is the string `CLASS`, either `Employable` or `LessEmployable`. The input starts with five ratings:

- `MANNER OF SPEAKING`
- `MENTAL ALERTNESS`
- `SELF-CONFIDENCE`
- `ABILITY TO PRESENT IDEAS`
- `COMMUNICATION SKILLS`

Two additional inputs are engineered per row:

\[
\text{soft\_skill\_mean}=\frac{1}{5}\sum_{j=1}^{5}x_j,
\qquad
\text{soft\_skill\_min}=\min(x_1,\ldots,x_5)
\]

Thus the models receive seven columns. `Name of Student`, `GENERAL APPEARANCE`, `PHYSICAL CONDITION`, and `Student Performance Rating` are excluded. The pipeline investigates Student Performance Rating with point-biserial correlation; its recorded correlation is approximately −0.0008, so it is not retained. The dataset contains 2,982 rows. Class counts are 1,729 Employable and 1,253 LessEmployable.

The split uses `random_state=42` and `stratify=y`: 20% first held out for test (597); the remaining 80% is split 75/25 into train (1,788) and validation (597). The approximate proportions of each total are 60/20/20. The split preserves class prevalence.

## 2. Scaling and pipeline fitting

`StandardScaler` transforms a feature by

\[
z=\frac{x-\mu_{train}}{\sigma_{train}}
\]

where mean and standard deviation are learned on that training fold. Scaling is used in the Logistic Regression, SVM, Gradient Boosting, and KNN pipelines; the Decision Tree and Random Forest pipelines use raw Likert/aggregate values. Trees do not depend on feature units for their threshold ordering. Scaling is especially relevant to distance-based KNN and margin-based SVM. The five ratings already share the same nominal scale.

Each model is fit inside an scikit-learn `Pipeline`, so a scaler is refit within each cross-validation fold rather than using validation/test statistics. The training script uses a five-fold shuffled `StratifiedKFold` on the 1,788 training rows, with `random_state=42`. It then fits each model on the whole training split for validation evaluation. After selecting the model by validation **macro F1**, the script refits the model definitions on train plus validation (2,385 rows) and evaluates once on the 597-row test set. This is the intended flow recorded in source/report.

## 3. Six supervised algorithms actually trained

All models use the same 7 features and binary target. The dataset is small and numerical, and the script compares linear, tree, ensemble, margin, and instance-based methods rather than presuming one is best.

### Logistic Regression

It models the positive-class probability with a sigmoid over a weighted sum:

\[
P(y=1\mid x)=\sigma(w^Tx+b)=\frac{1}{1+e^{-(w^Tx+b)}}
\]

The decision is made from the learned probabilities/threshold. The configured estimator is `LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)` with scaling. It is a compact linear baseline and produces probabilities. It may underfit nonlinear relationships; it has the lowest saved accuracy/AUC among the six in both validation and test.

### Decision Tree

A tree repeatedly chooses a feature threshold that best separates labels. The implementation uses `DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=42)`. The default criterion is Gini impurity. For class proportions (p_k),

\[
Gini=1-\sum_k p_k^2
\]

The split's impurity reduction is weighted by child sizes. A shallow tree can be read as if/then rules; larger trees can overfit. The script also fits a separate depth-4 tree for the E11 visualization; that visualization-only tree is not the saved Decision Tree benchmark estimator.

### Random Forest

The estimator grows many decision trees using randomized bootstrap/features and aggregates their votes (probabilities are averaged by scikit-learn). Configuration: 200 trees, max depth 8, minimum leaf 10, balanced class weights, `n_jobs=-1`, and fixed seed. It reduces single-tree variance and supplies impurity-based feature importance, but its importance scores can favor correlated features and are not causal effects.

### Support Vector Machine (SVM)

The classifier seeks a wide-margin boundary. This project uses `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)` after standardization. The RBF kernel compares examples nonlinearly, commonly written

\[
K(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)
\]

`C` trades margin width against training errors. Probability estimates are enabled for ROC-AUC evaluation. SVM performs well here, but probabilities are not necessarily calibrated and kernel behavior depends on parameters/scaling.

### Gradient Boosting

Gradient boosting adds shallow trees sequentially. Each stage moves in the direction that reduces the loss on the current ensemble's errors. The project uses 200 estimators, max depth 4, learning rate 0.05, subsample 0.8, seed 42, preceded by a scaler. It is selected because it has the best validation macro F1 and then produces the best held-out accuracy/positive-class F1 in the recorded results. It remains vulnerable to noisy or biased labels and small-sample variation. Scaling is not ordinarily necessary for tree splits, though it is included in this saved pipeline.

### K-Nearest Neighbors (KNN)

For a new point, KNN finds nearby training observations and aggregates their labels. The project uses (k=9), Euclidean distance, `weights="distance"`, and standardized features. A common distance is

\[
d(x,z)=\sqrt{\sum_{j=1}^{7}(x_j-z_j)^2}
\]

Nearby examples receive more influence than farther ones. It is simple but prediction cost grows with retained data and behavior depends on scaling, feature choice, and neighborhood size.

## 4. Evaluation mathematics

Treat `Employable` as the positive class for binary precision, recall, and binary F1:

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
Precision=\frac{TP}{TP+FP},\qquad Recall=\frac{TP}{TP+FN}
\]

\[
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
\]

Macro F1 computes F1 independently for both classes and takes their unweighted average. It treats both labels equally, which is useful when class counts differ. A confusion matrix in the saved plots has **actual classes in rows and predicted classes in columns**, ordered `[Employable, LessEmployable]`.

ROC sweeps a probability threshold and plots true-positive rate against false-positive rate. AUC is the area under this curve; 0.5 represents chance ranking and 1.0 perfect ranking. AUC does not measure calibration or guarantee a useful threshold.

## 5. Model evaluation recorded in project outputs

### Five-fold cross-validation on training split

Values are the mean and standard deviation across five folds. Only accuracy and macro F1 are stored in the report's compact summary.

| Model | CV accuracy (mean ± std) | CV macro F1 (mean ± std) |
|---|---:|---:|
| Logistic Regression | 0.5984 ± 0.0232 | 0.5972 ± 0.0235 |
| Decision Tree | 0.6421 ± 0.0337 | 0.6362 ± 0.0310 |
| Random Forest | 0.7148 ± 0.0213 | 0.7072 ± 0.0213 |
| SVM | 0.7249 ± 0.0236 | 0.7134 ± 0.0226 |
| Gradient Boosting | **0.7634 ± 0.0125** | **0.7425 ± 0.0161** |
| KNN | 0.7338 ± 0.0179 | 0.7182 ± 0.0223 |

### Validation metrics

Precision, recall, and binary F1 are positive-class metrics. The extra macro F1 is included because it is the model-selection metric.

| Model | Accuracy | Precision | Recall | Binary F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5796 | 0.6568 | 0.5751 | 0.6133 | 0.5764 | 0.6001 |
| Decision Tree | 0.6432 | 0.7529 | 0.5723 | 0.6502 | 0.6431 | 0.7069 |
| Random Forest | 0.7052 | 0.7852 | 0.6763 | 0.7267 | 0.7034 | 0.8075 |
| SVM | 0.7303 | 0.7591 | 0.7832 | 0.7710 | 0.7215 | 0.7700 |
| Gradient Boosting | **0.7638** | 0.7531 | 0.8815 | **0.8123** | **0.7470** | **0.8347** |
| KNN | 0.6918 | 0.7355 | 0.7312 | 0.7333 | 0.6841 | 0.7898 |

### Held-out test metrics

| Model | Accuracy | Precision | Recall | Binary F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5930 | 0.6604 | 0.6127 | 0.6357 | 0.5873 | 0.6316 |
| Decision Tree | 0.6985 | 0.7139 | 0.8006 | 0.7548 | 0.6817 | 0.7766 |
| Random Forest | 0.7420 | 0.7697 | 0.7919 | 0.7806 | 0.7338 | 0.8132 |
| SVM | 0.7588 | 0.7440 | 0.8902 | 0.8105 | 0.7394 | 0.7926 |
| Gradient Boosting | **0.7688** | 0.7419 | **0.9220** | **0.8222** | **0.7460** | **0.8248** |
| KNN | 0.7119 | 0.7574 | 0.7399 | 0.7485 | 0.7056 | 0.7849 |

Gradient Boosting is the selected model because validation macro F1 is highest (0.7470); on test it is also best by accuracy, binary F1, and recall. Random Forest has a slightly lower test AUC than GB (0.8132 vs 0.8248). “Best” therefore depends on the stated metric; this report's selected model is based on validation macro F1.

### Held-out test confusion matrices

Each pair is `[[Employable predicted Employable, Employable predicted LessEmployable], [LessEmployable predicted Employable, LessEmployable predicted LessEmployable]]`. The source plot shows (N=597) in each matrix.

| Model | Test confusion matrix |
|---|---|
| Logistic Regression | `[[212, 134], [109, 142]]` |
| Decision Tree | `[[277, 69], [111, 140]]` |
| Random Forest | `[[274, 72], [82, 169]]` |
| SVM | `[[308, 38], [106, 145]]` |
| Gradient Boosting | `[[319, 27], [111, 140]]` |
| KNN | `[[256, 90], [82, 169]]` |

Gradient Boosting finds 319 of 346 Employable examples in this test split (recall 0.922) but labels 111 of 251 LessEmployable examples as Employable. That illustrates why precision and class-specific errors matter alongside accuracy.

## 6. Text retrieval and rule-based scoring

`TfidfVectorizer` is configured with 1- and 2-grams, sublinear term frequency, English stop words, and at most 10,000 features. `run_text_pipeline()` fits it on question and reference-answer text from **both train and eval Parquet files**. This uses the eval text to define IDF/vocabulary, though no candidate-quality labels are available and the eval split is not used to report supervised text-model accuracy.

For candidate answer (c), reference answer (r), and question (q):

\[
Rel=\cos(TFIDF(c),TFIDF(r)),\quad
Align=\cos(TFIDF(c),TFIDF(q))
\]

TF-IDF cosine similarity is

\[
\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}
\]

Jaccard overlap (J) uses content-word sets after cleanup and a fixed stopword list:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Structure (S) is a hand rule based on answer length and presence of up to ten connector strings. It awards a length component of 0.5 for 25–150 words, 0.4 above 150, and a proportional score below 25; connector component is 0.25 per detected connector, capped at 0.5. Responses under five words receive 0.2. The function does not parse grammar, sentence count, or logical validity despite comments that describe those ideas.

The exact composite is:

\[
Q=0.50Rel+0.20Align+0.15J+0.15S
\]

`Q` is clipped to `[0,1]`. UI/API presentation tiers are based on fixed thresholds (0.35 and 0.15). No human-scored candidate answers are used to learn or calibrate these weights; therefore (Q) is a heuristic assessment score, not a validated probability or supervised classifier output.

## 7. Audio feature calculations

`CandidateAudioAnalyzer(target_sr=16000, n_mfcc=13)` uses `librosa.load(..., sr=16000, mono=True)` for file input.

- **Duration:** waveform sample count divided by sample rate (via `librosa.get_duration`).
- **RMS energy:** mean of `librosa.feature.rms(y)`; describes signal amplitude/energy, not confidence or personality.
- **Non-silent intervals / pause ratio:** `librosa.effects.split(y, top_db=25)` defines intervals relative to clip level. Speaking duration is summed interval samples divided by sample rate. Pause ratio is `(duration - speaking_duration) / duration`, clipped to 0–1. It depends on threshold/recording noise and can classify quiet speech as silence.
- **F0/pitch:** `librosa.yin` estimates F0 within 65–400 Hz; code keeps values in `(70,380)` Hz and calculates mean, standard deviation, minimum, maximum. The extractor substitutes default pitch values on estimation failure; the new report changes pitch fields to unavailable for very low RMS audio, but other low-quality unvoiced signals may still yield unreliable estimates.
- **MFCC:** `librosa.feature.mfcc(..., n_mfcc=13)` and temporal mean per coefficient. These are spectral summaries, not direct pronunciation-accuracy measurements.
- **Speaking rate:** only in new `backend/main.py`; `corrected transcript word count / (duration_sec / 60)`. This is approximate because transcript words may not exactly match spoken words and denominator includes pauses.
- **Filler count:** fixed regex counts `um`, `uh`, `erm`, `ah`, `like`, `you know`, `i mean` in the corrected transcript. It can miss variants/languages and count normal uses of “like”.

The extractor also computes fixed-formula estimated fluency/prosody scores, but the new report route removes these fields. No trained acoustic classifier or regression model exists in `models/`.
