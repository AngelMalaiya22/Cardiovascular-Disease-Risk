"""
Cardiovascular Disease Risk Assessment & Prediction Model - Training Pipeline
=============================================================================
Dataset : data/heart.csv  (303 rows, 13 clinical features + binary target)
Models  : Logistic Regression, SVM, KNN, Decision Tree, Random Forest

Pipeline
--------
 1. Load data
 2. Inspection & cleaning (missing values, duplicates, invalid category codes, target coding)
 3. Exploratory Data Analysis (EDA)            -> plots saved to ./outputs
 4. Feature / target separation
 5. Stratified train-test split
 6. Preprocessing (scaling + one-hot encoding) -> fitted on TRAIN data only (no leakage)
 7. Baseline models (default hyperparameters)  -> test accuracy (+ 5-fold CV accuracy)
 8. Hyperparameter tuning with Optuna          -> objective = 5-fold CV accuracy on TRAIN data only
 9. Retrain with the best parameters           -> test accuracy
10. Comparison graph (before vs after tuning) and conclusion
11. Export the best model for the website      -> ./model/heart_model.joblib + model_metadata.json

Usage
-----
    pip install -r requirements-train.txt
    python train_model.py                       # uses data/heart.csv, 50 Optuna trials per model
    python train_model.py --trials 100 --no-show
"""

import argparse
import json
import os
import platform
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.join("data", "heart.csv"), help="path to heart.csv")
parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model")
parser.add_argument("--out", default="outputs", help="folder for plots / results table")
parser.add_argument("--model-dir", default="model", help="folder for the exported model")
parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
args = parser.parse_args()

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_TRIALS = args.trials
OUT_DIR = args.out
MODEL_DIR = args.model_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "target"
# In this copy of the dataset the raw label 1 means "NO heart disease" (see section 2).
# Set to False only if you use a version of the file where 1 already means "disease".
INVERT_TARGET = True

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]   # continuous -> scale
NOMINAL_COLS = ["cp", "restecg", "slope", "thal"]                   # categories -> one-hot
PASSTHROUGH_COLS = ["sex", "fbs", "exang", "ca"]                    # binary / small counts
FEATURE_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_DESC = {
    "age": "Age", "sex": "Sex (1=male)", "cp": "Chest pain type",
    "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting sugar > 120",
    "restecg": "Resting ECG", "thalach": "Max heart rate", "exang": "Exercise angina",
    "oldpeak": "ST depression", "slope": "ST slope", "ca": "Major vessels (0-3)",
    "thal": "Thalassemia", "target": "Heart disease",
}


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   [saved] {path}")
    if not args.no_show:
        plt.show()
    plt.close()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
section("1. LOAD DATA")
df = pd.read_csv(args.data)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())

# ----------------------------------------------------------------------------
# 2. DATA INSPECTION & CLEANING
# ----------------------------------------------------------------------------
section("2. DATA INSPECTION & CLEANING")
print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

n_dup = df.duplicated().sum()
print(f"\nDuplicate rows: {n_dup}")
if n_dup > 0:
    # Removed BEFORE splitting so the same row cannot sit in both train and test (leakage).
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates dropped -> new shape: {df.shape}")

print("\nStatistical summary:")
print(df.describe().T.round(2))

# Invalid category codes (known quirks of the UCI Cleveland heart dataset)
print("\nData-quality notes:")
print(f"  - 'ca' has {(df['ca'] == 4).sum()} rows with value 4 (valid range is 0-3)")
print(f"  - 'thal' has {(df['thal'] == 0).sum()} rows with value 0 (valid codes are 1-3)")
df.loc[df["ca"] == 4, "ca"] = df["ca"].mode()[0]
df.loc[df["thal"] == 0, "thal"] = df["thal"].mode()[0]
print("  -> replaced with the column mode (very few rows, so imputing beats dropping).")

# Target coding check --------------------------------------------------------
# Evidence that raw target == 1 means "no disease" in this file:
#  * class sizes (164 vs 138) match the original Cleveland data: 164 healthy / 139 diseased
#  * established risk markers are LOWER when raw target == 1, and max heart rate is HIGHER
print("\nTarget coding check (mean of risk markers by RAW target value):")
print(df.groupby(TARGET)[["exang", "ca", "oldpeak", "thalach"]].mean().round(2))
print("Raw target counts:", df[TARGET].value_counts().sort_index().to_dict())
if INVERT_TARGET:
    df[TARGET] = 1 - df[TARGET]
    print("-> target recoded so that 1 = heart disease present, 0 = no heart disease.")

# Outlier inspection (IQR rule) - reported only, NOT removed (small dataset, plausible values)
print("\nOutliers by IQR rule (reported, not removed):")
for col in NUMERIC_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col:9s}: {n_out}")

# ----------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
section("3. EXPLORATORY DATA ANALYSIS")

counts = df[TARGET].value_counts().sort_index()
print("Target distribution (after recoding):")
print(counts.rename({0: "No disease (0)", 1: "Disease (1)"}))
print((counts / len(df) * 100).round(1).astype(str) + " %")

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=TARGET, data=df, palette=["#4C72B0", "#C44E52"], ax=ax)
ax.set_xticklabels(["No disease", "Disease"])
ax.set_title("Target class distribution")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
save_fig("eda_1_target_distribution.png")

fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(20, 3.5))
for ax, col in zip(axes, NUMERIC_COLS):
    sns.histplot(df, x=col, hue=TARGET, kde=True, bins=20, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title(FEATURE_DESC[col])
save_fig("eda_2_numeric_distributions.png")

fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(20, 3.5))
for ax, col in zip(axes, NUMERIC_COLS):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title(FEATURE_DESC[col])
save_fig("eda_3_numeric_boxplots.png")

cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes.ravel(), cat_cols):
    sns.countplot(x=col, hue=TARGET, data=df, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title(FEATURE_DESC[col])
save_fig("eda_4_categorical_vs_target.png")

plt.figure(figsize=(11, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation matrix")
save_fig("eda_5_correlation_heatmap.png")

print("\nCorrelation of each feature with the target:")
print(df.corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

# ----------------------------------------------------------------------------
# 4. FEATURES / TARGET
# ----------------------------------------------------------------------------
section("4. FEATURES / TARGET")
X = df[FEATURE_ORDER]
y = df[TARGET]
print(f"X shape: {X.shape} | y shape: {y.shape}")

# ----------------------------------------------------------------------------
# 5. TRAIN-TEST SPLIT (stratified)
# ----------------------------------------------------------------------------
section("5. TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print("Train class ratio:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Test  class ratio:", y_test.value_counts(normalize=True).round(3).to_dict())

# ----------------------------------------------------------------------------
# 6. PREPROCESSING
# ----------------------------------------------------------------------------
# The ColumnTransformer sits INSIDE each model Pipeline, so the scaler/encoder are
# fitted only on the training part of every CV fold -> no information leakage.
section("6. PREPROCESSING (StandardScaler + OneHotEncoder inside a Pipeline)")


def make_pipeline(model):
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS),
        ]
    )
    return Pipeline([("prep", prep), ("model", model)])


print(f"Scaled (StandardScaler): {NUMERIC_COLS}")
print(f"One-hot encoded        : {NOMINAL_COLS}")
print(f"Passed through         : {PASSTHROUGH_COLS}")

# ----------------------------------------------------------------------------
# 7. BASELINE MODELS (default hyperparameters)
# ----------------------------------------------------------------------------
section("7. BASELINE MODELS (no hyperparameter tuning)")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# SVC(probability=True) lets the website show a risk probability (predictions are unchanged).
baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}

results = {}
for name, model in baseline_models.items():
    pipe = make_pipeline(model)
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    results[name] = {"base_test": test_acc, "base_cv": cv_acc}
    print(f"{name:20s} | CV acc: {cv_acc:.4f} | Test acc: {test_acc:.4f}")

# ----------------------------------------------------------------------------
# 8. HYPERPARAMETER TUNING WITH OPTUNA
# ----------------------------------------------------------------------------
# Objective = mean 5-fold CV accuracy on the TRAINING set only. The test set is never
# used for tuning, so the final test accuracy is an honest estimate.
section(f"8. OPTUNA HYPERPARAMETER TUNING ({N_TRIALS} trials per model)")


def build_model(name, trial):
    """Search space for every model; returns an un-fitted estimator."""
    if name == "Logistic Regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="liblinear", max_iter=2000, random_state=RANDOM_STATE,
        )
    if name == "SVM":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        params = dict(C=trial.suggest_float("C", 1e-2, 1e2, log=True), kernel=kernel,
                      probability=True, random_state=RANDOM_STATE)
        if kernel in ("rbf", "poly", "sigmoid"):
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        return SVC(**params)
    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 30),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),  # 1 = Manhattan, 2 = Euclidean
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=RANDOM_STATE,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    raise ValueError(name)


tuned_pipelines = {}
for name in baseline_models:
    def objective(trial, name=name):
        pipe = make_pipeline(build_model(name, trial))
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)

    # 9. Retrain with the best parameters and evaluate on the untouched test set
    best_pipe = make_pipeline(build_model(name, optuna.trial.FixedTrial(study.best_params)))
    best_pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, best_pipe.predict(X_test))

    results[name].update({"tuned_test": test_acc, "tuned_cv": study.best_value,
                          "best_params": study.best_params})
    tuned_pipelines[name] = best_pipe
    print(f"\n{name}")
    print(f"   best params : {study.best_params}")
    print(f"   best CV acc : {study.best_value:.4f} | Test acc after tuning: {test_acc:.4f}")

# ----------------------------------------------------------------------------
# 10. RESULTS TABLE, GRAPH & CONCLUSION
# ----------------------------------------------------------------------------
section("9. RESULTS: BEFORE vs AFTER HYPERPARAMETER TUNING")
res = pd.DataFrame(results).T[["base_cv", "tuned_cv", "base_test", "tuned_test", "best_params"]]
summary = res.drop(columns="best_params").astype(float).round(4)
summary["test_change"] = summary["tuned_test"] - summary["base_test"]
summary.columns = ["CV acc (before)", "CV acc (after)", "Test acc (before)",
                   "Test acc (after)", "Test change"]
print(summary.to_string())
summary.to_csv(os.path.join(OUT_DIR, "model_comparison_results.csv"))

models = list(res.index)
x = np.arange(len(models))
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
panels = [("Test-set accuracy", "base_test", "tuned_test"),
          (f"{CV_FOLDS}-fold CV accuracy (train set)", "base_cv", "tuned_cv")]
for ax, (title, b_col, t_col) in zip(axes, panels):
    before = res[b_col].astype(float).values * 100
    after = res[t_col].astype(float).values * 100
    b = ax.bar(x - w / 2, before, w, label="Before tuning", color="#8DA0CB")
    a = ax.bar(x + w / 2, after, w, label="After Optuna tuning", color="#66C2A5")
    for bars in (b, a):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}%",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, min(before.min(), after.min()) - 10), 105)
    ax.set_title(title)
    ax.legend(loc="upper right")
plt.suptitle("Heart disease prediction: accuracy before vs after hyperparameter tuning",
             fontsize=14, fontweight="bold")
save_fig("model_accuracy_before_vs_after_tuning.png")

# Best model = highest tuned TEST accuracy; ties are broken by tuned CV accuracy.
ranked = res.assign(tuned_test=res["tuned_test"].astype(float),
                    tuned_cv=res["tuned_cv"].astype(float)).sort_values(
    ["tuned_test", "tuned_cv"], ascending=False)
best_name = ranked.index[0]
best_pipe = tuned_pipelines[best_name]
y_pred = best_pipe.predict(X_test)

print(f"\nDetailed report for the best tuned model: {best_name}")
print(classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No disease", "Disease"],
                                        cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix - {best_name} (tuned)")
save_fig("best_model_confusion_matrix.png")

section("10. CONCLUSION")
best_before = float(res.loc[best_name, "base_test"])
best_after = float(res.loc[best_name, "tuned_test"])
best_cv = float(res.loc[best_name, "tuned_cv"])
n_test = len(y_test)

print(f"* Best performing model : {best_name}")
print(f"* Test accuracy before tuning : {best_before * 100:.2f}%")
print(f"* Test accuracy after tuning  : {best_after * 100:.2f}%  "
      f"({(best_after - best_before) * 100:+.2f} points)")
print(f"* Its tuned 5-fold CV accuracy (train set): {best_cv * 100:.2f}%")
print(f"* Best hyperparameters: {results[best_name]['best_params']}")

print("\nAll accuracies (test set):")
print(f"{'Model':22s}{'Before':>10s}{'After':>10s}{'Change':>10s}")
for m in models:
    bt, tt = float(res.loc[m, "base_test"]), float(res.loc[m, "tuned_test"])
    print(f"{m:22s}{bt * 100:9.2f}%{tt * 100:9.2f}%{(tt - bt) * 100:+9.2f}")

print(f"""
Note on interpretation:
  The test set holds only {n_test} patients, so one extra correct prediction moves accuracy
  by about {100 / n_test:.2f} percentage points. Small differences between models, or between
  before/after tuning, can be chance. The 5-fold CV accuracy is the steadier indicator;
  the test accuracy is a single noisy snapshot.
""")

# ----------------------------------------------------------------------------
# 11. EXPORT THE BEST MODEL FOR THE WEBSITE
# ----------------------------------------------------------------------------
section("11. EXPORT MODEL FOR DEPLOYMENT")
# The evaluation above used a train/test split. For the deployed model we refit the SAME
# best configuration on all rows, which is standard practice once model selection is done.
final_pipe = make_pipeline(build_model(best_name, optuna.trial.FixedTrial(results[best_name]["best_params"])))
final_pipe.fit(X, y)
assert list(final_pipe.classes_) == [0, 1], "unexpected class order"

metadata = {
    "project": "Cardiovascular Disease Risk Assessment & Prediction Model",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model_name": best_name,
    "best_params": results[best_name]["best_params"],
    "metrics": {
        "test_accuracy_before_tuning": round(best_before, 4),
        "test_accuracy_after_tuning": round(best_after, 4),
        "cv_accuracy_after_tuning": round(best_cv, 4),
        "all_models": {m: {k: (round(float(v), 4) if k != "best_params" else v)
                           for k, v in results[m].items()} for m in models},
    },
    "data": {"rows_used": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
             "positive_class": "1 = heart disease present", "target_recoded": INVERT_TARGET},
    "feature_order": FEATURE_ORDER,
    "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC_COLS},
    "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__,
                 "joblib": joblib.__version__},
}

model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
joblib.dump(final_pipe, model_path)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"[saved] {model_path}")
print(f"[saved] {meta_path}")
print("\nNext step: run the website with   python app.py")
