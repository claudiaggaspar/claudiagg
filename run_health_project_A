#!/usr/bin/env python3
# Project A — Rule-based target + Decision Tree
# Portable for GitHub/Codespaces:
#   - Reads CSV from data/raw/health_lifestyle.csv OR --csv/HEALTH_CSV
#   - Creates folders reports/ and reports/figures/
#   - Saves figures and a summary JSON

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

# ----------------------- CLI / ENV CONFIG -----------------------
parser = argparse.ArgumentParser(description="Project A — rule-based target + decision tree")
parser.add_argument("--csv", type=str, default=None, help="Path to source CSV")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_depth", type=int, default=5, help="DecisionTree max_depth")
args = parser.parse_args()

SEED = args.seed
MAX_DEPTH = args.max_depth
np.random.seed(SEED)

# Repo root = folder that contains this script
ROOT = Path(__file__).resolve().parent

# Standard project folders
DATA_RAW = ROOT / "data" / "raw" / "health_lifestyle.csv"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
for d in (DATA_RAW.parent, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# CSV source resolution order: --csv > HEALTH_CSV > data/raw/health_lifestyle.csv
env_csv = os.getenv("HEALTH_CSV")
CSV_SOURCE = Path(args.csv or env_csv) if (args.csv or env_csv) else None

# If a source was provided, copy into data/raw for reproducibility
if CSV_SOURCE and CSV_SOURCE.exists():
    DATA_RAW.write_bytes(CSV_SOURCE.read_bytes())

if not DATA_RAW.exists():
    raise FileNotFoundError(
        f"CSV not found. Provide --csv PATH or set HEALTH_CSV, or place file at {DATA_RAW}"
    )

# ----------------------- THRESHOLDS / TOKENS -----------------------
# Numeric thresholds; if a column is missing or not numeric, fallback uses quantiles
THRESHOLDS = {
    "bmi": 30,                 # obesity
    "blood_pressure": 140,     # systolic mmHg assumed
    "cholesterol": 240,        # mg/dL
    "glucose": 126,            # fasting mg/dL
    "sleep_hours_low": 6,      # short sleep
    "daily_steps_low": 5000    # sedentary
}
# Category tokens considered risky when present
RISK_TOKENS = {
    "smoking_level": ["heavy", "high", "daily", "smoker"],
    "alcohol_consumption": ["heavy", "high", "binge"]
}

# ----------------------- LOAD -----------------------
df = pd.read_csv(DATA_RAW)

# ----------------------- TARGET BUILD (RULE-BASED) -----------------------
def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def flag_numeric(col: str, thr, direction: str = ">=") -> pd.Series:
    """Return boolean flag for numeric threshold. If column missing or all NaN, use 0.85/0.15 quantile fallback."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    x = to_numeric(df[col])
    if x.isna().all():
        q = 0.85 if direction in (">", ">=") else 0.15
        thr = x.quantile(q)
    if direction == ">=":
        return (x >= thr).fillna(False)
    if direction == ">":
        return (x > thr).fillna(False)
    if direction == "<=":
        return (x <= thr).fillna(False)
    if direction == "<":
        return (x < thr).fillna(False)
    raise ValueError("Invalid direction")

def flag_categorical_contains(col: str, tokens) -> pd.Series:
    """Return boolean flag if any token appears in the string category (case-insensitive)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for t in tokens:
        mask = mask | s.str.contains(str(t).lower(), na=False)
    return mask

# Numeric risk flags
f_bmi        = flag_numeric("bmi", THRESHOLDS["bmi"], ">=")
f_bp         = flag_numeric("blood_pressure", THRESHOLDS["blood_pressure"], ">=")
f_chol       = flag_numeric("cholesterol", THRESHOLDS["cholesterol"], ">=")
f_glucose    = flag_numeric("glucose", THRESHOLDS["glucose"], ">=")
f_sleep_low  = flag_numeric("sleep_hours", THRESHOLDS["sleep_hours_low"], "<")
f_steps_low  = flag_numeric("daily_steps", THRESHOLDS["daily_steps_low"], "<")

# Categorical risk flags
f_smoke  = flag_categorical_contains("smoking_level", RISK_TOKENS["smoking_level"])
f_alc    = flag_categorical_contains("alcohol_consumption", RISK_TOKENS["alcohol_consumption"])

# Optional environment/genetic flags if present — numeric uses quantile fallback
f_env    = flag_numeric("environmental_risk_score", np.nan, ">=")
f_gene   = flag_categorical_contains("gene_marker_flag", ["1", "true", "yes", "positive"])

# Aggregate risk score and binary target
risk_score = (
    f_bmi.astype(int) + f_bp.astype(int) + f_chol.astype(int) + f_glucose.astype(int) +
    f_sleep_low.astype(int) + f_steps_low.astype(int) + f_smoke.astype(int) +
    f_alc.astype(int) + f_env.astype(int) + f_gene.astype(int)
)
df["target"] = (risk_score >= 2).astype("int64")

# ----------------------- BASIC SANITATION -----------------------
num_all = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_all] = df[num_all].replace([np.inf, -np.inf], np.nan)

# ----------------------- QUICK EDA ARTIFACTS -----------------------
REPORT_JSON = REPORTS / "eda_tree_summary.json"
REPORTS.mkdir(parents=True, exist_ok=True)

# dtypes and missing
df.isna().sum().sort_values(ascending=False).head(30).to_csv(REPORTS / "missing_top30.csv")
dtypes = df.dtypes.astype(str).reset_index()
dtypes.columns = ["column", "dtype"]
dtypes.to_csv(REPORTS / "dtypes.csv", index=False)

num_cols_all = [c for c in num_all if c != "target"]
if num_cols_all:
    df[num_cols_all].hist(bins=20, figsize=(12, 10))
    plt.suptitle("Numeric distributions", y=1.02)
    plt.tight_layout(); plt.savefig(FIG_DIR / "hist_numeric.png", dpi=150); plt.close()

    if len(num_cols_all) > 1:
        corr = df[num_cols_all].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, aspect="auto"); plt.colorbar()
        plt.title("Correlation matrix (numeric)")
        plt.xticks(range(len(num_cols_all)), num_cols_all, rotation=90, fontsize=7)
        plt.yticks(range(len(num_cols_all)), num_cols_all, fontsize=7)
        plt.tight_layout(); plt.savefig(FIG_DIR / "corr_numeric.png", dpi=150); plt.close()

# ----------------------- TRAIN / TEST -----------------------
X = df.drop(columns=["target"])
y = df["target"]

cls_counts = y.value_counts()
if len(cls_counts) < 2 or cls_counts.min() < 2:
    raise ValueError("Target still single-class. Adjust rules or thresholds to produce both classes.")

def safe_split(X, y, seed=SEED):
    for test_size in (0.2, 0.15, 0.1, 0.05):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        if y_tr.nunique() == 2 and y_te.nunique() == 2:
            return X_tr, X_te, y_tr, y_te, test_size
    raise ValueError("Could not split with both classes in train and test. Loosen rules or add data.")

X_train, X_test, y_train, y_test, used_ts = safe_split(X, y)

# ----------------------- PREPROCESSING -----------------------
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# ----------------------- MODEL -----------------------
clf = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    class_weight="balanced",
    random_state=SEED
)
pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])
pipe.fit(X_train, y_train)

# ----------------------- EVALUATION -----------------------
model = pipe.named_steps["model"]
has_proba = hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) == 2

if has_proba:
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)
else:
    pred_test  = pipe.predict(X_test).astype(int)
    proba_test = pred_test.astype(float)

# Text report
report = classification_report(y_test, pred_test, digits=3, zero_division=0)
with open(REPORTS / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# Confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test)).plot()
plt.title("Confusion Matrix — Test")
plt.tight_layout(); plt.savefig(FIG_DIR / "cm_test.png", dpi=150); plt.close()

# ROC / PR
if y_test.nunique() == 2 and has_proba:
    roc = roc_auc_score(y_test, proba_test)
    RocCurveDisplay.from_predictions(y_test, proba_test)
    plt.title(f"ROC Curve — Test (AUC={roc:.3f})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_test.png", dpi=150); plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, proba_test)
    plt.title("Precision-Recall — Test")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pr_test.png", dpi=150); plt.close()

# Tree viz
transformed_names = pipe.named_steps["prep"].get_feature_names_out()
plt.figure(figsize=(20, 10))
plot_tree(
    pipe.named_steps["model"],
    filled=True,
    max_depth=MAX_DEPTH,
    feature_names=transformed_names,
    class_names=["healthy", "unhealthy"]
)
plt.title("Decision Tree (after preprocessing)")
plt.tight_layout(); plt.savefig(FIG_DIR / "tree_structure.png", dpi=150); plt.close()

# ----------------------- SUMMARY JSON -----------------------
summary = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "target_balance": y.value_counts(normalize=True).to_dict(),
    "numeric_cols": num_cols,
    "categorical_cols": cat_cols,
    "test_size_used": used_ts,
    "max_depth": MAX_DEPTH,
    "seed": SEED
}
with open(REPORTS / "eda_tree_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done.")
print(f"Figures: {FIG_DIR}")
print(f"Summary: {REPORTS / 'eda_tree_summary.json'}")
