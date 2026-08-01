"""
Module 2A — train_cardio_model.py
Train a RandomForestClassifier on the Kaggle Cardiovascular Disease dataset.

Dataset : data/cardio_train.csv  (semicolon-separated, 70,000 rows)
Target  : cardio (0 = no CVD, 1 = CVD)
Output  : models/cardio_model.pkl

Preprocessing steps (per spec):
  1. Convert age from days → years  (age_years = age // 365)
  2. Derive BMI from height (cm) and weight (kg)
  3. Drop rows where diastolic > systolic (physiologically invalid)
  4. Filter to plausible ranges for height, weight, ap_hi, ap_lo
  5. Expect ~1,400-1,500 rows removed
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Suppress the NumPy 1.x/2.x compatibility warnings from numexpr/bottleneck
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "cardio_train.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cardio_model.pkl")

FEATURES = [
    "age_years", "gender", "bmi",
    "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active",
]
TARGET = "cardio"

RANDOM_STATE = 42


def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load the cardio dataset and apply all required cleaning steps.

    Steps:
        1. Read semicolon-separated CSV.
        2. Convert age (days) → age_years.
        3. Derive BMI.
        4. Drop rows where ap_lo > ap_hi.
        5. Filter to plausible vital-sign ranges.

    Returns:
        Cleaned DataFrame with derived columns.
    """
    print("Loading dataset from:", path)
    df = pd.read_csv(path, sep=";")
    n_original = len(df)
    print(f"  Raw rows: {n_original:,}")

    # --- Step 1: age in days → years ---
    df["age_years"] = df["age"] // 365
    print(f"  Age converted: days -> years (range {df['age_years'].min()}-"
          f"{df['age_years'].max()})")

    # --- Step 2: derive BMI ---
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    df["bmi"] = df["bmi"].round(1)

    # --- Step 3: drop physiologically invalid BP rows ---
    mask_bp_invalid = df["ap_lo"] > df["ap_hi"]
    n_bp_invalid = mask_bp_invalid.sum()
    df = df[~mask_bp_invalid].copy()
    print(f"  Dropped {n_bp_invalid:,} rows where ap_lo > ap_hi")

    # --- Step 4: filter to plausible ranges ---
    plausible = (
        df["height"].between(130, 210)
        & df["weight"].between(30, 200)
        & df["ap_hi"].between(80, 220)
        & df["ap_lo"].between(50, 140)
    )
    n_before_range = len(df)
    df = df[plausible].copy()
    n_range_dropped = n_before_range - len(df)
    print(f"  Dropped {n_range_dropped:,} rows outside plausible vital ranges")

    total_dropped = n_original - len(df)
    print(f"  Total rows removed: {total_dropped:,} / {n_original:,}")
    print(f"  Clean rows remaining: {len(df):,}")

    return df


def train_model(df: pd.DataFrame) -> None:
    """Train a RandomForest, evaluate, and save the model."""

    X = df[FEATURES]
    y = df[TARGET]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"\nTrain set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"Positive-class rate  train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    # Build model per spec
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("\nTraining RandomForestClassifier …")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print("CARDIO MODEL — EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1-score : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  ROC-AUC  : {auc:.4f}  ({auc*100:.1f}%)")
    print()
    print(classification_report(y_test, y_pred, target_names=["No CVD", "CVD"]))

    # Sanity-check against known benchmarks
    print("-" * 50)
    print("Benchmark check (expected: acc~73.6%, F1~71.8%, AUC~80.3%):")
    for name, val, expected in [("Accuracy", acc, 0.736),
                                 ("F1",       f1,  0.718),
                                 ("ROC-AUC",  auc, 0.803)]:
        delta = abs(val - expected) * 100
        status = "[OK]" if delta < 3.0 else "[CHECK]"
        print(f"  {name:10s}: {val*100:.1f}% (expected ~{expected*100:.1f}%)  "
              f"D={delta:.1f}pp  {status}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return acc, f1, auc


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_preprocess(DATA_PATH)
    train_model(df)
