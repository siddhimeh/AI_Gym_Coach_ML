"""
ML Classifier Training Module
Trains explainable Decision Tree and Random Forest classifiers on biomechanical angle features.
Uses grouped/temporal train-test splitting to prevent intra-clip data leakage.
Exports models/squat_classifier.joblib and training metrics.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_CSV = os.path.join(ROOT_DIR, "data", "processed", "squat_dataset.csv")
MODEL_OUT = os.path.join(ROOT_DIR, "models", "squat_classifier.joblib")


def prepare_train_test_split(df: pd.DataFrame):
    """
    Splits data by repetition/clip phase to prevent adjacent-frame data leakage.
    Ensures that test frames come from distinct repetition segments.
    """
    df = df.copy()

    # Create segment identifiers for train-test split
    # In squat_sample_good: Rep 1 (frames 1-110) vs Rep 2 (frames 111-213)
    train_mask = []
    for _, row in df.iterrows():
        clip = row["clip"]
        f = row["frame"]
        if clip == "squat_sample_good":
            # Rep 1 -> Train, Rep 2 -> Test (Strict temporal split)
            train_mask.append(f <= 110)
        elif clip == "squat_shallow_rep":
            # First 60% of rep -> Train, final 40% -> Test
            train_mask.append(f <= 70)
        else:
            # Augmented data: 70% train, 30% test
            train_mask.append(f <= 42)

    df["is_train"] = train_mask
    train_df = df[df["is_train"]].copy()
    test_df = df[~df["is_train"]].copy()

    features = ["knee_angle", "hip_angle", "back_lean"]
    X_train = train_df[features].values
    y_train = train_df["label"].values
    X_test = test_df[features].values
    y_test = test_df["label"].values

    return X_train, X_test, y_train, y_test, features, train_df, test_df


def train_models():
    print("=" * 65)
    print("PHASE 11: TRAINING ML SQUAT CLASSIFIERS")
    print("=" * 65)

    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Dataset not found: {DATA_CSV}")
        return

    df = pd.read_csv(DATA_CSV)
    X_train, X_test, y_train, y_test, feature_names, train_df, test_df = prepare_train_test_split(df)

    print(f"Dataset split by clip/repetition:")
    print(f"  - Training samples:   {len(X_train)} frames")
    print(f"  - Test holdout:       {len(X_test)} frames")
    print(f"  - Feature dimensions: {feature_names}")
    print("-" * 65)

    # 1. Model A: Explainable Decision Tree (Depth 4)
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)

    # 2. Model B: Random Forest Classifier (50 Trees)
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"MODEL PERFORMANCE ON HELD-OUT TEST REPETITIONS:")
    print(f"  1. Decision Tree Accuracy:  {dt_acc * 100:.2f}%")
    print(f"  2. Random Forest Accuracy:  {rf_acc * 100:.2f}%")
    print("-" * 65)

    # Feature Importance Analysis
    print("RANDOM FOREST FEATURE IMPORTANCE:")
    for feat, imp in zip(feature_names, rf_model.feature_importances_):
        print(f"  - {feat:<15}: {imp * 100:.1f}%")
    print("-" * 65)

    # Decision Tree Rules (Completely explainable for Viva)
    print("DECISION TREE EXPLAINABLE RULES (First 2 Levels):")
    tree_rules = export_text(dt_model, feature_names=feature_names, max_depth=2)
    print(tree_rules)
    print("-" * 65)

    # Save the selected best model (Random Forest) along with metadata
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    bundle = {
        "model": rf_model,
        "dt_model": dt_model,
        "features": feature_names,
        "classes": list(rf_model.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": float(rf_acc),
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"[SAVED] Trained model bundle saved to: {MODEL_OUT}")
    print("=" * 65)
    return bundle


if __name__ == "__main__":
    train_models()
