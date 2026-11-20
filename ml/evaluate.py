"""
Evaluation & Benchmarking Module
Directly compares Rule-Based Form Detection with Machine Learning Classification.
Computes Accuracy, Precision, Recall, F1-Score, and plots side-by-side Confusion Matrices.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from ml.train_classifier import prepare_train_test_split, MODEL_OUT, DATA_CSV

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


def predict_rule_based(knee_angle: float, back_lean: float) -> str:
    """
    Explainable, deterministic rule-based classifier based on Phase 4 kinematic analysis.
    """
    if knee_angle >= 155.0:
        return "standing"
    elif back_lean > 38.0:
        return "back_leaning_forward"
    elif knee_angle <= 95.0:
        return "good"
    else:
        return "too_shallow"


def evaluate_and_compare():
    print("=" * 70)
    print("PHASE 11: RULE-BASED VS. ML CLASSIFIER BENCHMARK EVALUATION")
    print("=" * 70)

    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Dataset not found: {DATA_CSV}")
        return
    if not os.path.exists(MODEL_OUT):
        from ml.train_classifier import train_models
        train_models()

    df = pd.read_csv(DATA_CSV)
    X_train, X_test, y_train, y_test, feature_names, train_df, test_df = prepare_train_test_split(df)
    bundle = joblib.load(MODEL_OUT)
    ml_model = bundle["model"]

    # 1. Predictions on Held-Out Test Set
    ml_preds = ml_model.predict(X_test)

    # 2. Rule-Based Predictions on Same Test Set
    rule_preds = [
        predict_rule_based(row["knee_angle"], row["back_lean"])
        for _, row in test_df.iterrows()
    ]

    labels = ["standing", "good", "too_shallow", "back_leaning_forward"]
    label_names = ["Standing", "Good Rep", "Too Shallow", "Forward Lean"]

    # Metric Calculations
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (Macro)": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "Recall (Macro)": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "F1-Score (Macro)": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "Precision (Weighted)": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            "Recall (Weighted)": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            "F1-Score (Weighted)": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        }

    rule_metrics = get_metrics(y_test, rule_preds)
    ml_metrics = get_metrics(y_test, ml_preds)

    # 3. Print Comparison Table
    print(f"{'Metric':<24} | {'Rule-Based Detection':<22} | {'ML (Random Forest)':<20}")
    print("-" * 72)
    for m in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "F1-Score (Weighted)"]:
        r_val = rule_metrics[m] * 100
        m_val = ml_metrics[m] * 100
        print(f"{m:<24} | {r_val:>19.2f}% | {m_val:>17.2f}%")
    print("-" * 72)

    # 4. Generate Side-by-Side Confusion Matrix Plots
    cm_rule = confusion_matrix(y_test, rule_preds, labels=labels)
    cm_ml = confusion_matrix(y_test, ml_preds, labels=labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(wspace=0.35)

    def plot_cm(ax, cm, title, cmap):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(label_names, fontsize=9)
        ax.set_ylabel("True Ground Truth Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

        # Print cell numbers
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(j, i, f"{val}", ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontweight="bold", fontsize=11)

    plot_cm(ax1, cm_rule, "Rule-Based Form Detection\n(Deterministic Kinematics)", plt.cm.Blues)
    plot_cm(ax2, cm_ml, "ML Random Forest Classifier\n(Angle Feature Vector)", plt.cm.Greens)

    cm_path = os.path.join(PROCESSED_DIR, "confusion_matrix_comparison.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] Side-by-side Confusion Matrix comparison saved to:")
    print(f"        {cm_path}")

    # 5. College Viva Takeaways
    print("-" * 72)
    print("CRITICAL FINDINGS FOR COLLEGE VIVA:")
    print("  1. Rule-Based Strengths:")
    print("     - 100% deterministic, explainable, and adheres to biomechanical anatomy.")
    print("     - Zero training time and instant inference on CPU without model weights.")
    print("  2. ML Classifier Strengths:")
    print("     - Captures multi-joint interactions (e.g. combined hip hinge + knee flexion).")
    print("     - Provides probabilistic confidence estimates rather than hard binary cutoffs.")
    print("  3. Dataset Limitations (Academic Honesty):")
    print(f"     - Evaluated on N={len(X_test)} held-out frames split by repetition/clip.")
    print("     - Real exercise datasets inherently lack severe injury-inducing form.")
    print("=" * 70)

    return {
        "rule_metrics": rule_metrics,
        "ml_metrics": ml_metrics,
        "cm_path": cm_path,
        "test_count": len(X_test),
    }


if __name__ == "__main__":
    evaluate_and_compare()
