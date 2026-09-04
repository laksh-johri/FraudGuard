"""
Train an XGBoost fraud classifier.

Data source (auto-detected, in priority order):
  1. Kaggle "Credit Card Fraud Detection" dataset at model/creditcard.csv, if present.
  2. Synthetic fallback via sklearn.datasets.make_classification (imbalanced ~0.5% fraud)
     so training always works even without the Kaggle download.

Outputs:
  model/artifacts/fraud_model.pkl      (joblib-dumped XGBClassifier)
  model/artifacts/feature_names.json   (exact ordered feature list the model expects)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib

HERE = Path(__file__).parent
ARTIFACTS_DIR = HERE / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
CSV_PATH = HERE / "creditcard.csv"


def load_data():
    if CSV_PATH.exists():
        print(f"[data] Found {CSV_PATH.name} -> using Kaggle Credit Card Fraud dataset (REAL mode)")
        df = pd.read_csv(CSV_PATH)
        feature_names = [c for c in df.columns if c not in ("Time", "Class")]
        X = df[feature_names]
        y = df["Class"]
        return X, y, feature_names

    print("[data] creditcard.csv not found -> generating synthetic imbalanced dataset (SYNTHETIC mode)")
    n_samples = 20000
    n_features = 15  # f0..f14
    X_arr, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        weights=[0.995, 0.005],  # ~0.5% positive
        flip_y=0.001,
        class_sep=1.2,
        random_state=42,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X_arr, columns=feature_names)

    # Synthetic transaction amount, loosely correlated with fraud label for realism
    rng = np.random.default_rng(42)
    base_amount = rng.gamma(shape=2.0, scale=50.0, size=n_samples)
    fraud_bump = y * rng.gamma(shape=3.0, scale=80.0, size=n_samples)
    df["Amount"] = np.round(base_amount + fraud_bump, 2)
    feature_names.append("Amount")

    X = df[feature_names]
    y = pd.Series(y, name="Class")
    return X, y, feature_names


def main():
    X, y, feature_names = load_data()
    print(f"[data] {len(X)} rows, {len(feature_names)} features, "
          f"{y.sum()} positive ({100 * y.mean():.3f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"[train] scale_pos_weight = {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # NOTE: accuracy is a misleading metric here — with ~0.5% fraud, a model that
    # predicts "not fraud" for everything would still score >99% accuracy.
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test set metrics ===")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}  (average precision)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(cm)

    # Latency benchmark: single-row predict_proba, the real hot-path call
    sample_row = X_test.iloc[[0]]
    n_iters = 200
    # warm up
    for _ in range(10):
        model.predict_proba(sample_row)
    start = time.perf_counter()
    for _ in range(n_iters):
        model.predict_proba(sample_row)
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters
    print(f"\n[latency] single-row predict_proba: {elapsed_ms:.3f} ms/call (avg over {n_iters} calls)")

    model_path = ARTIFACTS_DIR / "fraud_model.pkl"
    features_path = ARTIFACTS_DIR / "feature_names.json"
    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    # Per-feature mean/std/correlation-with-label, used by the backend as a
    # lightweight fallback explanation (z-score * correlation sign) when the
    # SHAP explainer is unavailable or too slow to run per-request.
    stats_path = ARTIFACTS_DIR / "feature_stats.json"
    feature_stats = {}
    for name in feature_names:
        col = X_train[name]
        feature_stats[name] = {
            "mean": float(col.mean()),
            "std": float(col.std() or 1.0),
            "corr_with_fraud": float(np.corrcoef(col, y_train)[0, 1]),
        }
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)

    print(f"\n[artifacts] saved {model_path}")
    print(f"[artifacts] saved {features_path}")
    print(f"[artifacts] saved {stats_path}")


if __name__ == "__main__":
    main()
