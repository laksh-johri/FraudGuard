"""
Generate an explainable-AI feature-importance visual from the trained fraud model.

Tries a SHAP summary (beeswarm) plot first, since it shows both feature
importance and the direction/magnitude of each feature's effect on individual
predictions. Falls back to XGBoost's built-in feature_importances_ bar chart
if SHAP errors out or takes too long (e.g. no SHAP install, or slow on a large
sample), so this always produces something useful for the pitch deck.

Output: model/artifacts/shap_feature_importance.png (or
        model/artifacts/feature_importance_fallback.png on fallback)
"""

import json
import signal
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ARTIFACTS_DIR = HERE / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
CSV_PATH = HERE / "creditcard.csv"

SHAP_TIMEOUT_SECONDS = 60
SHAP_SAMPLE_SIZE = 500


def load_model_and_features():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        print(f"[error] Model artifacts not found in {ARTIFACTS_DIR}. Run `python model/train.py` first.")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text())
    return model, feature_names


def load_background_data(feature_names):
    """A sample of rows to explain. Uses the real Kaggle CSV if present, else
    regenerates the same synthetic distribution train.py falls back to."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        X = df[feature_names]
    else:
        from sklearn.datasets import make_classification

        n_features = len(feature_names) - 1  # exclude Amount
        X_arr, y = make_classification(
            n_samples=2000,
            n_features=n_features,
            n_informative=8,
            n_redundant=3,
            n_classes=2,
            weights=[0.995, 0.005],
            flip_y=0.001,
            class_sep=1.2,
            random_state=42,
        )
        df = pd.DataFrame(X_arr, columns=feature_names[:-1])
        rng = np.random.default_rng(42)
        base_amount = rng.gamma(shape=2.0, scale=50.0, size=len(df))
        fraud_bump = y * rng.gamma(shape=3.0, scale=80.0, size=len(df))
        df["Amount"] = np.round(base_amount + fraud_bump, 2)
        X = df[feature_names]

    if len(X) > SHAP_SAMPLE_SIZE:
        X = X.sample(n=SHAP_SAMPLE_SIZE, random_state=42)
    return X.reset_index(drop=True)


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def try_shap_plot(model, X):
    out_path = ARTIFACTS_DIR / "shap_feature_importance.png"

    # SIGALRM is POSIX-only; on Windows just run without a hard timeout guard.
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(SHAP_TIMEOUT_SECONDS)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Feature Importance — Fraud Model")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path
    finally:
        if has_alarm:
            signal.alarm(0)


def fallback_plot(model, feature_names):
    out_path = ARTIFACTS_DIR / "feature_importance_fallback.png"

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8, 6))
    plt.barh(sorted_names[::-1], sorted_importances[::-1], color="#5b8cff")
    plt.xlabel("XGBoost feature_importances_")
    plt.title("Feature Importance — Fraud Model (fallback)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    model, feature_names = load_model_and_features()

    try:
        print("[shap] loading data sample and computing SHAP values...")
        X = load_background_data(feature_names)
        out_path = try_shap_plot(model, X)
        print(f"[shap] saved SHAP summary plot -> {out_path}")
    except Exception as exc:
        print(f"[shap] failed or timed out ({exc!r}); falling back to feature_importances_ bar chart")
        out_path = fallback_plot(model, feature_names)
        print(f"[fallback] saved feature importance bar chart -> {out_path}")


if __name__ == "__main__":
    main()
