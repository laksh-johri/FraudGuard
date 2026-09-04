"""
FastAPI backend for the real-time fraud detection demo.

Loads the XGBoost model trained by model/train.py and exposes:
  GET  /health    -> liveness check
  GET  /features  -> ordered feature list the model expects
  GET  /sample    -> a randomly generated transaction (for demo/streaming)
  POST /predict   -> score a transaction, returns fraud probability + verdict
"""

import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, RootModel

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"

FRAUD_THRESHOLD = 0.5

# Fixed feature vectors picked from the training distribution where the model is
# extremely confident (proba > 0.999 fraud / < 1e-5 legit), used by /sample?force=
# so demo buttons on stage trigger reliably instead of depending on random luck.
FRAUD_ARCHETYPE = {
    "f0": -5.645, "f1": -3.0287, "f2": 0.7904, "f3": -0.7754, "f4": 2.6919,
    "f5": -2.1907, "f6": 1.3861, "f7": 0.4205, "f8": -0.4367, "f9": -0.33,
    "f10": 1.665, "f11": -0.0928, "f12": -2.0329, "f13": -0.6147, "f14": 5.0187,
    "Amount": 790.69,
}
LEGIT_ARCHETYPE = {
    "f0": 0.1081, "f1": 2.8937, "f2": 0.7839, "f3": -2.8963, "f4": -2.3084,
    "f5": -3.5742, "f6": 2.7522, "f7": -1.944, "f8": -2.1419, "f9": 2.1316,
    "f10": 1.8236, "f11": -1.8461, "f12": 4.4286, "f13": 0.759, "f14": 3.6087,
    "Amount": 30.25,
}

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_names: list[str] = []


@app.on_event("startup")
def load_artifacts():
    global model, feature_names
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise RuntimeError(
            f"Model artifacts not found in {ARTIFACTS_DIR}. Run `python model/train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text())


class Transaction(RootModel[Dict[str, float]]):
    """Feature name -> value, e.g. {"f0": 0.12, ..., "Amount": 59.99}"""


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/features")
def get_features():
    return {"feature_names": feature_names}


@app.get("/sample")
def sample_transaction(force: str | None = Query(default=None, pattern="^(fraud|legit)$")):
    """Generate a transaction for demo/streaming purposes.

    Without `force`, loosely mirrors the synthetic training distribution
    (mostly normal transactions, occasionally shifted to look fraud-like)
    so scored results vary in an interesting way.

    With `force=fraud` or `force=legit`, jitters one of the fixed
    high-confidence archetypes above so the result reliably scores on the
    intended side of the threshold — for demo buttons that must not depend
    on random luck.
    """
    if force == "fraud":
        return _jitter_archetype(FRAUD_ARCHETYPE)
    if force == "legit":
        return _jitter_archetype(LEGIT_ARCHETYPE)

    is_fraud_like = np.random.random() < 0.05
    features = {}
    for name in feature_names:
        if name == "Amount":
            continue
        val = np.random.normal(0, 1)
        if is_fraud_like:
            val += np.random.normal(1.5, 0.5)
        features[name] = round(float(val), 4)

    amount = np.random.gamma(2.0, 50.0)
    if is_fraud_like:
        amount += np.random.gamma(3.0, 80.0)
    features["Amount"] = round(float(amount), 2)

    return features


def _jitter_archetype(base: Dict[str, float]) -> Dict[str, float]:
    features = {}
    for name, value in base.items():
        if name == "Amount":
            jittered = value + np.random.normal(0, value * 0.1)
            features[name] = round(float(max(1.0, jittered)), 2)
        else:
            features[name] = round(float(value + np.random.normal(0, 0.15)), 4)
    return features


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    data = transaction.root
    missing = [f for f in feature_names if f not in data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    row = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)

    start = time.perf_counter()
    proba = float(model.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= FRAUD_THRESHOLD,
        latency_ms=round(latency_ms, 3),
    )
