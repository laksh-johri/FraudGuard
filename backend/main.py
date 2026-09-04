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
    "V1": -10.6324, "V2": 7.2519, "V3": -17.6811, "V4": 8.2041, "V5": -10.1666,
    "V6": -4.5103, "V7": -12.9816, "V8": 6.7836, "V9": -4.6593, "V10": -14.9247,
    "V11": 8.3891, "V12": -16.4655, "V13": 0.3385, "V14": -14.2244, "V15": 0.5566,
    "V16": -11.684, "V17": -15.8416, "V18": -5.7532, "V19": 3.813, "V20": -0.8101,
    "V21": 2.7154, "V22": 0.6956, "V23": -1.1381, "V24": 0.4594, "V25": 0.3863,
    "V26": 0.5224, "V27": -1.4166, "V28": -0.4883, "Amount": 188.52,
}
LEGIT_ARCHETYPE = {
    "V1": -5.7509, "V2": 3.4624, "V3": -3.6983, "V4": -0.808, "V5": -2.5988,
    "V6": -1.3617, "V7": -0.5209, "V8": 2.1423, "V9": 1.9125, "V10": 2.5502,
    "V11": -1.6532, "V12": 0.3457, "V13": -0.9575, "V14": 1.1066, "V15": -0.1167,
    "V16": 0.5254, "V17": 0.388, "V18": -0.6016, "V19": -0.4596, "V20": 0.6049,
    "V21": -0.6069, "V22": -1.1039, "V23": 0.098, "V24": -0.0718, "V25": 0.1951,
    "V26": 0.1645, "V27": 0.5708, "V28": 0.4782, "Amount": 89.99,
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
