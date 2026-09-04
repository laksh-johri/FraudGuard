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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, RootModel

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"

FRAUD_THRESHOLD = 0.5

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
def sample_transaction():
    """Generate a random transaction for demo/streaming purposes.

    Loosely mirrors the synthetic training distribution (mostly normal
    transactions, occasionally shifted to look fraud-like) so scored
    results vary in an interesting way.
    """
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
