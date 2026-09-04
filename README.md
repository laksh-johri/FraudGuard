# Real-Time Fraud Detection System

## Problem

Traditional fraud prevention leans on static, hand-written rules ("block if amount > ₹50,000 and country != billing country"). Rules engines like this are slow to adapt — every new fraud pattern needs a human to notice it, write a new rule, and ship it, often days or weeks after attackers have moved on. They're also blunt instruments: fixed thresholds can't weigh dozens of correlated signals at once, so they either miss subtle, low-amount fraud that doesn't trip any single rule, or over-block legitimate customers who happen to match one. A machine-learned model trained on historical transactions can instead learn the *combinations* of signals that actually separate fraud from legitimate behavior, adapt as new data comes in, and score every transaction in milliseconds — fast enough to sit directly in the checkout path instead of running as an after-the-fact batch report.

## Architecture

```
                 ┌────────────────┐
                 │    Frontend     │
                 │  (index.html)   │
                 │  checkout UI    │
                 └────────┬────────┘
                          │ 1. GET /sample  (transaction features)
                          │ 2. POST /predict
                          ▼
                 ┌────────────────┐
                 │    FastAPI      │
                 │   (backend)     │
                 └────────┬────────┘
                          │ 3. predict_proba(row)
                          ▼
                 ┌────────────────┐
                 │    XGBoost      │
                 │  fraud_model    │
                 │   (~3ms/call)   │
                 └────────┬────────┘
                          │ 4. fraud_probability
                          ▼
                 ┌────────────────┐
                 │ Decision point  │
                 │ final_risk>0.5? │
                 └───┬────────┬────┘
                     │        │
              yes ───┘        └─── no
                     ▼              ▼
          ┌──────────────────┐  ┌───────────────────┐
          │  BLOCK            │  │  Open Razorpay     │
          │  transaction      │  │  checkout (₹)      │
          │  before payment   │  │                    │
          └──────────────────┘  └───────────────────┘
```

The model scores the transaction **before** the Razorpay checkout modal ever opens. A high fraud probability short-circuits the flow and the payment provider is never invoked; only transactions the model clears reach Razorpay.

## Key design choice: no LLM in the hot path

This system deliberately uses a small, structured **XGBoost** classifier instead of an LLM for scoring. Checkout is a latency-sensitive path — a real payment decision needs to happen in single-digit milliseconds, not the hundreds of milliseconds to seconds an LLM call would add, and a numeric fraud model has no need for the language-reasoning capability an LLM provides. XGBoost on ~29 structured features gets sub-100ms inference (measured ~1.6ms per call) with strong accuracy (ROC-AUC 0.98), which is what a synchronous "block before payment" decision requires.

## Model metrics

Trained on the real Kaggle "Credit Card Fraud Detection" dataset (`model/creditcard.csv`, 284,807 transactions, 492 fraud / 0.173%), evaluated on a held-out test split:

| Metric | Value |
|---|---|
| ROC-AUC | 0.98 |
| PR-AUC (avg. precision) | 0.86 |
| Precision | 0.61 |
| Recall | 0.88 |
| F1 | 0.72 |
| Inference latency | ~1.6ms per transaction (single-row `predict_proba`) |

Confusion matrix on the held-out test split ([[TN, FP], [FN, TP]]):

```
[[56810    54]
 [   12    86]]
```

(Run `python model/train.py` to reproduce this output. If `model/creditcard.csv` is absent, training falls back to a synthetic imbalanced dataset instead.)

## Setup & run

### 1. Model

```bash
cd model
pip install -r requirements.txt
python train.py
```

This trains the XGBoost classifier and writes `model/artifacts/fraud_model.pkl` and `model/artifacts/feature_names.json`. If `model/creditcard.csv` (the Kaggle "Credit Card Fraud Detection" dataset) is present it's used automatically; otherwise a synthetic imbalanced dataset is generated so training always works out of the box.

Optionally, generate the SHAP explainability plot:

```bash
python generate_shap_plot.py
```

Saves `model/artifacts/shap_feature_importance.png` (falls back to a `feature_importances_` bar chart if SHAP errors or times out).

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Serves the API at `http://127.0.0.1:8000`:

- `GET /health` — liveness check
- `GET /features` — ordered feature list the model expects
- `GET /sample` — a randomly generated transaction for demo/streaming (`?force=fraud` / `?force=legit` for reliable demo outcomes)
- `POST /predict` — score a transaction, returns fraud probability + verdict + latency

### 3. Frontend

No build step — it's a single static HTML file. Just open it:

```bash
# from the project root
start frontend/index.html   # Windows
```

or serve it with any static file server. Set the **API URL** field to your backend address (default `http://127.0.0.1:8000`), then use **Score One Transaction** / **Start Live Stream** to watch transactions get scored, or use the **Checkout — Pay Now** panel to run the full "score → block or open Razorpay" flow. Amounts throughout the UI are in ₹ (INR), matching the currency Razorpay is configured to charge.
