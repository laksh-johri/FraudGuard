"""
Honest latency benchmark for the real scoring path.

Fires N single-row predictions through the *actual* hybrid pipeline used by
backend/main.py's /predict handler -- model.predict_proba + the velocity /
rule layer -- by importing that module directly and calling its functions,
rather than re-implementing the logic here (which could silently drift out
of sync with what production actually runs). Each transaction uses a fresh
entity id so every call takes the "first transaction for this id" velocity
path, matching the common case rather than an artificially escalated one.

Reports p50, p95, p99, min, and max latency in milliseconds and saves them
to model/artifacts/latency_stats.json.

Usage:
    python model/benchmark_latency.py [--n 1000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent / "backend"
ARTIFACTS_DIR = HERE / "artifacts"
STATS_PATH = ARTIFACTS_DIR / "latency_stats.json"

sys.path.insert(0, str(BACKEND_DIR))
import main as backend_main  # noqa: E402


def run_benchmark(n_iters: int, warmup: int = 20) -> dict:
    backend_main.load_artifacts()
    model = backend_main.model
    feature_names = backend_main.feature_names

    def score_once(entity_id: str) -> float:
        """Exactly the timed block inside backend.main.predict(), called
        directly against the real functions so this benchmark can't diverge
        from what the API actually does."""
        data = backend_main.sample_transaction(force=None)
        row = pd.DataFrame([[float(data[f]) for f in feature_names]], columns=feature_names)

        start = time.perf_counter()
        model_score = float(model.predict_proba(row)[0, 1])
        txn_count_last_60s, amount_vs_avg = backend_main.compute_velocity_features(
            entity_id, float(data["Amount"])
        )
        rule_score = backend_main.compute_rule_score(txn_count_last_60s, amount_vs_avg)
        final_risk = max(model_score, rule_score)  # noqa: F841 (parity with predict())
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms

    for i in range(warmup):
        score_once(f"bench_warmup_{i}")

    latencies = np.empty(n_iters)
    for i in range(n_iters):
        latencies[i] = score_once(f"bench_{i}")

    stats = {
        "n_iters": n_iters,
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "min_ms": round(float(latencies.min()), 3),
        "max_ms": round(float(latencies.max()), 3),
        "mean_ms": round(float(latencies.mean()), 3),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="Number of scored transactions")
    args = parser.parse_args()

    print(f"[bench] running {args.n} single-row predictions through the real scoring path...")
    stats = run_benchmark(args.n)

    print("\n=== Latency benchmark (ms) ===")
    print(f"n_iters: {stats['n_iters']}")
    print(f"min:  {stats['min_ms']:.3f}")
    print(f"p50:  {stats['p50_ms']:.3f}")
    print(f"p95:  {stats['p95_ms']:.3f}")
    print(f"p99:  {stats['p99_ms']:.3f}")
    print(f"max:  {stats['max_ms']:.3f}")
    print(f"mean: {stats['mean_ms']:.3f}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[artifacts] saved {STATS_PATH}")


if __name__ == "__main__":
    main()
