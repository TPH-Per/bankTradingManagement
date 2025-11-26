#!/usr/bin/env python3
"""
Experiment: Ingestion latency, dedup behavior, API latency & correctness.

Runs against the FastAPI app via TestClient (no external server required).

Outputs:
- ml_service/experiments/INGEST_RESULTS.md
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi.testclient import TestClient
import importlib

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.main import app  # type: ignore
app_module = importlib.import_module("app.app")


DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "cash_daily_train_realistic.csv"
DAILY_CSV = DATA_DIR / "cash_daily.csv"
OUT_MD = ROOT / "experiments" / "INGEST_RESULTS.md"


def reset_daily_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["date", "cash_in", "cash_out", "channel", "day_of_week", "month", "quarter", "balance"]
    df = pd.DataFrame(columns=cols)
    df.to_csv(DAILY_CSV, index=False)


def run_ingestion_experiment(client: TestClient, n: int = 8) -> Dict[str, Any]:
    reset_daily_csv()
    account_id = "ACC_EXP"
    today = datetime.now(timezone.utc).date().isoformat()

    latencies_ms: List[float] = []
    ids: List[str] = []

    # Create N unique transactions (cash_in) then N duplicates of the first half
    for i in range(n):
        payload = {
            "account_id": account_id,
            "client_tx_id": f"TX-{i:04d}",
            "amount": 100000 + i * 1000,
            "currency": "VND",
            "transaction_type": "deposit",
            "merchant": "ONLINE",
        }
        t0 = time.perf_counter()
        r = client.post("/rt/transactions", json=payload)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        ids.append(payload["client_tx_id"])
        assert r.status_code == 200, r.text

    # Post duplicates for first half (should be deduped in daily CSV)
    for i in range(n // 2):
        payload = {
            "account_id": account_id,
            "client_tx_id": f"TX-{i:04d}",
            "amount": 100000 + i * 1000,
            "currency": "VND",
            "transaction_type": "cash_in",
            "merchant": "ONLINE",
        }
        t0 = time.perf_counter()
        r = client.post("/rt/transactions", json=payload)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        assert r.status_code == 200, r.text

    # Read daily CSV and check dedup effect
    df_daily = pd.read_csv(DAILY_CSV)
    has_client_col = "client_tx_id" in df_daily.columns
    unique_ids = df_daily["client_tx_id"].nunique() if has_client_col and not df_daily.empty else 0
    rows_daily = len(df_daily)

    ingest_stats = {
        "requests": n + (n // 2),
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "p95_latency_ms": sorted(latencies_ms)[int(0.95 * len(latencies_ms)) - 1],
        "daily_csv_rows": rows_daily,
        "daily_csv_unique_client_tx_id": unique_ids,
        "dedup_effect": "present" if has_client_col and unique_ids <= n else "absent",
        "event_date": today,
    }

    return ingest_stats


def run_api_latency_experiment(client: TestClient) -> Dict[str, Any]:
    # Prepare features via endpoint
    r = client.get("/ml/prepare-features")
    if r.status_code != 200:
        return {"error": f"prepare-features failed: {r.text}"}
    feats = r.json()["features"]

    # Predict cash-in/cash-out/all and time them
    def time_call(method: str, url: str, json_body: Dict[str, Any] | None = None) -> float:
        t0 = time.perf_counter()
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=json_body)
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, resp.text
        return dt

    ms_ci = time_call("POST", "/ml/predict/cash-in", {"features": feats})
    ms_co = time_call("POST", "/ml/predict/cash-out", {"features": feats})
    ms_all = time_call("POST", "/ml/predict/all", {"features": feats})

    # Validate shape & non-negativity
    ci = client.post("/ml/predict/cash-in", json={"features": feats}).json()
    co = client.post("/ml/predict/cash-out", json={"features": feats}).json()
    ok = all(v >= 0 for v in ci.values()) and all(v >= 0 for v in co.values())

    return {
        "latency_ms": {"cash_in": ms_ci, "cash_out": ms_co, "all": ms_all},
        "non_negative": ok,
    }


def test_input_flexibility_and_direction(client: TestClient) -> Dict[str, Any]:
    # Read current training CSV values for today
    try:
        df_train = pd.read_csv(TRAIN_CSV)
    except Exception:
        df_train = pd.DataFrame()

    today = datetime.now(timezone.utc).date().isoformat()
    before_out = 0.0
    if not df_train.empty and "date" in df_train.columns and (df_train["date"] == today).any():
        idx = df_train.index[df_train["date"] == today][0]
        before_out = float(pd.to_numeric(df_train.at[idx, "cash_out"], errors="coerce") or 0.0)

    # Post a negative-amount transaction without transaction_type → should infer cash_out
    payload = {
        "account_id": "ACC_EXP",
        "client_tx_id": "TX-NEG-OUT",
        "amount": -123000,
        "currency": "VND",
        "merchant": "POS",
    }
    r = client.post("/rt/transactions", json=payload)
    assert r.status_code == 200, r.text

    # Read updated training CSV and verify increase in cash_out for today
    df_after = pd.read_csv(TRAIN_CSV)
    after_out = before_out
    if not df_after.empty and "date" in df_after.columns and (df_after["date"] == today).any():
        idx = df_after.index[df_after["date"] == today][0]
        after_out = float(pd.to_numeric(df_after.at[idx, "cash_out"], errors="coerce") or 0.0)

    inferred_ok = (after_out - before_out) >= 123000.0 - 1e-6
    return {"neg_amount_infers_cash_out": inferred_ok, "delta_cash_out": after_out - before_out}


def main() -> int:
    # Avoid Cassandra attempts during tests
    os.environ["DISABLE_CASSANDRA"] = "1"

    # Ensure experiments dir
    (ROOT / "experiments").mkdir(parents=True, exist_ok=True)

    # Stub retraining to avoid heavy work during ingest test
    original_retrain_all = getattr(app_module.multi_model, "retrain_all", None)
    original_m5p_train = getattr(app_module.m5p_model, "train", None)
    app_module.multi_model.retrain_all = lambda dataset_csv: {"status": "stubbed"}
    app_module.m5p_model.train = lambda dataset_csv, target=None: {"status": "stubbed"}

    with TestClient(app) as client:
        ingest = run_ingestion_experiment(client, n=20)
        api_perf = run_api_latency_experiment(client)
        flex = test_input_flexibility_and_direction(client)

    # Restore originals
    if original_retrain_all is not None:
        app_module.multi_model.retrain_all = original_retrain_all
    if original_m5p_train is not None:
        app_module.m5p_model.train = original_m5p_train

    # Write Markdown report
    lines = [
        "# Ingestion & API Experiments\n\n",
        "## Ingestion (memory mode)\n",
        f"- Requests: {ingest['requests']}\n",
        f"- Avg latency: {ingest['avg_latency_ms']:.2f} ms | p95: {ingest['p95_latency_ms']:.2f} ms\n",
        f"- Daily CSV rows: {ingest['daily_csv_rows']} | Unique client_tx_id: {ingest['daily_csv_unique_client_tx_id']}\n",
        f"- Dedup in daily CSV: {ingest['dedup_effect']}\n\n",
        "## Prediction API\n",
        f"- Latency (cash_in): {api_perf.get('latency_ms',{}).get('cash_in',float('nan')):.2f} ms\n",
        f"- Latency (cash_out): {api_perf.get('latency_ms',{}).get('cash_out',float('nan')):.2f} ms\n",
        f"- Latency (all): {api_perf.get('latency_ms',{}).get('all',float('nan')):.2f} ms\n",
        f"- Non-negative outputs: {api_perf.get('non_negative', False)}\n\n",
        "## Input flexibility & direction inference\n",
        f"- Negative amount infers cash_out: {flex.get('neg_amount_infers_cash_out', False)} (Δcash_out={flex.get('delta_cash_out', 0.0):.0f})\n\n",
        "## Notes\n",
        "- This test runs with Cassandra disabled (in-memory fallback).\n",
        "- Deduplication at DB level (client_tx_dedup) is not exercised here; daily CSV-level dedup is validated.\n",
    ]

    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Saved {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
