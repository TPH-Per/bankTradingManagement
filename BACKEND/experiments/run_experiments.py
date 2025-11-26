#!/usr/bin/env python3
"""
Run reproducible experiments on the BankTrading ML service codebase.

Outputs:
- Console summary
- ml_service/experiments/results.json
- ml_service/experiments/RESULTS.md
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Ensure we can import the app package (uses relative imports internally)
sys.path.insert(0, str(ROOT))

from app.multi_target_model import MultiTargetCashModel  # type: ignore


@dataclass
class TrainStats:
    rmse: float | None
    r2: float | None
    mae: float | None
    target_column: str | None
    model_path: str | None
    seconds: float


def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(map(str, df.columns.tolist())),
    }
    try:
        df_dates = pd.to_datetime(df["date"], errors="coerce")
        summary["date_min"] = str(df_dates.min().date()) if not df_dates.isna().all() else None
        summary["date_max"] = str(df_dates.max().date()) if not df_dates.isna().all() else None
    except Exception:
        summary["date_min"] = None
        summary["date_max"] = None
    for col in ("cash_in", "cash_out"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 0:
                summary[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                }
    return summary


def prepare_features_from_latest(df: pd.DataFrame) -> Dict[str, Any]:
    last_row = df.iloc[-1]
    # Safely parse date to derive flags
    try:
        from datetime import datetime
        date_obj = datetime.fromisoformat(str(last_row["date"]))
        day_of_month = date_obj.day
    except Exception:
        day_of_month = 15

    def at(df_, idx, col, default=0.0):
        try:
            return float(df_.iloc[idx][col])
        except Exception:
            return float(default)

    feats = {
        "cash_in_d0": float(last_row.get("cash_in", 0.0)),
        "cash_out_d0": float(last_row.get("cash_out", 0.0)),
        "cash_net_d0": float(last_row.get("cash_in", 0.0)) - float(last_row.get("cash_out", 0.0)),
        "lag1_in": at(df, -2, "cash_in", 0.0) if len(df) > 1 else 0.0,
        "lag7_in": at(df, -8, "cash_in", 0.0) if len(df) > 7 else 0.0,
        "roll_mean_7_in": float(pd.to_numeric(df.tail(7)["cash_in"], errors="coerce").mean()),
        "lag1_out": at(df, -2, "cash_out", 0.0) if len(df) > 1 else 0.0,
        "lag7_out": at(df, -8, "cash_out", 0.0) if len(df) > 7 else 0.0,
        "roll_mean_7_out": float(pd.to_numeric(df.tail(7)["cash_out"], errors="coerce").mean()),
        "dow": int(last_row.get("day_of_week", 0)),
        "is_weekend": int(int(last_row.get("day_of_week", 0)) >= 5),
        "is_month_end": int(day_of_month >= 25),
        "is_payday": int(day_of_month == 15 or day_of_month >= 25),
        "channel": str(last_row.get("channel", "DEFAULT")),
    }
    return feats


def train_with_timing(multi_model: MultiTargetCashModel, dataset_csv: Path) -> Dict[str, TrainStats]:
    results: Dict[str, TrainStats] = {}

    # Define training plan (model attr, target_col, key)
    plan = [
        (multi_model.cash_in_next_day, "cash_in_next_day", "cash_in_next_day"),
        (multi_model.cash_out_next_day, "cash_out_next_day", "cash_out_next_day"),
        (multi_model.cash_in_h7, "cash_in_h7_sum", "cash_in_h7"),
        (multi_model.cash_out_h7, "cash_out_h7_sum", "cash_out_h7"),
        (multi_model.cash_in_next_month, "cash_in_next_month_sum", "cash_in_next_month"),
        (multi_model.cash_out_next_month, "cash_out_next_month_sum", "cash_out_next_month"),
    ]

    for model_api, target, key in plan:
        t0 = time.perf_counter()
        stats = model_api.train(str(dataset_csv), target_column=target)
        dt = time.perf_counter() - t0
        results[key] = TrainStats(
            rmse=float(stats.get("rmse")) if stats.get("rmse") is not None else None,
            r2=float(stats.get("r2")) if stats.get("r2") is not None else None,
            mae=float(stats.get("mae")) if stats.get("mae") is not None else None,
            target_column=str(stats.get("target_column")) if stats.get("target_column") else target,
            model_path=str(stats.get("model_path")) if stats.get("model_path") else None,
            seconds=dt,
        )
    return results


def benchmark_inference(multi_model: MultiTargetCashModel, features: Dict[str, Any], loops: int = 1000) -> Dict[str, Any]:
    # Warmup
    multi_model.predict_all(features)
    t0 = time.perf_counter()
    for _ in range(loops):
        multi_model.predict_all(features)
    t_all = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(loops):
        multi_model.predict_cash_in(features)
    t_ci = time.perf_counter() - t1

    t2 = time.perf_counter()
    for _ in range(loops):
        multi_model.predict_cash_out(features)
    t_co = time.perf_counter() - t2

    return {
        "loops": loops,
        "predict_all_ms_per_call": (t_all / loops) * 1000.0,
        "predict_cash_in_ms_per_call": (t_ci / loops) * 1000.0,
        "predict_cash_out_ms_per_call": (t_co / loops) * 1000.0,
    }


def main() -> int:
    dataset = DATA_DIR / "cash_daily_train_realistic.csv"
    if not dataset.exists():
        print(f"ERROR: Dataset not found: {dataset}")
        return 1

    df = pd.read_csv(dataset)
    ds_summary = dataset_summary(df)

    # Initialize models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    multi_model = MultiTargetCashModel(base_model_dir=MODELS_DIR)

    # Train all with timing
    print("Training 6 models with timing...")
    t0 = time.perf_counter()
    train_results = train_with_timing(multi_model, dataset)
    total_train_s = time.perf_counter() - t0

    # Prepare features and benchmark inference
    feats = prepare_features_from_latest(df)
    bench = benchmark_inference(multi_model, feats, loops=1000)

    # Compose results
    output = {
        "dataset": {
            "path": str(dataset),
            **ds_summary,
        },
        "training": {
            "total_seconds": total_train_s,
            "per_model": {k: asdict(v) for k, v in train_results.items()},
        },
        "inference": bench,
        "feature_sample": feats,
    }

    # Write JSON and Markdown summaries
    out_dir = ROOT / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    md = [
        "# Experimental Results\n",
        f"Dataset: `{dataset}`\n",
        f"Rows: {ds_summary.get('rows')} | Date range: {ds_summary.get('date_min')} → {ds_summary.get('date_max')}\n\n",
        "## Training\n",
        f"Total time: {total_train_s:.2f}s\n\n",
    ]
    for key, stats in train_results.items():
        md.append(f"- {key}: rmse={stats.rmse:.2f} r2={stats.r2:.4f} mae={stats.mae:.2f} time={stats.seconds:.2f}s\n")
    md.append("\n## Inference latency (CPU)\n")
    md.append(f"- predict_all: {bench['predict_all_ms_per_call']:.3f} ms/call\n")
    md.append(f"- predict_cash_in: {bench['predict_cash_in_ms_per_call']:.3f} ms/call\n")
    md.append(f"- predict_cash_out: {bench['predict_cash_out_ms_per_call']:.3f} ms/call\n")

    (out_dir / "RESULTS.md").write_text("".join(md), encoding="utf-8")

    # Print concise console summary
    print("\n=== DATASET ===")
    print(json.dumps(ds_summary, indent=2))
    print("\n=== TRAINING (total) ===")
    print(f"{total_train_s:.2f}s for 6 models")
    print("\nPer-model:")
    for k, v in train_results.items():
        print(f"  - {k:22s} rmse={v.rmse:.2f} r2={v.r2:.4f} mae={v.mae:.2f} time={v.seconds:.2f}s")
    print("\n=== INFERENCE (ms/call) ===")
    print(json.dumps(bench, indent=2))
    print(f"\nSaved: {out_dir / 'results.json'} and {out_dir / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
