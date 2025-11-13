# main.py (extended)
import os
import io
import sys
import uuid
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Body, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator
from threading import Lock

from .dual_m5p import DualCashModelAPI
from .ml_m5p import M5PModelAPI
from .multi_target_model import MultiTargetCashModel
from .cassandra_service import CassandraUnavailable, cassandra_service
from .scheduler import DailyAggregationScheduler

# -----------------------------------------------------------------------------
# Config & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bankTrading")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def _resolve_dir(path_value: Optional[str], fallback: Path) -> Path:
    if path_value:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = fallback.parent / candidate
        return candidate.resolve()
    return fallback.resolve()


MODEL_DIR = _resolve_dir(os.environ.get("MODEL_DIR"), ROOT_DIR / "models")
DATA_DIR = _resolve_dir(os.environ.get("DATA_DIR"), ROOT_DIR / "data")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MODEL_DIR", str(MODEL_DIR))
os.environ.setdefault("DATA_DIR", str(DATA_DIR))

TRAINING_BASE_DATASET = DATA_DIR / "cash_daily_train_realistic.csv"
TRAINING_DATASET_PATH = TRAINING_BASE_DATASET
DAILY_CSV_PATH = DATA_DIR / "cash_daily.csv"  # Temporary daily data
DEFAULT_TRAINING_COLUMNS: List[str] = [
    "date",
    "cash_in",
    "cash_out",
    "channel",
    "day_of_week",
    "month",
    "quarter",
    "balance",
]
DEFAULT_TRAINING_COLUMNS_INDEX = pd.Index(DEFAULT_TRAINING_COLUMNS)
CURRENCY_TO_VND: Dict[str, float] = {
    "VND": 1.0,
    "USD": 23650.0,
    "EUR": 25800.0,
    "GBP": 29800.0,
    "JPY": 160.0,
    "AUD": 15400.0,
    "SGD": 17500.0,
}
DEFAULT_CURRENCY = "VND"

# Multi-target model system (6 models: next_day, h7, next_month for both cash_in and cash_out)
multi_model = MultiTargetCashModel(base_model_dir=MODEL_DIR)

# Backward-compatible reference
m5p_model = multi_model.cash_in_next_day

# Daily aggregation scheduler
scheduler: Optional[DailyAggregationScheduler] = None

# -----------------------------------------------------------------------------
# FastAPI + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="bankTrading ML Service", version="2.3")
api_router = APIRouter()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev cho phép tất cả; khi deploy nhớ siết domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def retrain_all_models():
    """
    Callback to retrain all 6 models after daily aggregation.
    """
    logger.info("Retraining all models after daily aggregation...")
    try:
        stats = multi_model.retrain_all(dataset_csv=str(TRAINING_DATASET_PATH))
        logger.info("Retrain complete: %s", stats)
    except Exception as e:
        logger.exception("Failed to retrain models")


@app.on_event("startup")
async def startup_bootstrap():
    """
    Bootstrap multi-target models on service startup.
    Load existing models from disk or train if needed.
    Also start the daily aggregation scheduler.
    """
    global scheduler

    dataset = os.getenv("M5P_DATASET", str(TRAINING_DATASET_PATH))
    force = os.getenv("M5P_FORCE_RETRAIN", "0") == "1"

    logger.info("Bootstrapping multi-target cash flow models (6 models)...")
    try:
        stats = multi_model.bootstrap(
            dataset_csv=dataset,
            force_retrain=force
        )
        logger.info("Multi-target model bootstrap complete: %s", stats)
    except Exception as e:
        logger.warning("Multi-target model bootstrap failed (will use on-demand loading): %s", e)

    # Ensure the daily CSV exists before starting scheduler
    try:
        _ensure_daily_csv()
    except Exception as e:
        logger.warning("Failed to ensure daily CSV exists: %s", e)

    # Start daily aggregation scheduler
    logger.info("Starting daily aggregation scheduler...")
    scheduler = DailyAggregationScheduler(
        daily_csv=DAILY_CSV_PATH,
        training_csv=TRAINING_DATASET_PATH,
        retrain_callback=retrain_all_models
    )
    scheduler.start()
    logger.info("Scheduler started - will run at 0:00 AM daily")


@app.on_event("shutdown")
async def shutdown_cleanup():
    """
    Stop the scheduler on shutdown.
    """
    global scheduler
    if scheduler:
        logger.info("Stopping scheduler...")
        await scheduler.stop()
        logger.info("Scheduler stopped")


@app.middleware("http")
async def cassandra_audit_middleware(request: Request, call_next):
    start_ts = datetime.now(timezone.utc)
    response = await call_next(request)
    try:
        if cassandra_service.available():
            account_id = request.query_params.get("account_id")
            client_ip = request.client.host if request.client else None
            cassandra_service.log_api_call(
                day=start_ts.date(),
                ts=start_ts,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                account_id=account_id,
                client_ip=client_ip,
            )
    except Exception:
        logger.exception("Failed to log API call to Cassandra.")
    return response

# -----------------------------------------------------------------------------
# In-memory stores (placeholder) -> thay bằng Cassandra khi sẵn sàng
# -----------------------------------------------------------------------------
TX_STORE: List[Dict[str, Any]] = []   # mỗi item: {account_id,event_ts,event_date,tx_id,amount,...}
KPI_STORE: Dict[tuple, float] = {}    # key: (event_date:str, metric:str) -> value: float
LAST_TRAIN_METRICS: Optional[Dict[str, Any]] = None
LOCK = Lock()


def _update_daily_csv_from_transaction(tx: Dict[str, Any], direction: str, amount_vnd: float, event_date_str: str, event_date_obj: date) -> None:
    """
    Update the daily CSV with the latest transaction for scheduler aggregation.
    """
    try:
        daily_csv_path = _ensure_daily_csv()
        if not daily_csv_path.exists():
            return

        df = pd.read_csv(daily_csv_path) if daily_csv_path.exists() else pd.DataFrame()
        if "date" not in df.columns:
            df = pd.DataFrame(columns=pd.Index(["date", "cash_in", "cash_out", "channel", "day_of_week", "month", "quarter", "balance"]))

        # Check if transaction already exists (by client_tx_id)
        client_tx_id = tx.get("client_tx_id", "")
        if client_tx_id and "client_tx_id" in df.columns and (df["client_tx_id"] == client_tx_id).any():
            logger.info(f"Transaction {client_tx_id} already exists in daily CSV, skipping")
            return

        # Add transaction to daily CSV
        new_row = {
            "date": event_date_str,
            "cash_in": amount_vnd if direction == "cash_in" else 0.0,
            "cash_out": amount_vnd if direction == "cash_out" else 0.0,
            "channel": tx.get("merchant", "DEFAULT") or "DEFAULT",
            "day_of_week": event_date_obj.weekday(),
            "month": event_date_obj.month,
            "quarter": ((event_date_obj.month - 1) // 3) + 1,
            "balance": 0.0,  # Will be calculated during aggregation
        }
        
        # Add client_tx_id if available for deduplication
        if client_tx_id:
            new_row["client_tx_id"] = client_tx_id

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(daily_csv_path, index=False)
        logger.info(f"Updated daily CSV with transaction for {event_date_str}")
    except Exception:
        logger.exception("Failed to update daily CSV from transaction.")



def _resolve_data_path(path_value: str) -> Path:
    """
    Resolve a provided data path against common directories so the GUI can
    reference files located under ml_service/data or absolute paths.
    """
    candidate = Path(path_value)
    search_candidates: List[Path] = []

    if candidate.is_absolute():
        search_candidates.append(candidate)
    else:
        search_candidates.extend([
            Path.cwd() / candidate,
            DATA_DIR / candidate,
            ROOT_DIR / candidate,
        ])
        search_candidates.extend([
            DATA_DIR / candidate.name,
            (ROOT_DIR / "data") / candidate.name,
        ])

    seen: set[Path] = set()
    for path in search_candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Data file not found: {path_value}")


def _ensure_training_dataset() -> Path:
    """
    Ensure a writable training dataset exists that we can continuously
    update with realtime cash-in/cash-out activity.
    """
    dataset_path = TRAINING_DATASET_PATH
    if dataset_path.exists():
        return dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if TRAINING_BASE_DATASET.exists() and dataset_path != TRAINING_BASE_DATASET:
        shutil.copyfile(TRAINING_BASE_DATASET, dataset_path)
    elif not dataset_path.exists():
        pd.DataFrame(columns=DEFAULT_TRAINING_COLUMNS_INDEX).to_csv(dataset_path, index=False)
    return dataset_path


def _ensure_daily_csv() -> Path:
    """
    Ensure the temporary daily aggregation CSV exists with the expected columns.
    """
    dataset_path = DAILY_CSV_PATH
    if dataset_path.exists():
        return dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "date",
        "cash_in",
        "cash_out",
        "channel",
        "day_of_week",
        "month",
        "quarter",
        "balance",
    ]
    column_index = pd.Index(columns)
    pd.DataFrame(columns=column_index).to_csv(dataset_path, index=False)
    return dataset_path


def _normalize_transaction_type(value: Optional[str]) -> Optional[str]:
    """
    Normalize incoming transaction type labels to the canonical
    'cash_in' / 'cash_out' strings expected by downstream logic.
    """
    if not value:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in {"cash_in", "in", "deposit", "cashin"}:
        return "cash_in"
    if cleaned in {"cash_out", "out", "withdraw", "withdrawal", "cashout"}:
        return "cash_out"
    return cleaned or None


def _infer_cash_direction(tx: Dict[str, Any]) -> str:
    """
    Attempt to determine whether a transaction should count as cash-in or cash-out.
    Priority:
        1. transaction_type field (if present)
        2. extra_json.type / extra_json.direction value
        3. sign of amount (negative -> cash_out)
    """
    # Check transaction_type field first
    tx_type = tx.get("transaction_type")
    normalized_type = _normalize_transaction_type(tx_type)
    if normalized_type == "cash_out":
        return "cash_out"
    if normalized_type == "cash_in":
        return "cash_in"

    # Check extra_json
    extra = tx.get("extra_json")
    direction = None
    if isinstance(extra, dict):
        direction = extra.get("type") or extra.get("direction")
    if direction:
        normalized_direction = _normalize_transaction_type(direction)
        if normalized_direction == "cash_out":
            return "cash_out"
        if normalized_direction == "cash_in":
            return "cash_in"

    # Fallback to amount sign
    try:
        amount_value = float(tx.get("amount", 0.0))
    except (TypeError, ValueError):
        amount_value = 0.0
    return "cash_out" if amount_value < 0 else "cash_in"


def _update_training_dataset_from_transaction(tx: Dict[str, Any]) -> None:
    """
    Update the rolling training dataset with the latest transaction and
    trigger a retrain of the M5P model.
    Also update the daily CSV for scheduler aggregation.
    """
    global LAST_TRAIN_METRICS
    try:
        dataset_path = _ensure_training_dataset()
        if not dataset_path.exists():
            return
        
        # Also update the daily CSV for scheduler
        daily_csv_path = _ensure_daily_csv()

        event_date_str = tx.get("event_date")
        if not event_date_str:
            event_ts = tx.get("event_ts")
            if isinstance(event_ts, datetime):
                event_date_str = event_ts.date().isoformat()
            elif isinstance(event_ts, str) and event_ts:
                event_date_str = event_ts.split("T", 1)[0]
        if not event_date_str:
            logger.warning("Skipping training update; transaction missing event_date: %s", tx)
            return

        try:
            event_date_obj = datetime.fromisoformat(event_date_str).date()
        except ValueError:
            event_date_obj = datetime.now(timezone.utc).date()

        direction = _infer_cash_direction(tx)
        currency_code = str(tx.get("currency") or DEFAULT_CURRENCY).upper()
        rate = CURRENCY_TO_VND.get(currency_code)
        if rate is None:
            logger.warning("Skipping training update; unsupported currency %s", currency_code)
            return
        try:
            amount_raw = float(tx.get("amount", 0.0))
        except (TypeError, ValueError):
            amount_raw = 0.0
        amount_value = abs(amount_raw)
        if amount_value == 0.0:
            logger.info("Transaction amount zero; skipping training update.")
            return
        amount_vnd = amount_value * rate

        df = pd.read_csv(dataset_path) if dataset_path.exists() else pd.DataFrame()
        if "date" not in df.columns:
            df = pd.DataFrame(columns=DEFAULT_TRAINING_COLUMNS_INDEX)

        if "date" in df.columns and (df["date"] == event_date_str).any():
            idx = df.index[df["date"] == event_date_str][0]
        else:
            new_row = {
                "date": event_date_str,
                "cash_in": 0.0,
                "cash_out": 0.0,
                "day_of_week": event_date_obj.weekday(),
                "month": event_date_obj.month,
                "quarter": ((event_date_obj.month - 1) // 3) + 1,
                "balance": 0.0,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            idx = df.index[df["date"] == event_date_str][0]

        for numeric_col in ["cash_in", "cash_out", "balance"]:
            if numeric_col not in df.columns:
                df[numeric_col] = 0.0
            numeric_result = pd.to_numeric(df[numeric_col], errors="coerce")
            # Ensure we're working with a Series before using pandas methods
            if not isinstance(numeric_result, pd.Series):
                numeric_result = pd.Series(numeric_result, index=df.index if hasattr(df, 'index') else None)
            df[numeric_col] = numeric_result.fillna(0.0)

        value_in = df.at[idx, "cash_in"]
        value_out = df.at[idx, "cash_out"]
        current_cash_in = (
            float(value_in) if value_in not in ("", None) and not pd.isna(value_in) else 0.0
        )
        current_cash_out = (
            float(value_out) if value_out not in ("", None) and not pd.isna(value_out) else 0.0
        )

        if direction == "cash_out":
            current_cash_out += amount_vnd
        else:
            current_cash_in += amount_vnd

        df.at[idx, "cash_in"] = current_cash_in
        df.at[idx, "cash_out"] = current_cash_out
        df.at[idx, "balance"] = current_cash_in - current_cash_out

        df = df.sort_values("date")
        df.to_csv(dataset_path, index=False)
        
        # Also update the daily CSV for scheduler aggregation
        _update_daily_csv_from_transaction(tx, direction, amount_vnd, event_date_str, event_date_obj)

        # Retrain all multi-target models with updated dataset
        with LOCK:
            try:
                multi_stats = multi_model.retrain_all(dataset_csv=str(dataset_path))
                LAST_TRAIN_METRICS = multi_stats
                logger.info(
                    "Retrained all 6 models after transaction; "
                    "cash_in_next_day RMSE=%.4f, cash_out_next_day RMSE=%.4f",
                    multi_stats.get("cash_in_next_day", {}).get("rmse", float("nan")),
                    multi_stats.get("cash_out_next_day", {}).get("rmse", float("nan"))
                )
            except Exception as e:
                logger.warning("Multi-target model retrain failed, falling back to single model: %s", e)
                # Fallback to legacy single model
                metrics = m5p_model.train(str(dataset_path))
                LAST_TRAIN_METRICS = metrics
                logger.info("Retrained legacy M5P model after transaction; RMSE=%.4f", metrics.get("rmse", float("nan")))
    except Exception:
        logger.exception("Failed to update training dataset from transaction.")


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class TxCreate(BaseModel):
    account_id: Optional[str] = Field(default=None)
    client_tx_id: Optional[str] = Field(default=None)
    amount: float
    currency: str = Field(default=DEFAULT_CURRENCY)
    transaction_type: Optional[str] = Field(default=None, description='"cash_in" or "cash_out"')
    merchant: Optional[str] = None
    status: Optional[str] = None
    extra_json: Optional[Dict[str, Any]] = None
    # Frontend-specific helper fields
    sender_id: Optional[str] = Field(default=None, description="Peer-to-peer sender")
    receiver_id: Optional[str] = Field(default=None, description="Peer-to-peer receiver")
    type: Optional[str] = Field(default=None, description="Form-specific transaction type")
    description: Optional[str] = None

    @root_validator(pre=True)
    def _normalize_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        account_id = values.get("account_id")
        sender = values.get("sender_id")
        receiver = values.get("receiver_id")
        inferred_type = values.get("transaction_type") or values.get("type")

        if not account_id:
            if sender:
                account_id = sender
                inferred_type = inferred_type or "cash_out"
            elif receiver:
                account_id = receiver
                inferred_type = inferred_type or "cash_in"
            values["account_id"] = account_id

        if not values.get("client_tx_id"):
            values["client_tx_id"] = values.get("client_transfer_id") or str(uuid.uuid4())

        if inferred_type:
            values["transaction_type"] = inferred_type

        if not values.get("currency"):
            values["currency"] = DEFAULT_CURRENCY

        extra_fields = {
            "sender_id": sender,
            "receiver_id": receiver,
            "form_type": values.get("type"),
            "description": values.get("description"),
        }
        extra_fields = {k: v for k, v in extra_fields.items() if v}
        if extra_fields:
            base_extra = values.get("extra_json") if isinstance(values.get("extra_json"), dict) else {}
            merged = dict(base_extra)
            merged.update(extra_fields)
            values["extra_json"] = merged

        if not values.get("account_id"):
            raise ValueError("account_id or sender_id/receiver_id is required.")

        return values

class TxRecord(TxCreate):
    tx_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_ts: datetime
    event_date: str

class TrainReq(BaseModel):
    data_file_path: str
    target_column: Optional[str] = None

class PredictReq(BaseModel):
    features: Dict[str, Any]

class BatchPredictReq(BaseModel):
    items: List[Dict[str, Any]]

class BulkTxCreateReq(BaseModel):
    items: List[TxCreate]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _merge_extra_payload(
    base: Optional[Dict[str, Any]], additions: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    additions = additions or {}
    merged: Dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    for key, value in additions.items():
        if value is not None:
            merged[key] = value
    return merged or None


def _prepare_tx_records(payload: TxCreate) -> List[TxRecord]:
    """
    Create one or two TxRecord instances depending on whether the payload
    represents a peer-to-peer transfer (sender + receiver) or a single account
    transaction.
    """
    now_utc = datetime.now(timezone.utc)
    event_date_str = now_utc.date().isoformat()

    def _build_record(data: Dict[str, Any]) -> TxRecord:
        return TxRecord(**data, event_ts=now_utc, event_date=event_date_str)

    base_data = payload.dict()
    if payload.sender_id and payload.receiver_id:
        base_data["extra_json"] = _merge_extra_payload(
            base_data.get("extra_json"),
            {
                "p2p_role": "sender",
                "counterparty_account_id": payload.receiver_id,
            },
        )
    primary_record = _build_record(base_data)
    records: List[TxRecord] = [primary_record]

    if payload.sender_id and payload.receiver_id:
        mirror_data = payload.dict()
        mirror_data.update(
            {
                "account_id": payload.receiver_id,
                "transaction_type": "cash_in",
                "client_tx_id": str(uuid.uuid4()),
                "extra_json": _merge_extra_payload(
                    mirror_data.get("extra_json"),
                    {
                        "p2p_role": "receiver",
                        "counterparty_account_id": payload.sender_id,
                        "mirror_of": primary_record.tx_id,
                    },
                ),
            }
        )
        mirror_record = _build_record(mirror_data)
        records.append(mirror_record)

    return records


def _persist_single_record(rec: TxRecord, tx_type: Optional[str]) -> Dict[str, Any]:
    """
    Persist a single transaction either to Cassandra or to the in-memory store.
    """
    try:
        if cassandra_service.available():
            result = cassandra_service.record_transaction(
                account_id=rec.account_id,
                client_tx_id=rec.client_tx_id,
                event_ts=rec.event_ts,
                tx_id=uuid.UUID(rec.tx_id) if isinstance(rec.tx_id, str) else rec.tx_id,
                amount=rec.amount,
                currency=rec.currency,
                merchant=rec.merchant,
                status=rec.status,
                extra_json=rec.extra_json,
                transaction_type=tx_type,
            )
            if isinstance(result, dict) and result.get("status") == "success":
                tx_payload = result.get("transaction")
                if isinstance(tx_payload, dict):
                    _update_training_dataset_from_transaction(tx_payload)
            return result
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable while recording transaction; using memory store.")
    except Exception:
        logger.exception("Failed to write transaction to Cassandra; using memory store.")

    record = rec.dict()
    if isinstance(record.get("event_ts"), datetime):
        record["event_ts"] = record["event_ts"].isoformat()
    if isinstance(record.get("event_date"), (datetime, date)):
        record["event_date"] = (
            record["event_date"].date().isoformat()
            if isinstance(record["event_date"], datetime)
            else record["event_date"].isoformat()
        )
    TX_STORE.append(record)
    _update_training_dataset_from_transaction(record)
    return {"status": "success", "transaction": record}

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@api_router.get("/health")
def health():
    # hỗ trợ cả 2 chỗ lưu: root và models/
    ml_ready = bool(getattr(m5p_model, "is_trained", False)) or m5p_model.has_persisted_model()
    return {"ok": True, "ml_trained": ml_ready, "version": app.version}

@api_router.get("/healthz/liveness")
def liveness():
    return {"ok": True, "service": "alive"}

@api_router.get("/healthz/readiness")
def readiness():
    try:
        _ = getattr(m5p_model, "is_trained", False)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Realtime Transactions (match GUI routes) + extensions
# -----------------------------------------------------------------------------
@api_router.post("/rt/transactions")
def create_transaction(payload: TxCreate):
    records = _prepare_tx_records(payload)
    results: List[Dict[str, Any]] = []
    for rec in records:
        tx_type = _normalize_transaction_type(rec.transaction_type)
        results.append(_persist_single_record(rec, tx_type))
    if len(results) == 1:
        return results[0]
    return {"status": "success", "transactions": results}

@api_router.post("/rt/transactions/bulk")
def create_transactions_bulk(req: BulkTxCreateReq):
    created: List[Dict[str, Any]] = []
    for item in req.items:
        for rec in _prepare_tx_records(item):
            tx_type = _normalize_transaction_type(rec.transaction_type)
            created.append(_persist_single_record(rec, tx_type))
    return {"status": "success", "count": len(created), "items": created}

@api_router.get("/rt/transactions")
def list_transactions(
    account_id: str = Query(...),
    event_date: date = Query(...),
    limit: int = Query(5, ge=1, le=100),
):
    try:
        if cassandra_service.available():
            items = cassandra_service.list_transactions(account_id, event_date, limit)
            return {"account_id": account_id, "event_date": event_date.isoformat(), "items": items}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during list_transactions; using memory store.")
    rows = [
        tx for tx in TX_STORE
        if tx["account_id"] == account_id and tx["event_date"] == event_date.isoformat()
    ]
    rows = sorted(rows, key=lambda x: x["event_ts"], reverse=True)[:limit]
    sanitized: List[Dict[str, Any]] = []
    for tx in rows:
        copy = dict(tx)
        if isinstance(copy.get("event_ts"), datetime):
            copy["event_ts"] = copy["event_ts"].replace(tzinfo=timezone.utc).isoformat()
        event_date_value = copy.get("event_date")
        if isinstance(event_date_value, datetime):
            copy["event_date"] = event_date_value.date().isoformat()
        elif isinstance(event_date_value, date):
            copy["event_date"] = event_date_value.isoformat()
        sanitized.append(copy)
    return {"account_id": account_id, "event_date": event_date.isoformat(), "items": sanitized}


@api_router.get("/rt/transactions/all")
def list_transactions_all(limit: int = Query(500, ge=1, le=5000)):
    try:
        if cassandra_service.available():
            items = cassandra_service.list_all_transactions(limit)
            return {"count": len(items), "items": items}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during all-transactions fetch; using memory store.")
    except Exception:
        logger.exception("Failed to fetch all transactions from Cassandra; using memory store.")

    def _sort_key(tx: Dict[str, Any]) -> str:
        ts_value = tx.get("event_ts")
        if isinstance(ts_value, datetime):
            return ts_value.isoformat()
        return str(ts_value or "")

    rows = sorted(TX_STORE, key=_sort_key, reverse=True)
    limited = rows[:limit]
    sanitized: List[Dict[str, Any]] = []
    for tx in limited:
        copy = dict(tx)
        ts_value = copy.get("event_ts")
        if isinstance(ts_value, datetime):
            copy["event_ts"] = ts_value.replace(tzinfo=timezone.utc).isoformat()
        event_date_value = copy.get("event_date")
        if isinstance(event_date_value, datetime):
            copy["event_date"] = event_date_value.date().isoformat()
        elif isinstance(event_date_value, date):
            copy["event_date"] = event_date_value.isoformat()
        sanitized.append(copy)
    return {"count": len(sanitized), "items": sanitized}

@api_router.get("/rt/transactions/range")
def list_transactions_range(
    account_id: str = Query(...),
    date_from: date = Query(...),
    date_to: date = Query(...),
    limit: int = Query(100, ge=1, le=1000),
):
    dfrom = date_from.isoformat()
    dto   = date_to.isoformat()
    try:
        if cassandra_service.available():
            items = cassandra_service.list_transactions_range(account_id, date_from, date_to, limit)
            return {"account_id": account_id, "from": dfrom, "to": dto, "items": items}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during range fetch; using memory store.")
    rows = [
        tx for tx in TX_STORE
        if tx["account_id"] == account_id and dfrom <= tx["event_date"] <= dto
    ]
    rows = sorted(rows, key=lambda x: (x["event_date"], x["event_ts"]), reverse=True)[:limit]
    sanitized: List[Dict[str, Any]] = []
    for tx in rows:
        copy = dict(tx)
        ts_value = copy.get("event_ts")
        if isinstance(ts_value, datetime):
            copy["event_ts"] = ts_value.replace(tzinfo=timezone.utc).isoformat()
        event_date_value = copy.get("event_date")
        if isinstance(event_date_value, datetime):
            copy["event_date"] = event_date_value.date().isoformat()
        elif isinstance(event_date_value, date):
            copy["event_date"] = event_date_value.isoformat()
        sanitized.append(copy)
    return {"account_id": account_id, "from": dfrom, "to": dto, "items": sanitized}

@api_router.get("/rt/transactions/by-id/{tx_id}")
def get_tx_by_id(tx_id: str):
    try:
        if cassandra_service.available():
            tx = cassandra_service.get_transaction_by_id(tx_id=tx_id)
            if tx:
                return {"transaction": tx}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during get_tx_by_id; using memory store.")
    for tx in TX_STORE:
        if tx["tx_id"] == tx_id:
            return {"transaction": tx}
    raise HTTPException(status_code=404, detail="Transaction not found")

# -----------------------------------------------------------------------------
# ML: Train / Predict (match GUI routes) + extensions
# -----------------------------------------------------------------------------
@api_router.post("/ml/m5p/train")
def train_m5p(req: TrainReq):
    global LAST_TRAIN_METRICS
    try:
        data_path = _resolve_data_path(req.data_file_path)
        with LOCK:
            metrics = m5p_model.train(str(data_path), req.target_column)
            LAST_TRAIN_METRICS = metrics
        return metrics
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {req.data_file_path}")
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/ml/m5p/train/upload")
async def train_m5p_upload(file: UploadFile = File(...), target_column: Optional[str] = Body(None)):
    """
    Upload CSV và train trực tiếp. Lưu file tạm vào DATA_DIR.
    """
    global LAST_TRAIN_METRICS
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        save_path = DATA_DIR / f"uploaded_{file.filename or 'train.csv'}"
        df.to_csv(save_path, index=False)
        with LOCK:
            metrics = m5p_model.train(str(save_path), target_column)
            LAST_TRAIN_METRICS = metrics
        return {"source": "upload", **metrics}
    except Exception as e:
        logger.exception("Training upload failed")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/ml/m5p/status")
def m5p_status():
    model = getattr(m5p_model, "model", None)
    trained = bool(getattr(m5p_model, "is_trained", False))
    info: Dict[str, Any] = {
        "trained": trained,
        "version": app.version,
        "last_train_metrics": LAST_TRAIN_METRICS,
    }
    if model is not None:
        info.update({
            "tree_depth": model.model_.get_tree_depth() if getattr(model, "model_", None) else None,
            "n_leaves": model.model_.get_n_leaves() if getattr(model, "model_", None) else None,
            "n_features_in": model.n_features_in_,
        })
    return info

@api_router.get("/ml/m5p/schema")
def m5p_schema():
    # theo code mặc định, schema dump ra m5p_schema.json ở working dir
    schema_path = m5p_model.get_schema_path()
    if schema_path and schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    fallback = MODEL_DIR / "m5p_schema.json"
    if fallback.exists():
        with open(fallback, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Schema file not found")

@api_router.get("/ml/m5p/feature-mapping")
def feature_mapping():
    model = getattr(m5p_model, "model", None)
    if not model:
        raise HTTPException(status_code=400, detail="Model chưa được train.")
    orig = model.all_feature_names_ or []
    proc = model.processed_feature_names_ or []
    return {"original": orig, "processed": proc}

@api_router.get("/ml/prepare-features")
def prepare_features_from_latest():
    """
    Automatically prepare prediction features from the latest data in CSV.
    This endpoint calculates all engineered features needed for prediction.

    Returns:
        Dict with all required features including lag features and date-based flags
    """
    try:
        dataset_path = _ensure_training_dataset()
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Training dataset not found")

        df = pd.read_csv(dataset_path)

        if len(df) < 7:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 7 days of data for feature calculation. Current: {len(df)} days"
            )

        # Get last row (most recent day)
        last_row = df.iloc[-1]

        # Parse date
        try:
            date_obj = datetime.fromisoformat(str(last_row["date"]))
            day_of_month = date_obj.day
        except Exception:
            day_of_month = 15  # Default

        # Calculate features
        features = {
            # Today's values
            "cash_in_d0": float(last_row["cash_in"]),
            "cash_out_d0": float(last_row["cash_out"]),
            "cash_net_d0": float(last_row["cash_in"] - last_row["cash_out"]),

            # Lag features for cash_in
            "lag1_in": float(df.iloc[-2]["cash_in"]) if len(df) > 1 else 0.0,
            "lag7_in": float(df.iloc[-8]["cash_in"]) if len(df) > 7 else 0.0,
            "roll_mean_7_in": float(df.tail(7)["cash_in"].mean()),

            # Lag features for cash_out
            "lag1_out": float(df.iloc[-2]["cash_out"]) if len(df) > 1 else 0.0,
            "lag7_out": float(df.iloc[-8]["cash_out"]) if len(df) > 7 else 0.0,
            "roll_mean_7_out": float(df.tail(7)["cash_out"].mean()),

            # Date features
            "dow": int(last_row["day_of_week"]),
            "is_weekend": int(last_row["day_of_week"] >= 5),
            "is_month_end": 1 if day_of_month >= 25 else 0,
            "is_payday": 1 if day_of_month == 15 or day_of_month >= 25 else 0,

            # Channel
            "channel": str(last_row.get("channel", "DEFAULT"))
        }

        return {
            "status": "success",
            "features": features,
            "data_date": str(last_row["date"]),
            "data_points_used": len(df)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to prepare features")
        raise HTTPException(status_code=500, detail=f"Feature preparation failed: {str(e)}")


@api_router.post("/ml/m5p/predict")
def predict_m5p(req: PredictReq):
    try:
        with LOCK:
            y = m5p_model.predict(req.features)
        return {"prediction": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/ml/m5p/predict/batch")
def predict_m5p_batch(req: BatchPredictReq):
    """
    Batch predict: input là list các dict feature.
    """
    try:
        preds: List[float] = []
        with LOCK:
            for item in req.items:
                preds.append(m5p_model.predict(item))
        return {"count": len(preds), "predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/ml/predict/all")
def predict_all_targets(req: PredictReq):
    """
    Predict all 6 cash flow targets at once:
    - cash_in_next_day, cash_out_next_day
    - cash_in_h7_sum, cash_out_h7_sum
    - cash_in_next_month_sum, cash_out_next_month_sum

    Input: JSON features according to the training schema.
    Output: Dict with all 6 predictions
    """
    try:
        with LOCK:
            predictions = multi_model.predict_all(req.features)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/spark/trigger-etl")
async def trigger_spark_etl(background_tasks=None):
    """
    Manually trigger Spark ETL pipeline to process daily data.

    This endpoint runs the Spark ETL job that:
    1. Reads cash_daily.csv
    2. Aggregates transactions by date
    3. Engineers features (lag, rolling windows)
    4. Merges with existing training data
    5. Writes to cash_daily_train_realistic.csv
    6. Clears cash_daily.csv
    """
    try:
        import subprocess

        spark_script = Path(__file__).parent.parent / "spark-etl.py"
        if not spark_script.exists():
            raise HTTPException(404, detail=f"Spark ETL script not found: {spark_script}")

        # Run Spark ETL
        result = subprocess.run(
            [
                sys.executable,  # Use current Python interpreter
                str(spark_script),
                "--mode", "local",
                "--local-base", str(DATA_DIR)
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=str(spark_script.parent)
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Spark ETL completed successfully",
                "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(500, detail={
                "message": "Spark ETL failed",
                "error": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            })

    except subprocess.TimeoutExpired:
        raise HTTPException(408, detail="Spark ETL timeout (5 minutes)")
    except Exception as e:
        logger.exception("Spark ETL trigger failed")
        raise HTTPException(500, detail=f"Failed to trigger Spark ETL: {str(e)}")


@api_router.get("/spark/status")
def spark_status():
    """
    Check Spark ETL status and configuration.
    """
    spark_script = Path(__file__).parent.parent / "spark-etl.py"

    return {
        "spark_installed": True,  # PySpark is installed
        "spark_script_exists": spark_script.exists(),
        "spark_script_path": str(spark_script),
        "daily_csv_path": str(DAILY_CSV_PATH),
        "daily_csv_exists": DAILY_CSV_PATH.exists(),
        "training_csv_path": str(TRAINING_DATASET_PATH),
        "training_csv_exists": TRAINING_DATASET_PATH.exists(),
        "scheduler_running": scheduler is not None and scheduler.running if scheduler else False,
    }


@api_router.post("/ml/predict/cash-in")
def predict_cash_in_all(req: PredictReq):
    """
    Predict all cash_in targets: next_day, h7_sum, next_month_sum.

    Returns:
    {
        "next_day": float,
        "h7_sum": float,
        "next_month_sum": float
    }
    """
    try:
        with LOCK:
            predictions = multi_model.predict_cash_in(req.features)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/ml/predict/cash-out")
def predict_cash_out_all(req: PredictReq):
    """
    Predict all cash_out targets: next_day, h7_sum, next_month_sum.

    Returns:
    {
        "next_day": float,
        "h7_sum": float,
        "next_month_sum": float
    }
    """
    try:
        with LOCK:
            predictions = multi_model.predict_cash_out(req.features)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/ml/status")
def multi_model_status():
    """
    Get status of all multi-target models.
    """
    return {
        "ready": multi_model.is_ready(),
        "models": {
            "cash_in_next_day": multi_model.cash_in_next_day.is_trained,
            "cash_out_next_day": multi_model.cash_out_next_day.is_trained,
            "cash_in_h7": multi_model.cash_in_h7.is_trained,
            "cash_out_h7": multi_model.cash_out_h7.is_trained,
            "cash_in_next_month": multi_model.cash_in_next_month.is_trained,
            "cash_out_next_month": multi_model.cash_out_next_month.is_trained,
        },
        "version": app.version,
    }

@api_router.get("/ml/m5p/rules")
def rules():
    try:
        model = m5p_model.model
        if not model:
            raise ValueError("Model chưa được train.")
        return {"rules": model.export_rules()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/ml/m5p/feature-importances")
def feature_importances():
    try:
        model = m5p_model.model
        if not model:
            raise ValueError("Model chưa được train.")
        return {"importances": model.get_feature_importances_with_names()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/ml/m5p/load")
def load_model():
    """
    Cố gắng load lại model từ disk (m5p_model.pkl).
    """
    try:
        with LOCK:
            path = m5p_model.load_from_disk()
        return {"status": "loaded", "path": str(path)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="m5p_model.pkl not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------------------------------------------------------
# KPI (match GUI route) + extensions
# -----------------------------------------------------------------------------
@api_router.post("/kpi/daily/upsert")
def upsert_kpi(
    event_date: date = Query(...),
    metric: str = Query(...),
    value: float = Query(...),
):
    try:
        if cassandra_service.available():
            record = cassandra_service.upsert_kpi(event_date, metric, value)
            return {"status": "upserted", **record}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during KPI upsert; using memory store.")
    key = (event_date.isoformat(), metric)
    KPI_STORE[key] = float(value)
    return {"status": "upserted", "event_date": key[0], "metric": metric, "value": KPI_STORE[key]}

@api_router.get("/kpi/daily/get")
def get_kpi(
    event_date: date = Query(...),
    metric: str = Query(...),
):
    try:
        if cassandra_service.available():
            record = cassandra_service.get_kpi(event_date, metric)
            if record:
                return record
            raise HTTPException(status_code=404, detail="KPI not found")
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during KPI get; using memory store.")
    key = (event_date.isoformat(), metric)
    if key not in KPI_STORE:
        raise HTTPException(status_code=404, detail="KPI not found")
    return {"event_date": key[0], "metric": metric, "value": KPI_STORE[key]}

@api_router.get("/kpi/daily/list")
def list_kpi(
    date_from: date = Query(...),
    date_to: date = Query(...),
):
    dfrom = date_from.isoformat()
    dto   = date_to.isoformat()
    try:
        if cassandra_service.available():
            current = date_from
            items: List[Dict[str, Any]] = []
            while current <= date_to:
                day_items = cassandra_service.list_kpis(current)
                items.extend(day_items)
                current += timedelta(days=1)
            items = sorted(items, key=lambda x: (x["event_date"], x["metric"]))
            return {"from": dfrom, "to": dto, "items": items}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during KPI range list; using memory store.")
    items = [
        {"event_date": d, "metric": m, "value": v}
        for (d, m), v in KPI_STORE.items()
        if dfrom <= d <= dto
    ]
    items = sorted(items, key=lambda x: (x["event_date"], x["metric"]))
    return {"from": dfrom, "to": dto, "items": items}

# -----------------------------------------------------------------------------
# Data utilities (optional but handy)
# -----------------------------------------------------------------------------
@api_router.post("/data/aggregate/daily")
async def aggregate_daily_now():
    """
    Manually trigger the midnight aggregation pipeline.
    Useful for testing or when running in environments without a scheduler.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not running")
    try:
        stats = await scheduler.trigger_now()
        return {"status": stats.get("status", "unknown"), "details": stats}
    except Exception as e:
        logger.exception("Manual aggregation failed")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/preview")
def data_preview(path: str = Query(...), n: int = Query(5, ge=1, le=50)):
    try:
        resolved = _resolve_data_path(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        df = pd.read_csv(resolved, nrows=n)
        return {"path": str(resolved), "head": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.include_router(api_router, prefix="/api")
# Temporary compatibility: expose the same routes at root-level for frontends that
# still call http://host/endpoint without /api prefix.
app.include_router(api_router)

# Dev run:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload)
