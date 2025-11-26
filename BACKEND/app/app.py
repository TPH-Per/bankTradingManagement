# main.py (extended)
import os
import io
import sys
import uuid
import random
import string
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from BACKEND directory (parent of app directory)
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip loading .env
    pass

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Body, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
# Use structured logging if available
try:
    from app.logging_config import setup_logging, logger
    # Setup structured logging
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        use_json=os.getenv("LOG_JSON", "false").lower() == "true"
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("bankTrading")

# Always add file handler for debugging
fh = logging.FileHandler("backend_debug.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

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

# Setup rate limiting
try:
    from app.middleware.rate_limit import setup_rate_limiting, rate_limit
    limiter = setup_rate_limiting(app)
    logger.info("Rate limiting enabled")
except ImportError:
    limiter = None
    rate_limit = lambda x: lambda y: y  # No-op decorator if not available
    logger.warning("Rate limiting not available (slowapi not installed)")

# Setup monitoring middleware
try:
    from app.monitoring import track_request
    import time
    
    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        track_request(request, response, duration)
        return response
    logger.info("Monitoring middleware enabled")
except ImportError:
    logger.warning("Monitoring not available (prometheus-client not installed)")

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
    # Auto-start Docker services (Redis, HDFS)
    logger.info("="*70)
    logger.info("Step 1: Auto-starting Docker services...")
    logger.info("="*70)
    try:
        from pathlib import Path
        from app.auto_start_docker import start_docker_services
        
        # Get project root directory (parent of BACKEND)
        project_root = Path(__file__).parent.parent.parent
        
        # Start Docker services (skip Cassandra if running on Windows)
        start_docker_services(
            project_dir=project_root,
            use_cassandra_docker=False  # Use Windows native Cassandra
        )
    except Exception as e:
        logger.warning("Docker auto-start failed (services may already be running or Docker not available): %s", e)
    
    # Bootstrap multi-target models
    logger.info("="*70)
    logger.info("Step 2: Bootstrapping ML models...")
    logger.info("="*70)
    global scheduler

    dataset = TRAINING_DATASET_PATH
    force = os.getenv("FORCE_RETRAIN", "false").lower() == "true"

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
    
    # Initialize P2P Transaction Service
    try:
        from app.services.transaction_service import P2PTransactionService
        global transaction_service
        transaction_service = P2PTransactionService(cassandra_service)
        logger.info("P2P Transaction Service initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize transaction service: {e}")
        transaction_service = None


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
ACCOUNT_STORE: Dict[str, Dict[str, Any]] = {}  # account_id -> account data
BALANCE_STORE: Dict[str, float] = {}  # account_id -> balance
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


def _update_account_balance_from_transaction(tx: Dict[str, Any]) -> None:
    """
    Update account balance based on transaction.
    For cash_in: add to balance
    For cash_out: subtract from balance
    For P2P transactions: subtract from sender, add to receiver
    """
    try:
        account_id = tx.get("account_id")
        if not account_id:
            logger.warning("Transaction missing account_id, skipping balance update")
            return
        
        amount = float(tx.get("amount", 0.0))
        if amount == 0.0:
            return
        
        transaction_type = tx.get("transaction_type") or tx.get("direction")
        extra_json = tx.get("extra_json") or {}
        
        # Check if this is a P2P transaction
        is_p2p_sender = extra_json.get("p2p_role") == "sender"
        is_p2p_receiver = extra_json.get("p2p_role") == "receiver"
        
        # Determine if this increases or decreases balance
        if is_p2p_sender:
            # Sender: subtract amount
            balance_delta = -abs(amount)
        elif is_p2p_receiver:
            # Receiver: add amount
            balance_delta = abs(amount)
        elif transaction_type and "cash_out" in str(transaction_type).lower():
            # Cash out: subtract
            balance_delta = -abs(amount)
        else:
            # Cash in or default: add
            balance_delta = abs(amount)
        
        # Update balance in Cassandra if available
        if cassandra_service.available():
            try:
                # Get current balance
                current_balance_data = cassandra_service.get_account_balance(account_id)
                current_balance = current_balance_data.get("balance", 0.0) if current_balance_data else 0.0
                new_balance = current_balance + balance_delta
                
                # Update using the update_balance method
                cassandra_service.update_account_balance(
                    account_id=account_id,
                    amount_delta=new_balance,
                    operation="set"
                )
                logger.debug(f"Updated balance for {account_id} in Cassandra: {current_balance} -> {new_balance}")
            except Exception:
                logger.exception("Failed to update balance in Cassandra; using memory store.")
        
        # Update in-memory store
        current_balance = BALANCE_STORE.get(account_id, 0.0)
        new_balance = current_balance + balance_delta
        BALANCE_STORE[account_id] = new_balance
        logger.debug(f"Updated balance for {account_id} in memory: {current_balance} -> {new_balance}")
        
    except Exception:
        logger.exception("Failed to update account balance from transaction.")


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

class AccountCreate(BaseModel):
    account_id: Optional[str] = Field(default=None, description="Account ID (auto-generated if not provided)")
    customer_id: str = Field(..., description="Customer ID (required)")
    currency: str = Field(default="VND", description="Currency code")
    status: str = Field(default="ACTIVE", description="Account status: ACTIVE, LOCKED, CLOSED")
    extra_json: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class AccountUpdate(BaseModel):
    status: Optional[str] = Field(default=None, description="Account status: ACTIVE, LOCKED, CLOSED")
    extra_json: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

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
    logger.info(f"_persist_single_record: account_id={rec.account_id}, tx_id={rec.tx_id}, type={tx_type}, amount={rec.amount}, cassandra_available={cassandra_service.available()}")
    try:
        if cassandra_service.available():
            logger.info("Attempting to save transaction to Cassandra...")
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
            logger.info(f"Cassandra result: {result}")
            if isinstance(result, dict) and result.get("status") == "success":
                tx_payload = result.get("transaction")
                if isinstance(tx_payload, dict):
                    logger.info(f"Updating training dataset and balance for transaction: {tx_payload.get('tx_id')}")
                    _update_training_dataset_from_transaction(tx_payload)
                    # Update account balance
                    _update_account_balance_from_transaction(tx_payload)
            return result
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable while recording transaction; using memory store.")
    except Exception as e:
        logger.exception(f"Failed to write transaction to Cassandra: {e}; using memory store.")

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
    _update_account_balance_from_transaction(record)
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

@api_router.get("/health/detailed")
async def detailed_health():
    """Comprehensive health check with all service status"""
    from fastapi.responses import JSONResponse
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "services": {}
    }
    
    # Check Cassandra
    try:
        if cassandra_service.available():
            session = cassandra_service.session_rt
            if session:
                health_status["services"]["cassandra"] = {"status": "healthy"}
                try:
                    from app.monitoring import update_cassandra_health
                    update_cassandra_health(True)
                except:
                    pass
            else:
                health_status["services"]["cassandra"] = {"status": "unhealthy", "error": "No session"}
        else:
            health_status["services"]["cassandra"] = {"status": "unavailable"}
    except Exception as e:
        health_status["services"]["cassandra"] = {"status": "unhealthy", "error": str(e)}
        try:
            from app.monitoring import update_cassandra_health
            update_cassandra_health(False)
        except:
            pass
    
    # Check HDFS
    try:
        if hdfs_service:
            hdfs_health = hdfs_service.check_health()
            health_status["services"]["hdfs"] = hdfs_health
            try:
                from app.monitoring import update_hdfs_health
                update_hdfs_health(hdfs_health.get("status") == "healthy")
            except:
                pass
        else:
            health_status["services"]["hdfs"] = {"status": "disabled"}
    except Exception as e:
        health_status["services"]["hdfs"] = {"status": "error", "error": str(e)}
        try:
            from app.monitoring import update_hdfs_health
            update_hdfs_health(False)
        except:
            pass
    
    # Check ML Models
    try:
        model_status = {}
        for model_name in ["cash_in_next_day", "cash_out_next_day", "cash_in_h7", 
                          "cash_out_h7", "cash_in_next_month", "cash_out_next_month"]:
            model = getattr(multi_model, model_name, None)
            if model:
                model_status[model_name] = {
                    "status": "loaded" if hasattr(model, "model") and model.model else "not_loaded"
                }
        health_status["services"]["ml_models"] = {
            "status": "healthy" if all(m.get("status") == "loaded" for m in model_status.values()) else "partial",
            "models": model_status
        }
    except Exception as e:
        health_status["services"]["ml_models"] = {"status": "error", "error": str(e)}
    
    # System metrics
    try:
        import psutil
        health_status["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        }
    except ImportError:
        health_status["system"] = {"status": "metrics_unavailable", "note": "psutil not installed"}
    except Exception as e:
        health_status["system"] = {"status": "error", "error": str(e)}
    
    # Check Redis cache
    try:
        from app.cache_service import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            redis_client.ping()
            health_status["services"]["redis"] = {"status": "healthy"}
        else:
            health_status["services"]["redis"] = {"status": "unavailable"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
    
    # Determine overall health: Only Cassandra is required, everything else is optional
    cassandra_healthy = health_status["services"].get("cassandra", {}).get("status") == "healthy"
    
    if cassandra_healthy:
        # Cassandra OK = system is operational
        health_status["status"] = "healthy"
        status_code = 200
    else:
        # Cassandra down = system degraded
        health_status["status"] = "degraded"
        status_code = 503
    
    return JSONResponse(content=health_status, status_code=status_code)

@api_router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from app.monitoring import get_metrics_response
        return get_metrics_response()
    except ImportError:
        from fastapi.responses import Response
        return Response(
            content="# Prometheus metrics not available\n",
            media_type="text/plain"
        )

# -----------------------------------------------------------------------------
# Account Management
# -----------------------------------------------------------------------------
def _generate_bank_account_id() -> str:
    """
    Generate a 7-digit bank account ID in format: 000XXXX
    Prefix: 000
    Suffix: 4 random digits
    """
    # Generate 4 random digits
    suffix = ''.join(random.choices(string.digits, k=4))
    return f"000{suffix}"

@api_router.post("/accounts")
def create_account(payload: AccountCreate):
    """
    Create a new account. If account_id is not provided, it will be auto-generated.
    
    Validation: If customer_id already has an account, a new account will NOT be created.
    """
    try:
        # Check if customer_id already has an account
        existing_customer_accounts = []
        try:
            if cassandra_service.available():
                existing_customer_accounts = cassandra_service.get_accounts_by_customer(payload.customer_id)
            else:
                # Check in memory store
                for acc_id, acc_data in ACCOUNT_STORE.items():
                    if acc_data.get("customer_id") == payload.customer_id:
                        existing_customer_accounts.append({"account_id": acc_id, **acc_data})
        except Exception:
            logger.exception("Failed to check existing customer accounts")
        
        if existing_customer_accounts:
            existing_account_id = existing_customer_accounts[0].get("account_id")
            raise HTTPException(
                status_code=409,
                detail=f"Customer {payload.customer_id} already has an account: {existing_account_id}. Cannot create duplicate account for the same customer."
            )

        # Auto-generate account_id (format: 000XXXX - 7 digits)
        # Do NOT allow user to provide account_id
        account_id = None
        for _ in range(10):  # Try 10 times to find a unique ID
            candidate_id = _generate_bank_account_id()
            # Check uniqueness
            is_unique = True
            try:
                if cassandra_service.available():
                    if cassandra_service.get_account_by_id(candidate_id):
                        is_unique = False
                else:
                    if ACCOUNT_STORE.get(candidate_id):
                        is_unique = False
            except Exception:
                pass
            
            if is_unique:
                account_id = candidate_id
                break
        
        if not account_id:
            # Fallback: generate with timestamp if still no unique ID
            import time
            account_id = f"000{int(time.time()) % 10000:04d}"

        # Check if account already exists (by account_id)
        existing = None
        try:
            if cassandra_service.available():
                existing = cassandra_service.get_account_by_id(account_id)
            else:
                existing = ACCOUNT_STORE.get(account_id)
        except Exception:
            pass

        if existing:
            raise HTTPException(status_code=409, detail=f"Account {account_id} already exists")

        # Create account
        if cassandra_service.available():
            try:
                result = cassandra_service.create_account(
                    account_id=account_id,
                    customer_id=payload.customer_id,
                    currency=payload.currency,
                    status=payload.status,
                    extra_json=payload.extra_json,
                )
                return {"status": "success", "account": result}
            except CassandraUnavailable:
                logger.warning("Cassandra unavailable; using memory store.")
            except Exception:
                logger.exception("Failed to create account in Cassandra; using memory store.")

        # Fallback to memory store
        now_utc = datetime.now(timezone.utc)
        account_data = {
            "account_id": account_id,
            "customer_id": payload.customer_id,
            "currency": payload.currency,
            "status": payload.status,
            "opened_at": now_utc.isoformat(),
            "updated_at": now_utc.isoformat(),
            "extra_json": payload.extra_json or {},
        }
        ACCOUNT_STORE[account_id] = account_data
        return {"status": "success", "account": account_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create account")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/accounts/all")
def get_all_accounts(limit: int = Query(1000, description="Maximum number of accounts to return")):
    """
    Get all accounts (for dashboard statistics).
    Returns accounts from Cassandra if available, otherwise from memory store.
    NOTE: This route MUST be defined BEFORE /accounts/{account_id} to avoid conflict.
    """
    try:
        accounts = []
        
        if cassandra_service.available():
            try:
                accounts = cassandra_service.list_all_accounts(limit=limit)
                logger.info(f"Retrieved {len(accounts)} accounts from Cassandra")
            except Exception as e:
                logger.warning(f"Failed to get accounts from Cassandra: {e}; using memory store.")
        
        # Fallback to memory store
        if not accounts:
            accounts = list(ACCOUNT_STORE.values())
            logger.info(f"Retrieved {len(accounts)} accounts from memory store")
        
        return {"count": len(accounts), "accounts": accounts}
    
    except Exception as e:
        logger.exception("Failed to get all accounts")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/accounts/{account_id}")
def get_account(account_id: str):
    """
    Get account by account_id.
    """
    try:
        if cassandra_service.available():
            try:
                account = cassandra_service.get_account_by_id(account_id)
                if account:
                    return {"account": account}
            except CassandraUnavailable:
                logger.warning("Cassandra unavailable; using memory store.")
            except Exception:
                logger.exception("Failed to get account from Cassandra; using memory store.")

        # Fallback to memory store
        account = ACCOUNT_STORE.get(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
        return {"account": account}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get account")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.patch("/accounts/{account_id}")
def update_account(account_id: str, payload: AccountUpdate):
    """
    Update account status and/or metadata.
    """
    try:
        if cassandra_service.available():
            try:
                result = cassandra_service.update_account(
                    account_id=account_id,
                    status=payload.status,
                    extra_json=payload.extra_json,
                )
                if result:
                    return {"status": "success", "account": result}
                raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
            except CassandraUnavailable:
                logger.warning("Cassandra unavailable; using memory store.")
            except Exception:
                logger.exception("Failed to update account in Cassandra; using memory store.")

        # Fallback to memory store
        account = ACCOUNT_STORE.get(account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

        if payload.status:
            account["status"] = payload.status
        if payload.extra_json:
            account["extra_json"] = {**(account.get("extra_json") or {}), **payload.extra_json}
        account["updated_at"] = datetime.now(timezone.utc).isoformat()

        return {"status": "success", "account": account}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update account")
        raise HTTPException(status_code=500, detail=str(e))





# -----------------------------------------------------------------------------
# Customer Management (compatibility layer - maps to accounts)
# -----------------------------------------------------------------------------
@api_router.get("/customers/search")
def search_customers(
    query: str = Query("", min_length=0, description="Search query (empty = all)"),
    search_type: str = Query("all", enum=["all", "identity", "account", "phone", "email"], description="Type of search: identity (CCCD), account (Account ID), phone, or email"),
):
    """
    Search customers by National ID (CCCD), Account ID, Phone, or Email.
    If query is empty, returns all customers.
    Returns a list of customers (usually 1).
    """
    try:
        results = []
        
        logger.info(f"Search customers: query='{query}', search_type='{search_type}', cassandra_available={cassandra_service.available()}")
        
        # If query is empty, load ALL customers
        if not query or not query.strip():
            logger.info("Empty query - loading all customers")
            # Try to load from Cassandra first
            if cassandra_service.available():
                try:
                    session = cassandra_service.session_rt
                    if session:
                        # Get full account data directly
                        query_str = "SELECT account_id, customer_id, currency, status, opened_at, updated_at, extra_json FROM accounts_by_id LIMIT 100"
                        rows = session.execute(query_str)
                        
                        for row in rows:
                            try:
                                # Parse extra_json
                                import json
                                extra = {}
                                if row.extra_json:
                                    if isinstance(row.extra_json, str):
                                        extra = json.loads(row.extra_json)
                                    elif isinstance(row.extra_json, dict):
                                        extra = row.extra_json
                                
                                results.append({
                                    "account_id": row.account_id,
                                    "customer_id": row.customer_id,
                                    "full_name": extra.get("full_name", row.customer_id),
                                    "email": extra.get("email", ""),
                                    "phone": extra.get("phone", ""),
                                    "national_id": extra.get("national_id", ""),
                                    "status": row.status,
                                    "created_at": row.opened_at.isoformat() if hasattr(row.opened_at, 'isoformat') else str(row.opened_at),
                                })
                            except Exception as e:
                                logger.warning(f"Failed to parse account: {e}")
                                pass
                except Exception as e:
                    logger.warning(f"Failed to load all from Cassandra: {e}")
            
            # Fallback or supplement with memory store
            if not results:
                for acc in ACCOUNT_STORE.values():
                    extra = acc.get("extra_json") or {}
                    results.append({
                        "account_id": acc.get("account_id"),
                        "customer_id": acc.get("customer_id"),
                        "full_name": extra.get("full_name", acc.get("customer_id")),
                        "email": extra.get("email", ""),
                        "phone": extra.get("phone", ""),
                        "national_id": extra.get("national_id", ""),
                        "status": acc.get("status", "inactive"),
                        "created_at": acc.get("opened_at"),
                    })
            
            return {"count": len(results), "items": results}
        
        # 1. Search by National ID (Identity)
        if search_type in ["all", "identity"]:
            if cassandra_service.available():
                try:
                    # Search in customers_by_identity
                    customer_by_identity = cassandra_service.get_customer_by_identity(query)
                    if customer_by_identity:
                        customer_id = customer_by_identity.get("customer_id")
                        if customer_id:
                            accounts = cassandra_service.get_accounts_by_customer(customer_id)
                            for acc in accounts:
                                full_account = cassandra_service.get_account_by_id(acc["account_id"])
                                if full_account:
                                    # Convert to customer format
                                    extra = full_account.get("extra_json") or {}
                                    results.append({
                                        "account_id": full_account.get("account_id"),
                                        "customer_id": full_account.get("customer_id"),
                                        "full_name": extra.get("full_name", full_account.get("customer_id")),
                                        "email": extra.get("email", ""),
                                        "phone": extra.get("phone", ""),
                                        "national_id": extra.get("national_id", ""),
                                        "status": full_account.get("status", "inactive"),
                                        "created_at": full_account.get("opened_at"),
                                    })
                except Exception:
                    pass
        
        # 2. Search by Account ID
        if search_type in ["all", "account"] and not results:
            try:
                account = None
                if cassandra_service.available():
                    account = cassandra_service.get_account_by_id(query)
                else:
                    account = ACCOUNT_STORE.get(query)
                
                if account:
                    extra = account.get("extra_json") or {}
                    results.append({
                        "account_id": account.get("account_id"),
                        "customer_id": account.get("customer_id"),
                        "full_name": extra.get("full_name", account.get("customer_id")),
                        "email": extra.get("email", ""),
                        "phone": extra.get("phone", ""),
                        "national_id": extra.get("national_id", ""),
                        "status": account.get("status", "inactive"),
                        "created_at": account.get("opened_at"),
                    })
            except Exception:
                pass
        
        # 3. Search by Phone
        if search_type in ["all", "phone"] and not results:
            logger.info(f"Searching by phone: {query}")
            if cassandra_service.available():
                try:
                    customer_by_phone = cassandra_service.accounts.get_customer_by_phone(query)
                    logger.info(f"get_customer_by_phone result: {customer_by_phone}")
                    if customer_by_phone:
                        customer_id = customer_by_phone.get("customer_id")
                        logger.info(f"Customer ID: {customer_id}")
                        if customer_id:
                            accounts = cassandra_service.get_accounts_by_customer(customer_id)
                            logger.info(f"Found {len(accounts) if accounts else 0} accounts")
                            for acc in accounts:
                                full_account = cassandra_service.get_account_by_id(acc["account_id"])
                                if full_account:
                                    extra = full_account.get("extra_json") or {}
                                    results.append({
                                        "account_id": full_account.get("account_id"),
                                        "customer_id": full_account.get("customer_id"),
                                        "full_name": extra.get("full_name", full_account.get("customer_id")),
                                        "email": extra.get("email", ""),
                                        "phone": extra.get("phone", ""),
                                        "national_id": extra.get("national_id", ""),
                                        "status": full_account.get("status", "inactive"),
                                        "created_at": full_account.get("opened_at"),
                                    })
                    else:
                        logger.warning(f"Phone {query} not found in customers_by_phone")
                except Exception as e:
                    logger.exception(f"Error searching by phone: {e}")
                    pass
        
        # 4. Search by Email
        if search_type in ["all", "email"] and not results:
            if cassandra_service.available():
                try:
                    customer_by_email = cassandra_service.accounts.get_customer_by_email(query)
                    if customer_by_email:
                        customer_id = customer_by_email.get("customer_id")
                        if customer_id:
                            accounts = cassandra_service.get_accounts_by_customer(customer_id)
                            for acc in accounts:
                                full_account = cassandra_service.get_account_by_id(acc["account_id"])
                                if full_account:
                                    extra = full_account.get("extra_json") or {}
                                    results.append({
                                        "account_id": full_account.get("account_id"),
                                        "customer_id": full_account.get("customer_id"),
                                        "full_name": extra.get("full_name", full_account.get("customer_id")),
                                        "email": extra.get("email", ""),
                                        "phone": extra.get("phone", ""),
                                        "national_id": extra.get("national_id", ""),
                                        "status": full_account.get("status", "inactive"),
                                        "created_at": full_account.get("opened_at"),
                                    })
                except Exception:
                    pass

        # 5. Always check memory store as fallback if no results yet
        # This ensures newly created accounts (not yet in Cassandra) are found
        if not results:
             for acc in ACCOUNT_STORE.values():
                extra = acc.get("extra_json") or {}
                
                match = False
                if search_type == "identity":
                    if extra.get("national_id") == query:
                        match = True
                elif search_type == "account":
                    if acc.get("account_id") == query:
                        match = True
                elif search_type == "phone":
                    if extra.get("phone") == query:
                        match = True
                elif search_type == "email":
                    if extra.get("email") == query:
                        match = True
                else: # all
                    if (acc.get("account_id") == query or 
                        extra.get("national_id") == query or
                        extra.get("phone") == query or
                        extra.get("email") == query):
                        match = True
                        
                if match:
                     results.append({
                        "account_id": acc.get("account_id"),
                        "customer_id": acc.get("customer_id"),
                        "full_name": extra.get("full_name", acc.get("customer_id")),
                        "email": extra.get("email", ""),
                        "phone": extra.get("phone", ""),
                        "national_id": extra.get("national_id", ""),
                        "status": acc.get("status", "inactive"),
                        "created_at": acc.get("opened_at"),
                     })

        return {"count": len(results), "items": results}

    except Exception as e:
        logger.exception("Failed to search customers")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/customers/{account_id}")
def get_customer(account_id: str):
    """
    Get customer by account_id (for compatibility).
    """
    try:
        account = None
        if cassandra_service.available():
            try:
                account = cassandra_service.get_account_by_id(account_id)
            except Exception:
                pass

        if not account:
            account = ACCOUNT_STORE.get(account_id)

        if not account:
            raise HTTPException(status_code=404, detail=f"Customer/Account {account_id} not found")

        # Convert account to customer format
        extra = account.get("extra_json") or {}
        customer = {
            "account_id": account.get("account_id"),
            "customer_id": account.get("customer_id"),
            "full_name": extra.get("full_name", account.get("customer_id")),
            "email": extra.get("email", ""),
            "national_id": extra.get("national_id", ""),
            "status": account.get("status", "inactive"),
            "created_at": account.get("opened_at"),
        }
        return customer

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get customer")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/customers")
def create_customer(payload: Dict[str, Any]):
    """
    Create a customer (creates an account with customer info in extra_json).
    For compatibility with frontend.
    
    Validation: If customer_id already exists (based on national_id or email), 
    a new account will NOT be created.
    """
    try:
        # Extract customer data
        full_name = payload.get("full_name", "")
        email = payload.get("email", "") or None  # Treat empty string as None
        national_id = payload.get("national_id", "") or None  # Treat empty string as None
        
        # Validate national_id if provided (not empty)
        if national_id:
            national_id = national_id.strip()  # Remove whitespace
            if not national_id.isdigit():
                raise HTTPException(
                    status_code=400,
                    detail="CMND/CCCD phải là số."
                )
            if len(national_id) != 10 and len(national_id) != 12:
                raise HTTPException(
                    status_code=400,
                    detail="CMND/CCCD phải có độ dài đúng 10 hoặc 12 ký tự."
                )
        
        # Extract phone
        phone = payload.get("phone", "") or None  # Treat empty string as None

        
        # Check if national_id already exists
        if national_id:
            try:
                if cassandra_service.available():
                    existing_by_identity = cassandra_service.get_customer_by_identity(national_id)
                    if existing_by_identity:
                        raise HTTPException(
                            status_code=409,
                            detail=f"CMND/CCCD '{national_id}' đã được sử dụng bởi khách hàng khác. Không thể tạo tài khoản trùng lặp."
                        )
            except HTTPException:
                raise
            except Exception:
                logger.exception("Failed to check existing national_id in Cassandra")
        
        
        # Check if email already exists (using indexed table)
        if email:
            try:
                if cassandra_service.available():
                    existing_by_email = cassandra_service.accounts.get_customer_by_email(email)
                    if existing_by_email:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Email '{email}' đã được sử dụng bởi khách hàng {existing_by_email.get('customer_id')}. Không thể tạo tài khoản trùng lặp."
                        )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Failed to check email in Cassandra: {e}")
            
            # Also check in memory store as fallback
            for acc_id, acc_data in ACCOUNT_STORE.items():
                extra = acc_data.get("extra_json") or {}
                if extra.get("email") == email:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Email '{email}' đã được sử dụng bởi tài khoản {acc_id}. Không thể tạo tài khoản trùng lặp."
                    )
        
        # Check if phone already exists (using indexed table)
        if phone:
            try:
                if cassandra_service.available():
                    logger.info(f"Checking duplicate phone in Cassandra: {phone}")
                    existing_by_phone = cassandra_service.accounts.get_customer_by_phone(phone)
                    if existing_by_phone:
                        logger.warning(f"Duplicate phone found: {phone} -> {existing_by_phone.get('customer_id')}")
                        raise HTTPException(
                            status_code=409,
                            detail=f"Số điện thoại '{phone}' đã được sử dụng bởi khách hàng {existing_by_phone.get('customer_id')}. Không thể tạo tài khoản trùng lặp."
                        )
                    else:
                        logger.info(f"Phone {phone} is unique in Cassandra.")
                else:
                    logger.warning("Cassandra NOT available during phone check!")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Failed to check phone in Cassandra: {e}")
            
            # Also check in memory store as fallback
            for acc_id, acc_data in ACCOUNT_STORE.items():
                extra = acc_data.get("extra_json") or {}
                if extra.get("phone") == phone:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Số điện thoại '{phone}' đã được sử dụng bởi tài khoản {acc_id}. Không thể tạo tài khoản trùng lặp."
                    )
        
        
        
        
        # Generate customer_id from national_id or email
        customer_id = national_id or email.split("@")[0] if email else str(uuid.uuid4())
        
        # Auto-generate account_id (format: 000XXXX)
        # Do not allow user to provide account_id
        account_id = _generate_bank_account_id()
        for _ in range(10):
            # Check uniqueness
            existing_acc = None
            if cassandra_service.available():
                try:
                    existing_acc = cassandra_service.get_account_by_id(account_id)
                except:
                    pass
            else:
                existing_acc = ACCOUNT_STORE.get(account_id)
            
            if not existing_acc:
                break
            
            # If exists, generate a new one
            account_id = _generate_bank_account_id()

        # Create account with customer info in extra_json
        account_payload = AccountCreate(
            account_id=account_id,
            customer_id=customer_id,
            currency="VND",
            status="ACTIVE",
            extra_json={
                "full_name": full_name,
                "email": email,
                "national_id": national_id,
                "phone": phone,
                "dob": payload.get("dob", ""),
            },
        )

        return create_account(account_payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create customer")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.patch("/customers/{account_id}")
def update_customer(account_id: str, payload: Dict[str, Any]):
    """
    Update customer (updates account extra_json).
    For compatibility with frontend.
    """
    try:
        # Get current account
        account = None
        if cassandra_service.available():
            try:
                account = cassandra_service.get_account_by_id(account_id)
            except Exception:
                pass

        if not account:
            account = ACCOUNT_STORE.get(account_id)

        if not account:
            raise HTTPException(status_code=404, detail=f"Customer/Account {account_id} not found")

        # Update extra_json with customer data
        extra = account.get("extra_json") or {}
        if "full_name" in payload:
            extra["full_name"] = payload["full_name"]
        if "email" in payload:
            extra["email"] = payload["email"]
        if "national_id" in payload:
            extra["national_id"] = payload["national_id"]
        if "status" in payload:
            # Map frontend status to account status
            status_map = {
                "active": "ACTIVE",
                "inactive": "LOCKED",
                "suspended": "LOCKED",
            }
            account_status = status_map.get(payload["status"], "ACTIVE")
        else:
            account_status = account.get("status")

        account_update = AccountUpdate(
            status=account_status,
            extra_json=extra,
        )

        return update_account(account_id, account_update)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update customer")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Account Balance Management
# -----------------------------------------------------------------------------
@api_router.get("/accounts/{account_id}/balance")
@rate_limit("100/minute")
def get_account_balance(request: Request, account_id: str):
    """
    Get account balance.
    Results are cached for 5 minutes.
    """
    try:
        # Try to use cache
        from app.cache_service import cache_result
        cached_get_balance = cache_result(ttl=300, key_prefix="account")(_get_account_balance_impl)
        return cached_get_balance(account_id)
    except Exception as e:
        logger.exception("Failed to get account balance")
        raise HTTPException(status_code=500, detail=str(e))

def _get_account_balance_impl(account_id: str):
    """Internal implementation for getting account balance"""
    try:
        if cassandra_service.available():
            try:
                balance = cassandra_service.get_account_balance(account_id)
                if balance:
                    return balance
            except Exception:
                logger.exception("Failed to get balance from Cassandra; using memory store.")
        
        # Fallback to memory store
        balance_value = BALANCE_STORE.get(account_id, 0.0)
        return {
            "account_id": account_id,
            "balance": balance_value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Failed to get account balance")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/accounts/{account_id}/balance/update")
def update_account_balance_endpoint(
    account_id: str,
    amount: float = Body(...),
    operation: str = Body("add", description="Operation: 'add', 'subtract', or 'set'"),
):
    """
    Update account balance manually.
    """
    try:
        if cassandra_service.available():
            try:
                # Check for negative balance before updating in Cassandra
                # Note: This is a check-then-act race condition, but acceptable for this level of consistency.
                # Ideally, use LWT (Lightweight Transactions) in Cassandra or atomic operations if supported.
                current_balance_data = cassandra_service.get_account_balance(account_id)
                current_balance = current_balance_data.get("balance", 0.0) if current_balance_data else 0.0
                
                if operation == "subtract" and current_balance < abs(amount):
                     raise HTTPException(
                        status_code=400,
                        detail=f"Số dư không đủ. Số dư hiện tại: {current_balance:,.0f} VND, Số tiền cần trừ: {abs(amount):,.0f} VND"
                    )
                if operation == "set" and amount < 0:
                     raise HTTPException(
                        status_code=400,
                        detail="Số dư không thể là số âm."
                    )

                result = cassandra_service.update_account_balance(
                    account_id=account_id,
                    amount_delta=amount,
                    operation=operation,
                )
                return {"status": "success", **result}
            except HTTPException:
                raise
            except Exception:
                logger.exception("Failed to update balance in Cassandra; using memory store.")
        
        # Fallback to memory store
        current_balance = BALANCE_STORE.get(account_id, 0.0)
        
        # Calculate new balance
        if operation == "set":
            new_balance = amount
        elif operation == "subtract":
            new_balance = current_balance - abs(amount)
        else:  # "add"
            new_balance = current_balance + abs(amount)
            
        # Check for negative balance
        if new_balance < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Số dư không đủ. Số dư hiện tại: {current_balance:,.0f} VND, Số dư sau khi thực hiện: {new_balance:,.0f} VND. Không thể thực hiện giao dịch."
            )
            
        if operation == "set":
            pass # already calculated
        elif operation == "subtract":
            pass # already calculated
        else:  # "add"
            pass # already calculated
        
        BALANCE_STORE[account_id] = new_balance
        logger.info(f"Updated balance for {account_id}: {current_balance} -> {new_balance} (operation: {operation})")
        return {
            "status": "success",
            "account_id": account_id,
            "balance": new_balance,
            "previous_balance": current_balance,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Failed to update account balance")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/customers/{account_id}/stats/summary")
def get_customer_stats(
    account_id: str,
    period: str = Query("month", description="Period: 'day', 'week', 'month', 'year'"),
):
    """
    Get customer statistics summary.
    For compatibility with frontend.
    """
    try:
        # Get account
        account = None
        if cassandra_service.available():
            try:
                account = cassandra_service.get_account_by_id(account_id)
            except Exception:
                pass

        if not account:
            account = ACCOUNT_STORE.get(account_id)

        if not account:
            raise HTTPException(status_code=404, detail=f"Customer/Account {account_id} not found")

        # Calculate date range based on period
        today = datetime.now(timezone.utc).date()
        if period == "day":
            start_date = today
        elif period == "week":
            start_date = today - timedelta(days=7)
        elif period == "month":
            start_date = today.replace(day=1)  # First day of current month
        elif period == "year":
            start_date = today.replace(month=1, day=1)  # First day of current year
        else:
            start_date = today.replace(day=1)  # Default to month

        # Get transactions for the period
        total_in = 0.0
        total_out = 0.0
        transaction_count = 0

        if cassandra_service.available():
            try:
                transactions = cassandra_service.list_transactions_range(
                    account_id, start_date, today, limit=10000
                )
                for tx in transactions:
                    amount = float(tx.get("amount", 0))
                    direction = tx.get("direction", "").upper()
                    if direction == "CREDIT" or amount > 0:
                        total_in += abs(amount)
                    elif direction == "DEBIT" or amount < 0:
                        total_out += abs(amount)
                    transaction_count += 1
            except Exception:
                pass

        # Fallback to memory store
        if transaction_count == 0:
            for tx in TX_STORE:
                if tx.get("account_id") == account_id:
                    tx_date_str = tx.get("event_date")
                    if tx_date_str:
                        try:
                            tx_date = datetime.fromisoformat(tx_date_str.replace("Z", "+00:00")).date()
                            if start_date <= tx_date <= today:
                                amount = float(tx.get("amount", 0))
                                if amount > 0:
                                    total_in += amount
                                else:
                                    total_out += abs(amount)
                                transaction_count += 1
                        except Exception:
                            pass

        # Get balance from account_balances table
        balance = 0.0
        if cassandra_service.available():
            try:
                balance_data = cassandra_service.get_account_balance(account_id)
                if balance_data:
                    balance = balance_data.get("balance", 0.0)
                else:
                    # If no balance record exists, calculate from transactions
                    balance = total_in - total_out
            except Exception:
                # Fallback: calculate from transactions
                balance = total_in - total_out
        else:
            # Use memory store balance or calculate
            balance = BALANCE_STORE.get(account_id, total_in - total_out)

        return {
            "account_id": account_id,
            "period": period,
            "total_in": total_in,
            "total_out": total_out,
            "transaction_count": transaction_count,
            "balance": balance,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get customer stats")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Realtime Transactions (match GUI routes) + extensions
# -----------------------------------------------------------------------------
@api_router.post("/rt/transactions")
def create_transaction(payload: TxCreate):
    """
    Create a transaction. For P2P transactions, validates that both sender and receiver accounts exist.
    """
    logger.info(f"Creating transaction: account_id={payload.account_id}, type={payload.transaction_type}, amount={payload.amount}, sender={payload.sender_id}, receiver={payload.receiver_id}")
    
    # Validate P2P transaction: both accounts must exist
    if payload.sender_id and payload.receiver_id:
        # Check if sender account exists
        sender_account = None
        try:
            if cassandra_service.available():
                sender_account = cassandra_service.get_account_by_id(payload.sender_id)
            else:
                sender_account = ACCOUNT_STORE.get(payload.sender_id)
        except Exception:
            logger.exception("Failed to check sender account")
        
        if not sender_account:
            raise HTTPException(
                status_code=404,
                detail=f"Tài khoản người gửi không tồn tại: {payload.sender_id}"
            )
        
        # Check if receiver account exists
        receiver_account = None
        try:
            if cassandra_service.available():
                receiver_account = cassandra_service.get_account_by_id(payload.receiver_id)
            else:
                receiver_account = ACCOUNT_STORE.get(payload.receiver_id)
        except Exception:
            logger.exception("Failed to check receiver account")
        
        if not receiver_account:
            raise HTTPException(
                status_code=404,
                detail=f"Tài khoản người nhận không tồn tại: {payload.receiver_id}"
            )
        
        # Check if sender and receiver are different
        if payload.sender_id == payload.receiver_id:
            raise HTTPException(
                status_code=400,
                detail="Không thể chuyển tiền cho chính mình. Tài khoản người gửi và người nhận phải khác nhau."
            )
        
        # Check sender balance if it's a debit transaction
        try:
            sender_balance_data = None
            if cassandra_service.available():
                sender_balance_data = cassandra_service.get_account_balance(payload.sender_id)
            else:
                sender_balance = BALANCE_STORE.get(payload.sender_id, 0.0)
                sender_balance_data = {"balance": sender_balance}
            
            sender_balance = sender_balance_data.get("balance", 0.0) if sender_balance_data else 0.0
            if sender_balance < abs(payload.amount):
                raise HTTPException(
                    status_code=400,
                    detail=f"Số dư không đủ. Số dư hiện tại: {sender_balance:,.0f} VND, Số tiền cần chuyển: {abs(payload.amount):,.0f} VND"
                )
        except HTTPException:
            raise
        except Exception:
            logger.warning("Could not verify sender balance, proceeding anyway")
    
    # Validate single account transaction: account must exist
    elif payload.account_id:
        account = None
        try:
            if cassandra_service.available():
                account = cassandra_service.get_account_by_id(payload.account_id)
            else:
                account = ACCOUNT_STORE.get(payload.account_id)
        except Exception:
            logger.exception("Failed to check account")
        
        if not account:
            logger.warning(f"Account {payload.account_id} not found. Transaction will be saved but account validation failed.")
            # Don't raise error - allow transaction to proceed but log warning
            # This allows treasury transactions to work even if account doesn't exist yet
        
        # Check balance for cash_out transactions
        tx_type = payload.transaction_type
        if tx_type and tx_type.lower() in ["cash_out", "withdrawal", "debit"]:
            try:
                current_balance_data = None
                if cassandra_service.available():
                    current_balance_data = cassandra_service.get_account_balance(payload.account_id)
                else:
                    current_balance = BALANCE_STORE.get(payload.account_id, 0.0)
                    current_balance_data = {"balance": current_balance}
                
                current_balance = current_balance_data.get("balance", 0.0) if current_balance_data else 0.0
                
                if current_balance < abs(payload.amount):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Số dư không đủ. Số dư hiện tại: {current_balance:,.0f} VND, Số tiền cần rút: {abs(payload.amount):,.0f} VND"
                    )
            except HTTPException:
                raise
            except Exception:
                logger.warning("Could not verify account balance for cash_out, proceeding anyway")
    
    # Create transfer record for P2P transactions
    transfer_result = None
    if payload.sender_id and payload.receiver_id:
        try:
            if cassandra_service.available():
                transfer_result = cassandra_service.transfers.create_transfer(
                    from_account=payload.sender_id,
                    to_account=payload.receiver_id,
                    amount=abs(payload.amount),
                    currency=payload.currency,
                    status="SETTLED",
                    client_transfer_id=payload.client_tx_id,
                    extra_json={
                        "description": payload.description or f"P2P transfer from {payload.sender_id} to {payload.receiver_id}",
                        "transaction_type": "p2p",
                    }
                )
                logger.info(f"Created transfer record: {transfer_result.get('transfer_id')}")
                
                # Record P2P transaction in tracking tables
                try:
                    from uuid import UUID
                    transfer_id_str = transfer_result.get("transfer_id")
                    transfer_id = UUID(transfer_id_str) if transfer_id_str else None
                    
                    # Get customer IDs from accounts
                    sender_customer_id = sender_account.get("customer_id") if sender_account else None
                    receiver_customer_id = receiver_account.get("customer_id") if receiver_account else None
                    
                    cassandra_service.p2p_transactions.record_p2p_transaction(
                        from_account=payload.sender_id,
                        to_account=payload.receiver_id,
                        from_customer_id=sender_customer_id,
                        to_customer_id=receiver_customer_id,
                        event_ts=datetime.now(timezone.utc),
                        tx_id=UUID(payload.client_tx_id) if payload.client_tx_id else UUID(int=0),
                        transfer_id=transfer_id,
                        amount=abs(payload.amount),
                        currency=payload.currency,
                        status="SETTLED",
                        extra_json={
                            "description": payload.description or f"P2P transfer",
                            "transaction_type": "p2p",
                        }
                    )
                    logger.info(f"Recorded P2P transaction in tracking tables")
                except Exception as e:
                    logger.exception(f"Failed to record P2P transaction in tracking tables: {e}")
        except Exception:
            logger.exception("Failed to create transfer record; continuing with transaction creation")
    
    # Create transaction records
    records = _prepare_tx_records(payload)
    logger.info(f"Prepared {len(records)} transaction record(s)")
    results: List[Dict[str, Any]] = []
    for rec in records:
        tx_type = _normalize_transaction_type(rec.transaction_type)
        logger.info(f"Persisting transaction: tx_id={rec.tx_id}, account_id={rec.account_id}, type={tx_type}, amount={rec.amount}")
        # Add transfer_id to transaction if available
        if transfer_result and transfer_result.get("status") == "success":
            if not rec.extra_json:
                rec.extra_json = {}
            rec.extra_json["transfer_id"] = transfer_result.get("transfer_id")
        result = _persist_single_record(rec, tx_type)
        logger.info(f"Transaction persisted: status={result.get('status')}, tx_id={result.get('transaction', {}).get('tx_id')}")
        results.append(result)
    if len(results) == 1:
        final_result = results[0]
        logger.info(f"Transaction creation completed: status={final_result.get('status')}")
        return final_result
    logger.info(f"Multiple transactions created: {len(results)} records")
    return {"status": "success", "transactions": results, "transfer": transfer_result}

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
    event_date: Optional[date] = Query(None, description="Transaction date (defaults to today)"),
    limit: int = Query(5, ge=1, le=100),
):
    try:
        # Default to today if event_date not provided
        if event_date is None:
            event_date = datetime.now(timezone.utc).date()
        
        if cassandra_service.available():
            items = cassandra_service.list_transactions(account_id, event_date, limit)
            return {"account_id": account_id, "event_date": event_date.isoformat(), "items": items}
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during list_transactions; using memory store.")
    
    # Fallback to memory store
    rows = [
        tx for tx in TX_STORE
        if tx["account_id"] == account_id and tx.get("event_date") == event_date.isoformat()
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

@api_router.get("/rt/transfers")
def list_transfers(limit: int = Query(100, ge=1, le=1000)):
    """
    List all transfers (P2P transactions).
    Tries to fetch from transfers_by_id table, with fallback to extracting from transactions.
    """
    try:
        if cassandra_service.available():
            # Try to get from transfers table
            try:
                items = cassandra_service.transfers.list_transfers(limit)
                logger.info(f"Fetched {len(items)} transfers from transfers_by_id table")
                if items:
                    return {"count": len(items), "items": items}
            except Exception as e:
                logger.warning(f"Failed to fetch from transfers_by_id: {e}, trying fallback method")
                logger.exception("Exception details for transfers_by_id query:")
            
            # Fallback: Get all transactions and filter for P2P ones with transfer_id
            try:
                all_txs = cassandra_service.list_all_transactions(limit * 2)  # Get more to filter
                transfers_map = {}
                
                for tx in all_txs:
                    extra_json = tx.get("extra_json") or {}
                    if isinstance(extra_json, str):
                        try:
                            import json
                            extra_json = json.loads(extra_json)
                        except:
                            extra_json = {}
                    
                    transfer_id = extra_json.get("transfer_id")
                    p2p_role = extra_json.get("p2p_role")
                    
                    # Only process sender records to avoid duplicates
                    if transfer_id and p2p_role == "sender":
                        if transfer_id not in transfers_map:
                            counterparty = extra_json.get("counterparty_account_id")
                            event_ts = tx.get("event_ts")
                            if isinstance(event_ts, str):
                                created_at = event_ts
                            elif event_ts:
                                created_at = event_ts.isoformat() if hasattr(event_ts, 'isoformat') else str(event_ts)
                            else:
                                created_at = tx.get("event_date") or datetime.now(timezone.utc).isoformat()
                            
                            transfers_map[transfer_id] = {
                                "transfer_id": transfer_id,
                                "from_account": tx.get("account_id"),
                                "to_account": counterparty,
                                "amount": abs(float(tx.get("amount", 0))),
                                "currency": tx.get("currency", "VND"),
                                "created_at": created_at,
                                "status": "SETTLED",
                                "extra_json": extra_json,
                            }
                
                items = list(transfers_map.values())
                # Sort by created_at descending
                items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
                items = items[:limit]
                
                logger.info(f"Fetched {len(items)} transfers from transactions (fallback)")
                return {"count": len(items), "items": items}
            except Exception as e2:
                logger.exception(f"Fallback method also failed: {e2}")
    except CassandraUnavailable:
        logger.warning("Cassandra unavailable during transfers fetch.")
    except Exception as e:
        logger.exception(f"Failed to fetch transfers: {e}")
    
    logger.warning("Returning empty transfers list")
    return {"count": 0, "items": []}


@api_router.get("/rt/p2p/account-pair/history")
def get_p2p_account_pair_history(
    account_id1: str = Query(..., description="First account ID"),
    account_id2: str = Query(..., description="Second account ID"),
    month_yyyymm: int = Query(..., description="Month in YYYYMM format (e.g., 202511)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
):
    """
    Get P2P transaction history between two accounts for a specific month.
    Example: /rt/p2p/account-pair/history?account_id1=ACC_001&account_id2=ACC_999&month_yyyymm=202511
    """
    try:
        if not cassandra_service.available():
            raise HTTPException(
                status_code=503,
                detail="Cassandra service not available"
            )
        
        items = cassandra_service.p2p_transactions.get_account_pair_history(
            account_id1=account_id1,
            account_id2=account_id2,
            month_yyyymm=month_yyyymm,
            limit=limit
        )
        
        return {
            "account_id1": account_id1,
            "account_id2": account_id2,
            "month_yyyymm": month_yyyymm,
            "count": len(items),
            "items": items
        }
    except CassandraUnavailable:
        raise HTTPException(status_code=503, detail="Cassandra service unavailable")
    except Exception as e:
        logger.exception("Failed to get P2P account pair history")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/rt/p2p/directional/history")
def get_p2p_directional_history(
    from_account: str = Query(..., description="Source account ID"),
    to_account: str = Query(..., description="Destination account ID"),
    event_date: str = Query(..., description="Date in YYYY-MM-DD format"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
):
    """
    Get directional transaction history from account A to account B on a specific date.
    Example: /rt/p2p/directional/history?from_account=ACC_001&to_account=ACC_999&event_date=2025-11-23
    """
    try:
        if not cassandra_service.available():
            raise HTTPException(
                status_code=503,
                detail="Cassandra service not available"
            )
        
        items = cassandra_service.p2p_transactions.get_directional_history(
            from_account=from_account,
            to_account=to_account,
            event_date=event_date,
            limit=limit
        )
        
        return {
            "from_account": from_account,
            "to_account": to_account,
            "event_date": event_date,
            "count": len(items),
            "items": items
        }
    except CassandraUnavailable:
        raise HTTPException(status_code=503, detail="Cassandra service unavailable")
    except Exception as e:
        logger.exception("Failed to get P2P directional history")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/rt/p2p/customer-pair/history")
def get_p2p_customer_pair_history(
    customer_id1: str = Query(..., description="First customer ID"),
    customer_id2: str = Query(..., description="Second customer ID"),
    month_yyyymm: int = Query(..., description="Month in YYYYMM format (e.g., 202511)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
):
    """
    Get P2P transaction history between two customers for a specific month.
    Example: /rt/p2p/customer-pair/history?customer_id1=CUST_001&customer_id2=CUST_999&month_yyyymm=202511
    """
    try:
        if not cassandra_service.available():
            raise HTTPException(
                status_code=503,
                detail="Cassandra service not available"
            )
        
        items = cassandra_service.p2p_transactions.get_customer_pair_history(
            customer_id1=customer_id1,
            customer_id2=customer_id2,
            month_yyyymm=month_yyyymm,
            limit=limit
        )
        
        return {
            "customer_id1": customer_id1,
            "customer_id2": customer_id2,
            "month_yyyymm": month_yyyymm,
            "count": len(items),
            "items": items
        }
    except CassandraUnavailable:
        raise HTTPException(status_code=503, detail="Cassandra service unavailable")
    except Exception as e:
        logger.exception("Failed to get P2P customer pair history")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/accounts/{account_id}/statement")
def get_account_statement(
    account_id: str,
    date_from: date = Query(..., description="Start date (YYYY-MM-DD)"),
    date_to: date = Query(..., description="End date (YYYY-MM-DD)"),
):
    """
    Get account statement (Sao kê) with daily balance snapshots.
    """
    try:
        if date_from > date_to:
            raise HTTPException(status_code=400, detail="date_from must be <= date_to")
        
        # Get snapshots from database
        snapshots = []
        if cassandra_service.available():
            try:
                snapshots = cassandra_service.balance_snapshots.get_snapshot_range(
                    account_id, date_from, date_to
                )
            except Exception:
                logger.exception("Failed to get snapshots from Cassandra")
        
        # If no snapshots, calculate from transactions
        if not snapshots:
            try:
                if cassandra_service.available():
                    transactions = cassandra_service.list_transactions_range(
                        account_id, date_from, date_to, limit=10000
                    )
                else:
                    # Fallback to memory store
                    transactions = [
                        tx for tx in TX_STORE
                        if tx.get("account_id") == account_id
                        and date_from <= datetime.fromisoformat(tx.get("event_date", "2000-01-01")).date() <= date_to
                    ]
                
                # Group transactions by day and calculate snapshots
                from collections import defaultdict
                daily_txs = defaultdict(lambda: {"debits": [], "credits": []})
                
                for tx in transactions:
                    tx_date = tx.get("event_date")
                    if isinstance(tx_date, str):
                        tx_date = datetime.fromisoformat(tx_date.split("T")[0]).date()
                    elif isinstance(tx_date, datetime):
                        tx_date = tx_date.date()
                    
                    amount = float(tx.get("amount", 0))
                    if amount < 0:
                        daily_txs[tx_date]["debits"].append(abs(amount))
                    else:
                        daily_txs[tx_date]["credits"].append(amount)
                
                # Calculate snapshots for each day
                current_balance = 0.0
                if cassandra_service.available():
                    try:
                        balance_data = cassandra_service.get_account_balance(account_id)
                        current_balance = balance_data.get("balance", 0.0) if balance_data else 0.0
                    except:
                        current_balance = BALANCE_STORE.get(account_id, 0.0)
                else:
                    current_balance = BALANCE_STORE.get(account_id, 0.0)
                
                # Work backwards from today to calculate opening balances
                snapshots = []
                for day in sorted(daily_txs.keys(), reverse=True):
                    day_txs = daily_txs[day]
                    total_debit = sum(day_txs["debits"])
                    total_credit = sum(day_txs["credits"])
                    num_tx = len(day_txs["debits"]) + len(day_txs["credits"])
                    
                    balance_close = current_balance
                    balance_open = balance_close - total_credit + total_debit
                    current_balance = balance_open
                    
                    snapshots.append({
                        "account_id": account_id,
                        "day": day.isoformat(),
                        "balance_open": balance_open,
                        "balance_close": balance_close,
                        "total_debit": total_debit,
                        "total_credit": total_credit,
                        "num_tx": num_tx,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
                
                snapshots.sort(key=lambda x: x["day"], reverse=True)
            except Exception:
                logger.exception("Failed to calculate snapshots from transactions")
        
        return {
            "account_id": account_id,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "snapshots": snapshots,
            "count": len(snapshots),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get account statement")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/accounts/{account_id}/statement/generate")
def generate_daily_snapshot(
    account_id: str,
    day: date = Query(..., description="Date to generate snapshot for (YYYY-MM-DD)"),
):
    """
    Generate a daily balance snapshot for an account.
    Typically called at end of day to create statement records.
    """
    try:
        # Get all transactions for the day
        if cassandra_service.available():
            transactions = cassandra_service.list_transactions(account_id, day, limit=10000)
        else:
            transactions = [
                tx for tx in TX_STORE
                if tx.get("account_id") == account_id
                and tx.get("event_date") == day.isoformat()
            ]
        
        # Calculate totals
        total_debit = 0.0
        total_credit = 0.0
        for tx in transactions:
            amount = float(tx.get("amount", 0))
            if amount < 0:
                total_debit += abs(amount)
            else:
                total_credit += amount
        
        num_tx = len(transactions)
        
        # Get opening balance (previous day's closing balance)
        balance_open = 0.0
        if day > date.today() - timedelta(days=365):  # Only check if within reasonable range
            prev_day = day - timedelta(days=1)
            if cassandra_service.available():
                try:
                    prev_snapshot = cassandra_service.balance_snapshots.get_snapshot(account_id, prev_day)
                    if prev_snapshot:
                        balance_open = prev_snapshot.get("balance_close", 0.0)
                except:
                    pass
        
        # If no previous snapshot, get current balance and work backwards
        if balance_open == 0.0:
            if cassandra_service.available():
                try:
                    balance_data = cassandra_service.get_account_balance(account_id)
                    balance_open = balance_data.get("balance", 0.0) if balance_data else 0.0
                except:
                    balance_open = BALANCE_STORE.get(account_id, 0.0)
            else:
                balance_open = BALANCE_STORE.get(account_id, 0.0)
            
            # Adjust for today's transactions if generating for today
            if day == date.today():
                balance_open = balance_open - total_credit + total_debit
        
        balance_close = balance_open + total_credit - total_debit
        
        # Create snapshot
        if cassandra_service.available():
            snapshot = cassandra_service.balance_snapshots.create_snapshot(
                account_id=account_id,
                day=day,
                balance_open=balance_open,
                balance_close=balance_close,
                total_debit=total_debit,
                total_credit=total_credit,
                num_tx=num_tx,
            )
            return {"status": "success", "snapshot": snapshot}
        else:
            return {
                "status": "success",
                "snapshot": {
                    "account_id": account_id,
                    "day": day.isoformat(),
                    "balance_open": balance_open,
                    "balance_close": balance_close,
                    "total_debit": total_debit,
                    "total_credit": total_credit,
                    "num_tx": num_tx,
                },
                "message": "Cassandra unavailable; snapshot calculated but not persisted",
            }
    except Exception as e:
        logger.exception("Failed to generate daily snapshot")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dashboard/stats")
@rate_limit("60/minute")
def get_dashboard_stats(
    request: Request,
    period: str = Query("month", description="Period: 'day', 'week', 'month', 'year'"),
):
    """
    Get dashboard statistics from Cassandra.
    Cached for 1 minute to reduce database load.
    Returns: total_revenue, customer_count, transaction_count, system_status
    """
    try:
        # Try to use cache
        from app.cache_service import cache_result
        cached_get_stats = cache_result(ttl=60, key_prefix="dashboard")(_get_dashboard_stats_impl)
        return cached_get_stats(period)
    except Exception as e:
        logger.exception("Failed to get dashboard stats")
        raise HTTPException(status_code=500, detail=str(e))

def _get_dashboard_stats_impl(period: str):
    """Internal implementation for getting dashboard stats"""
    try:
        today = datetime.now(timezone.utc).date()
        
        # Calculate date range
        if period == "day":
            start_date = today
        elif period == "week":
            start_date = today - timedelta(days=7)
        elif period == "month":
            start_date = today.replace(day=1)
        elif period == "year":
            start_date = today.replace(month=1, day=1)
        else:
            start_date = today.replace(day=1)
        
        total_revenue = 0.0
        transaction_count = 0
        customer_count = 0
        recent_transactions = []
        
        if cassandra_service.available():
            try:
                # Get all accounts count
                all_accounts = cassandra_service.list_all_accounts(limit=10000)
                customer_count = len(all_accounts)
                
                # Get all transactions for the period
                all_transactions = cassandra_service.list_all_transactions(limit=10000)
                
                for tx in all_transactions:
                    tx_date = tx.get("event_date")
                    if isinstance(tx_date, str):
                        try:
                            tx_date = datetime.fromisoformat(tx_date.split("T")[0]).date()
                        except:
                            continue
                    elif isinstance(tx_date, datetime):
                        tx_date = tx_date.date()
                    else:
                        continue
                    
                    # Filter by date range
                    if start_date <= tx_date <= today:
                        amount = float(tx.get("amount", 0))
                        direction = tx.get("direction", "").upper()
                        
                        # Count revenue (only positive amounts / credits)
                        if direction == "CREDIT" or amount > 0:
                            total_revenue += abs(amount)
                        
                        transaction_count += 1
                        
                        # Collect recent transactions (last 10)
                        if len(recent_transactions) < 10:
                            recent_transactions.append(tx)
                
                # Sort recent transactions by date
                recent_transactions.sort(
                    key=lambda x: x.get("event_ts") or x.get("event_date") or "",
                    reverse=True
                )
                recent_transactions = recent_transactions[:10]
                
            except Exception:
                logger.exception("Failed to fetch dashboard stats from Cassandra")
        
        # Fallback to memory store
        if customer_count == 0:
            customer_count = len(ACCOUNT_STORE)
        
        if transaction_count == 0:
            for tx in TX_STORE:
                tx_date_str = tx.get("event_date")
                if tx_date_str:
                    try:
                        tx_date = datetime.fromisoformat(tx_date_str.split("T")[0]).date()
                        if start_date <= tx_date <= today:
                            amount = float(tx.get("amount", 0))
                            if amount > 0:
                                total_revenue += amount
                            transaction_count += 1
                            
                            if len(recent_transactions) < 10:
                                recent_transactions.append(tx)
                    except:
                        pass
        
        # Calculate system status (uptime percentage - simplified)
        system_status = 99.8  # Can be enhanced with actual uptime tracking
        
        return {
            "period": period,
            "total_revenue": total_revenue,
            "customer_count": customer_count,
            "transaction_count": transaction_count,
            "system_status": system_status,
            "recent_transactions": recent_transactions[:5],  # Top 5 most recent
        }
    except Exception as e:
        logger.exception("Failed to get dashboard stats")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Monthly KPI Management
# -----------------------------------------------------------------------------
@api_router.post("/kpi/monthly/company")
@rate_limit("60/minute")
def upsert_company_kpi(
    request: Request,
    month_yyyymm: int = Body(..., description="Month in YYYYMM format (e.g., 202411)"),
    metric: str = Body(..., description="Metric name (e.g., total_revenue, num_transactions)"),
    value: float = Body(..., description="Metric value"),
):
    """Create or update a company-wide monthly KPI."""
    try:
        if not cassandra_service.available():
            raise HTTPException(status_code=503, detail="Cassandra not available")
        
        result = cassandra_service.monthly_kpis.upsert_company_kpi(month_yyyymm, metric, value)
        return result
    except Exception as e:
        logger.exception("Failed to upsert company KPI")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/kpi/monthly/company")
@rate_limit("60/minute")
def get_company_kpi(
    request: Request,
    month_yyyymm: int = Query(..., description="Month in YYYYMM format"),
    metric: str = Query(None, description="Specific metric (optional, returns all if not provided)"),
):
    """Get company-wide monthly KPI(s)."""
    try:
        if not cassandra_service.available():
            raise HTTPException(status_code=503, detail="Cassandra not available")
        
        if metric:
            result = cassandra_service.monthly_kpis.get_company_kpi(month_yyyymm, metric)
            if not result:
                raise HTTPException(status_code=404, detail=f"KPI not found: {metric} for {month_yyyymm}")
            return result
        else:
            results = cassandra_service.monthly_kpis.list_company_kpis(month_yyyymm)
            return {"month_yyyymm": month_yyyymm, "kpis": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get company KPI")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/kpi/monthly/company/trend")
@rate_limit("60/minute")
def get_company_kpi_trend(
    request: Request,
    metric: str = Query(..., description="Metric name"),
    start_month: int = Query(..., description="Start month in YYYYMM format"),
    end_month: int = Query(..., description="End month in YYYYMM format"),
):
    """Get company KPI trend over a range of months."""
    try:
        if not cassandra_service.available():
            raise HTTPException(status_code=503, detail="Cassandra not available")
        
        results = cassandra_service.monthly_kpis.get_company_kpi_trend(metric, start_month, end_month)
        return {"metric": metric, "trend": results}
    except Exception as e:
        logger.exception("Failed to get company KPI trend")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/kpi/monthly/account")
@rate_limit("60/minute")
def upsert_account_kpi(
    request: Request,
    account_id: str = Body(..., description="Account ID"),
    month_yyyymm: int = Body(..., description="Month in YYYYMM format"),
    metric: str = Body(..., description="Metric name"),
    value: float = Body(..., description="Metric value"),
):
    """Create or update a per-account monthly KPI."""
    try:
        if not cassandra_service.available():
            raise HTTPException(status_code=503, detail="Cassandra not available")
        
        result = cassandra_service.monthly_kpis.upsert_account_kpi(account_id, month_yyyymm, metric, value)
        return result
    except Exception as e:
        logger.exception("Failed to upsert account KPI")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/kpi/monthly/account")
@rate_limit("60/minute")
def get_account_kpi(
    request: Request,
    account_id: str = Query(..., description="Account ID"),
    month_yyyymm: int = Query(..., description="Month in YYYYMM format"),
    metric: str = Query(None, description="Specific metric (optional, returns all if not provided)"),
):
    """Get per-account monthly KPI(s)."""
    try:
        if not cassandra_service.available():
            raise HTTPException(status_code=503, detail="Cassandra not available")
        
        if metric:
            result = cassandra_service.monthly_kpis.get_account_kpi(account_id, month_yyyymm, metric)
            if not result:
                raise HTTPException(status_code=404, detail=f"KPI not found: {metric} for account {account_id} in {month_yyyymm}")
            return result
        else:
            results = cassandra_service.monthly_kpis.list_account_kpis(account_id, month_yyyymm)
            return {"account_id": account_id, "month_yyyymm": month_yyyymm, "kpis": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get account KPI")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/kpi/monthly/account/trend")
@rate_limit("60/minute")
def get_account_kpi_trend(
    request: Request,
    account_id: str = Query(..., description="Account ID"),
    metric: str = Query(..., description="Metric name"),
    start_month: int = Query(..., description="Start month in YYYYMM format"),
    end_month: int = Query(..., description="End month in YYYYMM format"),
):
    """Get account KPI trend over a range of months."""
    try:
        if not cassandra_service.available():
            raise HTTPException(status_code=503, detail="Cassandra not available")
        
        results = cassandra_service.monthly_kpis.get_account_kpi_trend(account_id, metric, start_month, end_month)
        return {"account_id": account_id, "metric": metric, "trend": results}
    except Exception as e:
        logger.exception("Failed to get account KPI trend")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/stats")
def get_reports_stats(
    period: str = Query("month", description="Period: '30days', '90days', 'year'"),
):
    """
    Get comprehensive reports statistics from Cassandra.
    Returns: total_volume, transaction_count, active_customers, success_rate,
             monthly_volume_data, transaction_type_breakdown
    """
    try:
        today = datetime.now(timezone.utc).date()
        
        # Calculate date range
        if period == "30days":
            start_date = today - timedelta(days=30)
            months_back = 1
        elif period == "90days":
            start_date = today - timedelta(days=90)
            months_back = 3
        elif period == "year":
            start_date = today.replace(month=1, day=1)
            months_back = 12
        else:
            start_date = today - timedelta(days=30)
            months_back = 1
        
        total_volume = 0.0
        total_in = 0.0
        total_out = 0.0
        transaction_count = 0
        successful_tx = 0
        active_customers = set()
        
        # Monthly volume data (for chart)
        monthly_data = {}
        
        # Transaction type breakdown
        type_breakdown = {
            "P2P": 0,
            "cash_in": 0,
            "cash_out": 0,
            "other": 0,
        }
        
        if cassandra_service.available():
            try:
                # Get all transactions
                all_transactions = cassandra_service.list_all_transactions(limit=50000)
                
                for tx in all_transactions:
                    tx_date = tx.get("event_date")
                    if isinstance(tx_date, str):
                        try:
                            tx_date = datetime.fromisoformat(tx_date.split("T")[0]).date()
                        except:
                            continue
                    elif isinstance(tx_date, datetime):
                        tx_date = tx_date.date()
                    else:
                        continue
                    
                    # Filter by date range
                    if start_date <= tx_date <= today:
                        amount = float(tx.get("amount", 0))
                        direction = tx.get("direction", "").upper()
                        transaction_type = tx.get("transaction_type", "").lower()
                        extra_json = tx.get("extra_json") or {}
                        if isinstance(extra_json, str):
                            try:
                                import json
                                extra_json = json.loads(extra_json)
                            except:
                                extra_json = {}
                        
                        # Count volume (absolute value)
                        total_volume += abs(amount)
                        if direction == "CREDIT" or amount > 0:
                            total_in += abs(amount)
                        else:
                            total_out += abs(amount)
                        
                        transaction_count += 1
                        
                        # Track active customers
                        account_id = tx.get("account_id")
                        if account_id:
                            active_customers.add(account_id)
                        
                        # Count by transaction type
                        if extra_json.get("p2p_role") or transaction_type == "p2p":
                            type_breakdown["P2P"] += 1
                        elif transaction_type == "cash_in":
                            type_breakdown["cash_in"] += 1
                        elif transaction_type == "cash_out":
                            type_breakdown["cash_out"] += 1
                        else:
                            type_breakdown["other"] += 1
                        
                        # Track success rate
                        status = tx.get("status", "").upper()
                        if status in ["COMPLETED", "SETTLED", "SUCCESS"]:
                            successful_tx += 1
                        
                        # Monthly volume aggregation
                        month_key = tx_date.strftime("%Y-%m")
                        if month_key not in monthly_data:
                            monthly_data[month_key] = 0.0
                        monthly_data[month_key] += abs(amount)
                
            except Exception:
                logger.exception("Failed to fetch reports stats from Cassandra")
        
        # Fallback to memory store
        if transaction_count == 0:
            for tx in TX_STORE:
                tx_date_str = tx.get("event_date")
                if tx_date_str:
                    try:
                        tx_date = datetime.fromisoformat(tx_date_str.split("T")[0]).date()
                        if start_date <= tx_date <= today:
                            amount = float(tx.get("amount", 0))
                            total_volume += abs(amount)
                            if amount > 0:
                                total_in += amount
                            else:
                                total_out += abs(amount)
                            transaction_count += 1
                            
                            account_id = tx.get("account_id")
                            if account_id:
                                active_customers.add(account_id)
                            
                            # Simple type detection
                            tx_type = tx.get("type", "").lower()
                            if "p2p" in tx_type:
                                type_breakdown["P2P"] += 1
                            elif "cash_in" in tx_type:
                                type_breakdown["cash_in"] += 1
                            elif "cash_out" in tx_type:
                                type_breakdown["cash_out"] += 1
                            else:
                                type_breakdown["other"] += 1
                            
                            status = tx.get("status", "").upper()
                            if status in ["COMPLETED", "SETTLED", "SUCCESS"]:
                                successful_tx += 1
                            
                            month_key = tx_date.strftime("%Y-%m")
                            if month_key not in monthly_data:
                                monthly_data[month_key] = 0.0
                            monthly_data[month_key] += abs(amount)
                    except:
                        pass
        
        # Calculate success rate
        success_rate = (successful_tx / transaction_count * 100) if transaction_count > 0 else 0.0
        
        # Format monthly data for chart (last N months)
        monthly_labels = []
        monthly_values = []
        sorted_months = sorted(monthly_data.keys())
        for month_key in sorted_months[-months_back:]:
            month_name = datetime.strptime(month_key, "%Y-%m").strftime("Tháng %m")
            monthly_labels.append(month_name)
            monthly_values.append(monthly_data[month_key])
        
        return {
            "period": period,
            "total_volume": total_volume,
            "total_in": total_in,
            "total_out": total_out,
            "transaction_count": transaction_count,
            "active_customers": len(active_customers),
            "success_rate": round(success_rate, 2),
            "monthly_data": {
                "labels": monthly_labels,
                "values": monthly_values,
            },
            "transaction_types": type_breakdown,
        }
    except Exception as e:
        logger.exception("Failed to get reports stats")
        raise HTTPException(status_code=500, detail=str(e))

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
@rate_limit("30/minute")
def predict_cash_in_all(request: Request, req: PredictReq):
    """
    Predict all cash_in targets: next_day, h7_sum, next_month_sum.
    Results are cached for 1 hour.

    Returns:
    {
        "next_day": float,
        "h7_sum": float,
        "next_month_sum": float
    }
    """
    try:
        # Try to use cache
        from app.cache_service import cache_result
        cached_predict = cache_result(ttl=3600, key_prefix="ml")(_predict_cash_in_impl)
        return cached_predict(req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _predict_cash_in_impl(features: Dict[str, Any]):
    """Internal implementation for cash-in prediction"""
    with LOCK:
        return multi_model.predict_cash_in(features)

@api_router.post("/ml/predict/cash-out")
@rate_limit("30/minute")
def predict_cash_out_all(request: Request, req: PredictReq):
    """
    Predict all cash_out targets: next_day, h7_sum, next_month_sum.
    Results are cached for 1 hour.

    Returns:
    {
        "next_day": float,
        "h7_sum": float,
        "next_month_sum": float
    }
    """
    try:
        # Try to use cache
        from app.cache_service import cache_result
        cached_predict = cache_result(ttl=3600, key_prefix="ml")(_predict_cash_out_impl)
        return cached_predict(req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _predict_cash_out_impl(features: Dict[str, Any]):
    """Internal implementation for cash-out prediction"""
    with LOCK:
        return multi_model.predict_cash_out(features)

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

# -----------------------------------------------------------------------------
# HDFS Operations - Initialize service and define routes BEFORE including router
# -----------------------------------------------------------------------------
try:
    from app.hdfs_service import create_hdfs_service, HDFSService
    import os
    
    # Initialize HDFS service (optional, can be None if HDFS not available)
    hdfs_service: Optional[HDFSService] = None
    if os.getenv("HDFS_ENABLED", "false").lower() == "true":
        try:
            hdfs_service = create_hdfs_service(
                namenode_host=os.getenv("HDFS_NAMENODE_HOST", "localhost"),
                namenode_port=int(os.getenv("HDFS_NAMENODE_PORT", "9870")),
                base_path=os.getenv("HDFS_BASE_PATH", "/banktrading")
            )
            if hdfs_service:
                logger.info("HDFS service initialized successfully")
        except Exception as e:
            logger.warning(f"HDFS service not available: {e}")
except ImportError:
    hdfs_service = None
    logger.warning("HDFS service not available (hdfs_service module not found)")

# Define HDFS routes before including router
@api_router.get("/hdfs/health")
def hdfs_health():
    """Check HDFS connection and health"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        health = hdfs_service.check_health()
        return health
    except Exception as e:
        logger.exception("Error checking HDFS health")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/hdfs/list")
def hdfs_list_directory(path: str = Query("/", description="HDFS path to list")):
    """List files and directories in HDFS path"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        files = hdfs_service.list_directory(path)
        return {
            "status": "success",
            "path": path,
            "items": files,
            "count": len(files)
        }
    except Exception as e:
        logger.exception(f"Error listing HDFS directory {path}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/hdfs/status")
def hdfs_file_status(path: str = Query(..., description="HDFS file path")):
    """Get file/directory status in HDFS"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        status = hdfs_service.get_file_status(path)
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        return {"status": "success", **status}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting HDFS file status {path}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/hdfs/mkdir")
def hdfs_create_directory(path: str = Query(..., description="HDFS directory path")):
    """Create directory in HDFS"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        result = hdfs_service.create_directory(path)
        return result
    except Exception as e:
        logger.exception(f"Error creating HDFS directory {path}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/hdfs/upload")
async def hdfs_upload_file(
    file: UploadFile = File(..., description="File to upload"),
    hdfs_path: str = Query(..., description="HDFS destination path")
):
    """Upload file to HDFS"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Use filename if hdfs_path is a directory
        if hdfs_path.endswith('/'):
            hdfs_path = hdfs_path.rstrip('/') + '/' + file.filename
        
        # Upload to HDFS
        result = hdfs_service.upload_file_from_bytes(file_content, hdfs_path, file_size)
        return result
    except Exception as e:
        logger.exception(f"Error uploading file to HDFS")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/hdfs/upload-local")
async def hdfs_upload_local_file(
    hdfs_path: str = Query(..., description="HDFS destination path"),
    local_file_path: str = Query(..., description="Local file path to upload")
):
    """Upload local file to HDFS (server-side file)"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        result = hdfs_service.upload_file(local_file_path, hdfs_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error uploading file to HDFS")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/hdfs/download")
async def hdfs_download_file(
    hdfs_path: str = Query(..., description="HDFS file path")
):
    """Download file from HDFS (returns file directly)"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        file_content = hdfs_service.download_file_to_bytes(hdfs_path)
        
        # Get filename from path
        filename = hdfs_path.split('/')[-1] or 'download'
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type='application/octet-stream',
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.exception(f"Error downloading file from HDFS")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/hdfs/download-local")
def hdfs_download_file_to_local(
    hdfs_path: str = Query(..., description="HDFS file path"),
    local_file_path: str = Query(..., description="Local destination path")
):
    """Download file from HDFS to local (server-side)"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        result = hdfs_service.download_file(hdfs_path, local_file_path)
        return result
    except Exception as e:
        logger.exception(f"Error downloading file from HDFS")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/hdfs/delete")
def hdfs_delete_file(
    path: str = Query(..., description="HDFS file/directory path"),
    recursive: bool = Query(False, description="Delete recursively for directories")
):
    """Delete file or directory from HDFS"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        result = hdfs_service.delete_file(path, recursive=recursive)
        return result
    except Exception as e:
        logger.exception(f"Error deleting from HDFS")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/hdfs/size")
def hdfs_directory_size(path: str = Query("/", description="HDFS directory path")):
    """Get directory size and file count"""
    if not hdfs_service:
        raise HTTPException(status_code=503, detail="HDFS service not available")
    
    try:
        result = hdfs_service.get_directory_size(path)
        return {"status": "success", **result}
    except Exception as e:
        logger.exception(f"Error getting HDFS directory size {path}")
        raise HTTPException(status_code=500, detail=str(e))

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
