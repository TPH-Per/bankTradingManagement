import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

try:
    from cassandra.auth import PlainTextAuthProvider  # type: ignore
    from cassandra.cluster import Cluster, Session  # type: ignore
    from cassandra.query import PreparedStatement, SimpleStatement  # type: ignore
except ImportError:  # pragma: no cover - driver optional for local dev
    PlainTextAuthProvider = None  # type: ignore
    Cluster = None  # type: ignore
    Session = None  # type: ignore
    PreparedStatement = None  # type: ignore
    SimpleStatement = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from cassandra.auth import PlainTextAuthProvider as PlainTextAuthProviderType
    from cassandra.cluster import Cluster as ClusterType, Session as SessionType
    from cassandra.query import PreparedStatement as PreparedStatementType, SimpleStatement as SimpleStatementType
else:  # pragma: no cover - fallback for runtime without driver
    PlainTextAuthProviderType = Any
    ClusterType = Any
    SessionType = Any
    PreparedStatementType = Any
    SimpleStatementType = Any

logger = logging.getLogger("bankTrading.cassandra")


class CassandraUnavailable(RuntimeError):
    """Raised when Cassandra operations are requested but the driver/session is unavailable."""


@dataclass
class CassandraConfig:
    contact_points: Sequence[str]
    port: int
    username: Optional[str]
    password: Optional[str]
    rt_keyspace: str
    audit_keyspace: str
    enabled: bool


def _parse_contact_points(raw: Optional[str]) -> Sequence[str]:
    if not raw:
        return ["127.0.0.1"]
    if raw.startswith("jdbc:"):
        # Strip jdbc prefix if provided (e.g., jdbc:cassandra://host:port)
        raw = raw.replace("jdbc:", "", 1)
        raw = raw.replace("cassandra://", "", 1)
    host_part = raw
    if "://" in raw:
        host_part = raw.split("://", 1)[1]
    # Support 127.0.0.1:9042 style or comma separated
    items = [item.strip() for item in host_part.split(",") if item.strip()]
    cleaned: List[str] = []
    for item in items:
        if item:
            cleaned.append(item)
    return cleaned or ["127.0.0.1"]


def load_config() -> CassandraConfig:
    contact_points = _parse_contact_points(os.environ.get("CASSANDRA_CONTACT_POINTS"))
    port = int(os.environ.get("CASSANDRA_PORT", "9042"))
    username = os.environ.get("CASSANDRA_USERNAME")
    password = os.environ.get("CASSANDRA_PASSWORD")
    rt_keyspace = os.environ.get("CASSANDRA_KEYSPACE_RT", "pldt_rt")
    audit_keyspace = os.environ.get("CASSANDRA_KEYSPACE_AUDIT", "pldt_audit")
    enabled_env = os.environ.get("ENABLE_CASSANDRA", "").lower()
    disabled_env = os.environ.get("DISABLE_CASSANDRA", "").lower()
    enabled = True
    if disabled_env in {"1", "true", "yes"}:
        enabled = False
    elif enabled_env in {"0", "false", "no"}:
        enabled = False
    return CassandraConfig(
        contact_points=contact_points,
        port=port,
        username=username,
        password=password,
        rt_keyspace=rt_keyspace,
        audit_keyspace=audit_keyspace,
        enabled=enabled,
    )


def _normalize_amount(amount: float) -> Decimal:
    return Decimal(str(amount))


def _json_dumps(obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return None


def _json_loads(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    return None


def _cassandra_ts(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class CassandraService:
    """
    Thin wrapper around cassandra-driver to provide the minimal persistence
    primitives used by the FastAPI application.
    """

    def __init__(self, config: Optional[CassandraConfig] = None):
        self.config = config or load_config()
        self.cluster: Optional[ClusterType] = None
        self.session_rt: Optional[SessionType] = None
        self.session_audit: Optional[SessionType] = None
        self._prepared: Dict[str, PreparedStatementType] = {}
        if self.config.enabled and Cluster is not None:
            self._connect()
        else:
            if not self.config.enabled:
                logger.info("Cassandra integration disabled via environment.")
            elif Cluster is None:
                logger.warning("cassandra-driver not installed; running in memory-only mode.")

    # ------------------------------------------------------------------ Connection
    def _connect(self) -> None:
        try:
            if Cluster is None:
                raise CassandraUnavailable("cassandra-driver not installed.")

            auth_provider: Optional[PlainTextAuthProviderType] = None
            if self.config.username and self.config.password and PlainTextAuthProvider:
                auth_provider = PlainTextAuthProvider(
                    username=self.config.username,
                    password=self.config.password,
                )

            contact_points: List[str] = []
            for cp in self.config.contact_points:
                if ":" in cp:
                    host, _, port_str = cp.partition(":")
                    contact_points.append(host.strip())
                    if port_str.strip().isdigit():
                        self.config.port = int(port_str.strip())
                else:
                    contact_points.append(cp)

            cluster_kwargs = {
                "contact_points": contact_points,
                "port": self.config.port,
            }
            if auth_provider:
                cluster_kwargs["auth_provider"] = auth_provider

            cluster_instance = Cluster(**cluster_kwargs)  # type: ignore[arg-type]
            if cluster_instance is None:
                raise CassandraUnavailable("Unable to instantiate Cassandra cluster.")

            session_rt = cluster_instance.connect(self.config.rt_keyspace)
            session_audit = cluster_instance.connect(self.config.audit_keyspace)

            self.cluster = cluster_instance
            self.session_rt = session_rt
            self.session_audit = session_audit
            logger.info(
                "Connected to Cassandra at %s:%s (keyspace=%s)",
                contact_points,
                self.config.port,
                self.config.rt_keyspace,
            )
            self._prepare_statements()
        except Exception:
            logger.exception("Failed to connect to Cassandra. Falling back to in-memory store.")
            self.cluster = None
            self.session_rt = None
            self.session_audit = None

    def available(self) -> bool:
        return self.session_rt is not None

    def _prepare_statements(self) -> None:
        if not self.session_rt:
            return
        self._prepared["insert_tx"] = self.session_rt.prepare(
            """
            INSERT INTO tx_by_account_day (
                account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        self._prepared["insert_dedup"] = self.session_rt.prepare(
            """
            INSERT INTO client_tx_dedup (account_id, client_tx_id, tx_id, created_at)
            VALUES (?, ?, ?, toTimestamp(now()))
            IF NOT EXISTS
            """
        )
        self._prepared["select_dedup"] = self.session_rt.prepare(
            """
            SELECT tx_id, created_at FROM client_tx_dedup
            WHERE account_id=? AND client_tx_id=?
            """
        )
        self._prepared["select_tx_by_day"] = self.session_rt.prepare(
            """
            SELECT account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json
            FROM tx_by_account_day
            WHERE account_id=? AND event_date=?
            LIMIT ?
            """
        )
        self._prepared["select_tx_by_id"] = self.session_rt.prepare(
            """
            SELECT account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json
            FROM tx_by_account_day
            WHERE account_id=? AND event_date=? AND tx_id=?
            ALLOW FILTERING
            """
        )
        self._prepared["select_tx_by_id_only"] = self.session_rt.prepare(
            """
            SELECT account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json
            FROM tx_by_account_day
            WHERE tx_id=?
            ALLOW FILTERING
            """
        )
        self._prepared["upsert_kpi_daily"] = self.session_rt.prepare(
            """
            INSERT INTO kpi_daily (event_date, metric, value, updated_at)
            VALUES (?, ?, ?, ?)
            """
        )
        self._prepared["select_kpi_daily"] = self.session_rt.prepare(
            """
            SELECT metric, value, updated_at
            FROM kpi_daily
            WHERE event_date=? AND metric=?
            """
        )
        self._prepared["select_kpi_daily_all"] = self.session_rt.prepare(
            """
            SELECT metric, value, updated_at
            FROM kpi_daily
            WHERE event_date=?
            """
        )

    # ------------------------------------------------------------------ Transactions
    def record_transaction(
        self,
        account_id: str,
        client_tx_id: str,
        event_ts: datetime,
        tx_id,
        amount: float,
        currency: str,
        merchant: Optional[str],
        status: Optional[str],
        extra_json: Optional[Dict[str, Any]],
        transaction_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.session_rt or "insert_tx" not in self._prepared:
            raise CassandraUnavailable("Cassandra session not available.")

        event_ts = _cassandra_ts(event_ts)
        event_date = event_ts.date()
        normalized_type: Optional[str] = None
        if transaction_type:
            cleaned = str(transaction_type).strip().lower()
            if cleaned in {"cash_in", "cashin", "in", "deposit"}:
                normalized_type = "cash_in"
            elif cleaned in {"cash_out", "cashout", "out", "withdraw", "withdrawal"}:
                normalized_type = "cash_out"
            elif cleaned:
                normalized_type = cleaned

        dedup_stmt = self._prepared["insert_dedup"]
        result = self.session_rt.execute(dedup_stmt, (account_id, client_tx_id, tx_id))
        applied = result.one()
        if applied is None:
            raise RuntimeError("Unexpected empty result from Cassandra dedup insert.")

        if not applied.applied:  # type: ignore[attr-defined]
            existing = applied.tx_id  # type: ignore[attr-defined]
            created_at = getattr(applied, "created_at", None)
            existing_event_date = created_at.date() if isinstance(created_at, datetime) else None
            transaction = None
            if existing_event_date:
                transaction = self.get_transaction_by_id(
                    account_id=account_id,
                    tx_id=str(existing),
                    event_date=existing_event_date,
                )
            else:
                transaction = self.get_transaction_by_id(tx_id=str(existing))
            return {
                "status": "duplicate",
                "transaction": transaction,
                "tx_id": str(existing),
            }

        insert_stmt = self._prepared["insert_tx"]
        extra_payload_dict: Dict[str, Any] = {}
        if isinstance(extra_json, dict):
            extra_payload_dict.update(extra_json)
        elif extra_json is not None:
            extra_payload_dict["raw_extra"] = extra_json

        if client_tx_id:
            extra_payload_dict.setdefault("client_tx_id", client_tx_id)
        if normalized_type:
            extra_payload_dict.setdefault("transaction_type", normalized_type)
            extra_payload_dict.setdefault("direction", normalized_type)

        extra_payload = _json_dumps(extra_payload_dict)
        self.session_rt.execute(
            insert_stmt,
            (
                account_id,
                event_date,
                event_ts,
                tx_id,
                _normalize_amount(amount),
                currency,
                merchant,
                status,
                extra_payload,
            ),
        )
        return {
            "status": "success",
            "transaction": {
                "account_id": account_id,
                "client_tx_id": client_tx_id,
                "event_ts": event_ts.isoformat(),
                "event_date": event_date.isoformat(),
                "tx_id": str(tx_id),
                "amount": float(amount),
                "currency": currency,
                "merchant": merchant,
                "status": status,
                "transaction_type": normalized_type,
                "extra_json": extra_payload_dict or None,
            },
        }

    def list_transactions(self, account_id: str, event_date: date, limit: int) -> List[Dict[str, Any]]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._prepared["select_tx_by_day"]
        rows = self.session_rt.execute(stmt, (account_id, event_date, limit))
        return [self._row_to_transaction(row) for row in rows]

    def list_transactions_range(
        self,
        account_id: str,
        start_date: date,
        end_date: date,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")
        results: List[Dict[str, Any]] = []
        current = start_date
        step = timedelta(days=1)
        while current <= end_date and len(results) < limit:
            rows = self.list_transactions(account_id, current, limit)
            results.extend(rows)
            current += step
        # Sort by event_date desc, event_ts desc to align with API expectation
        results.sort(key=lambda r: (r["event_date"], r["event_ts"]), reverse=True)
        return results[:limit]

    def get_transaction_by_id(
        self,
        tx_id: str,
        account_id: Optional[str] = None,
        event_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")

        if account_id and event_date:
            stmt = self._prepared["select_tx_by_id"]
            rows = self.session_rt.execute(stmt, (account_id, event_date, tx_id))
        else:
            stmt = self._prepared["select_tx_by_id_only"]
            rows = self.session_rt.execute(stmt, (tx_id,))

        row = rows.one()
        if not row:
            return None
        return self._row_to_transaction(row)

    def list_all_transactions(self, limit: int = 500) -> List[Dict[str, Any]]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")
        limit = max(1, min(limit, 5000))
        cql = (
            "SELECT account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json "
            f"FROM tx_by_account_day LIMIT {limit}"
        )
        statement = SimpleStatement(cql) if SimpleStatement else cql
        rows = self.session_rt.execute(statement)
        items = [self._row_to_transaction(row) for row in rows]
        items.sort(
            key=lambda x: (
                str(x.get("event_date") or ""),
                str(x.get("event_ts") or ""),
            ),
            reverse=True,
        )
        return items[:limit]

    # ------------------------------------------------------------------ KPI helpers
    def upsert_kpi(self, event_date: date, metric: str, value: float) -> Dict[str, Any]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._prepared["upsert_kpi_daily"]
        now_ts = datetime.now(timezone.utc)
        self.session_rt.execute(stmt, (event_date, metric, float(value), now_ts))
        return {
            "event_date": event_date.isoformat(),
            "metric": metric,
            "value": float(value),
            "updated_at": now_ts.isoformat(),
        }

    def get_kpi(self, event_date: date, metric: str) -> Optional[Dict[str, Any]]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._prepared["select_kpi_daily"]
        row = self.session_rt.execute(stmt, (event_date, metric)).one()
        if not row:
            return None
        return {
            "event_date": event_date.isoformat(),
            "metric": row.metric,
            "value": float(row.value) if row.value is not None else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def list_kpis(self, event_date: date) -> List[Dict[str, Any]]:
        if not self.session_rt:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._prepared["select_kpi_daily_all"]
        rows = self.session_rt.execute(stmt, (event_date,))
        return [
            {
                "event_date": event_date.isoformat(),
                "metric": row.metric,
                "value": float(row.value) if row.value is not None else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ Audit logging
    def log_api_call(
        self,
        day: date,
        ts: datetime,
        endpoint: str,
        method: str,
        status_code: int,
        account_id: Optional[str],
        client_ip: Optional[str],
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.session_audit:
            return

        if "insert_audit" not in self._prepared:
            self._prepared["insert_audit"] = self.session_audit.prepare(
                """
                INSERT INTO api_calls (
                    day, ts, endpoint, method, status, account_id, client_ip, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            )
        payload = _json_dumps(extra_json)
        ts = _cassandra_ts(ts)
        day = ts.date() if day is None else day
        stmt = self._prepared["insert_audit"]
        try:
            self.session_audit.execute(
                stmt,
                (
                    day,
                    ts,
                    endpoint,
                    method,
                    int(status_code),
                    account_id,
                    client_ip,
                    payload,
                ),
            )
        except Exception:
            logger.exception("Failed to log API call to Cassandra.")

    # ------------------------------------------------------------------ transforms
    def _row_to_transaction(self, row) -> Dict[str, Any]:
        extra_json = _json_loads(getattr(row, "extra_json", None))

        amount_value = getattr(row, "amount", None)
        if isinstance(amount_value, Decimal):
            amount_value = float(amount_value)
        elif amount_value is not None:
            try:
                amount_value = float(amount_value)
            except (TypeError, ValueError):
                amount_value = None

        event_ts = getattr(row, "event_ts", None)
        iso_ts: Optional[str] = None
        if isinstance(event_ts, datetime):
            iso_ts = event_ts.replace(tzinfo=timezone.utc).isoformat()
        elif isinstance(event_ts, str):
            iso_ts = event_ts
        elif event_ts is not None:
            iso_ts = str(event_ts)

        event_date_raw = getattr(row, "event_date", None)
        iso_date: Optional[str] = None
        if isinstance(event_date_raw, datetime):
            iso_date = event_date_raw.date().isoformat()
        elif isinstance(event_date_raw, date):
            iso_date = event_date_raw.isoformat()
        elif isinstance(event_date_raw, str):
            iso_date = event_date_raw

        if not iso_date and iso_ts:
            iso_date = iso_ts.split("T", 1)[0]

        client_tx_id = None
        transaction_type = None
        if isinstance(extra_json, dict):
            client_tx_id = extra_json.get("client_tx_id") or extra_json.get("clientTxId")
            transaction_type = extra_json.get("transaction_type") or extra_json.get("direction")

        return {
            "account_id": getattr(row, "account_id", None),
            "event_ts": iso_ts,
            "event_date": iso_date,
            "tx_id": str(getattr(row, "tx_id", None)),
            "client_tx_id": client_tx_id,
            "amount": amount_value,
            "currency": getattr(row, "currency", None),
            "merchant": getattr(row, "merchant", None),
            "status": getattr(row, "status", None),
            "transaction_type": transaction_type,
            "extra_json": extra_json,
        }


# Singleton-style helper exposed for the FastAPI application -------------------
cassandra_service = CassandraService()
