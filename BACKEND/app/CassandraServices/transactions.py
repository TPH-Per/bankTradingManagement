from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .base import (
    CassandraSessionManager,
    CassandraUnavailable,
    PreparedStatementType,
    SimpleStatement,
    _cassandra_ts,
    _json_dumps,
    _json_loads,
    _normalize_amount,
    logger,
)


class TransactionService:
    """
    Encapsulates operations on the transactional ledger tables.
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}
        self._dedup_id_attr: str = "tx_id"
        self._dedup_table: Optional[str] = None

    # ------------------------------------------------------------------ preparation
    def prepare(self) -> None:
        session = self.sessions.session_rt
        if not session:
            return
        if "insert_tx" not in self._prepared:
            self._prepared["insert_tx"] = session.prepare(
                """
                INSERT INTO tx_by_account_day (
                    account_id,
                    event_date,
                    event_ts,
                    tx_id,
                    transfer_id,
                    direction,
                    counterparty_account_id,
                    amount,
                    currency,
                    merchant,
                    status,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )
        if "insert_dedup" not in self._prepared:
            self._prepare_dedup_statement(session)
        select_columns = (
            "account_id, event_date, event_ts, tx_id, transfer_id, direction, "
            "counterparty_account_id, amount, currency, merchant, status, extra_json"
        )
        if "select_tx_by_day" not in self._prepared:
            self._prepared["select_tx_by_day"] = session.prepare(
                f"""
                SELECT {select_columns}
                FROM tx_by_account_day
                WHERE account_id=? AND event_date=?
                LIMIT ?
                """
            )
        if "select_tx_by_id" not in self._prepared:
            self._prepared["select_tx_by_id"] = session.prepare(
                f"""
                SELECT {select_columns}
                FROM tx_by_account_day
                WHERE account_id=? AND event_date=? AND tx_id=?
                ALLOW FILTERING
                """
            )
        if "select_tx_by_id_only" not in self._prepared:
            self._prepared["select_tx_by_id_only"] = session.prepare(
                f"""
                SELECT {select_columns}
                FROM tx_by_account_day
                WHERE tx_id=?
                ALLOW FILTERING
                """
            )

    def _prepare_dedup_statement(self, session) -> None:
        """
        Support both legacy client_tx_dedup and the newer transfer_dedup table.
        """
        dedup_options = [
            {
                "table": "client_tx_dedup",
                "id_attr": "tx_id",
                "statement": """
                    INSERT INTO client_tx_dedup (account_id, client_tx_id, tx_id, created_at)
                    VALUES (?, ?, ?, toTimestamp(now()))
                    IF NOT EXISTS
                """,
            },
            {
                "table": "transfer_dedup",
                "id_attr": "transfer_id",
                "statement": """
                    INSERT INTO transfer_dedup (from_account, client_transfer_id, transfer_id, created_at)
                    VALUES (?, ?, ?, toTimestamp(now()))
                    IF NOT EXISTS
                """,
            },
        ]
        last_exc: Optional[Exception] = None
        for option in dedup_options:
            try:
                self._prepared["insert_dedup"] = session.prepare(option["statement"])
                self._dedup_id_attr = option["id_attr"]
                self._dedup_table = option["table"]
                if option["table"] != "client_tx_dedup":
                    logger.info(
                        "Falling back to %s for Cassandra deduplication entries.",
                        option["table"],
                    )
                return
            except Exception as exc:  # pragma: no cover - depends on Cassandra schema
                last_exc = exc
        if last_exc:
            raise last_exc

    # ------------------------------------------------------------------ core ops
    def record_transaction(
        self,
        account_id: str,
        client_tx_id: Optional[str],
        event_ts: datetime,
        tx_id,
        amount: float,
        currency: str,
        merchant: Optional[str],
        status: Optional[str],
        extra_json: Optional[Dict[str, Any]],
        transaction_type: Optional[str] = None,
        transfer_id: Optional[str] = None,
        direction: Optional[str] = None,
        counterparty_account_id: Optional[str] = None,
        use_dedup: bool = True,
    ) -> Dict[str, Any]:
        session = self.sessions.session_rt
        self.prepare()
        if not session or "insert_tx" not in self._prepared:
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

        if use_dedup and client_tx_id:
            dedup_stmt = self._prepared["insert_dedup"]
            result = session.execute(dedup_stmt, (account_id, client_tx_id, tx_id))
            applied = result.one()
            if applied is None:
                raise RuntimeError("Unexpected empty result from Cassandra dedup insert.")

            if not applied.applied:  # type: ignore[attr-defined]
                existing_attr = self._dedup_id_attr or "tx_id"
                existing = getattr(applied, existing_attr, None)
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
                    "tx_id": str(existing) if existing is not None else None,
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
        if transfer_id:
            extra_payload_dict.setdefault("transfer_id", str(transfer_id))
        if direction and not normalized_type:
            extra_payload_dict.setdefault("direction", direction.lower())
        if counterparty_account_id:
            extra_payload_dict.setdefault("counterparty_account_id", counterparty_account_id)

        extra_payload = _json_dumps(extra_payload_dict)
        session.execute(
            insert_stmt,
            (
                account_id,
                event_date,
                event_ts,
                tx_id,
                transfer_id,
                direction,
                counterparty_account_id,
                _normalize_amount(amount),
                currency,
                merchant,
                status,
                extra_payload,
            ),
        )
        if not transaction_type and direction_value:
            transaction_type = direction_value.lower() if isinstance(direction_value, str) else direction_value

        return {
            "status": "success",
            "transaction": {
                "account_id": account_id,
                "client_tx_id": client_tx_id,
                "event_ts": event_ts.isoformat(),
                "event_date": event_date.isoformat(),
                "tx_id": str(tx_id),
                "transfer_id": str(transfer_id) if transfer_id else None,
                "direction": direction or normalized_type,
                "counterparty_account_id": counterparty_account_id,
                "amount": float(amount),
                "currency": currency,
                "merchant": merchant,
                "status": status,
                "transaction_type": normalized_type,
                "extra_json": extra_payload_dict or None,
            },
        }

    def list_transactions(self, account_id: str, event_date: date, limit: int) -> List[Dict[str, Any]]:
        session = self.sessions.session_rt
        self.prepare()
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._prepared["select_tx_by_day"]
        rows = session.execute(stmt, (account_id, event_date, limit))
        return [self._row_to_transaction(row) for row in rows]

    def list_transactions_range(
        self,
        account_id: str,
        start_date: date,
        end_date: date,
        limit: int,
    ) -> List[Dict[str, Any]]:
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")
        results: List[Dict[str, Any]] = []
        current = start_date
        step = timedelta(days=1)
        while current <= end_date and len(results) < limit:
            day_rows = self.list_transactions(account_id, current, limit)
            results.extend(day_rows)
            current += step
        results.sort(key=lambda r: (r["event_date"], r["event_ts"]), reverse=True)
        return results[:limit]

    def get_transaction_by_id(
        self,
        tx_id: str,
        account_id: Optional[str] = None,
        event_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        session = self.sessions.session_rt
        self.prepare()
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")

        if account_id and event_date:
            stmt = self._prepared["select_tx_by_id"]
            rows = session.execute(stmt, (account_id, event_date, tx_id))
        else:
            stmt = self._prepared["select_tx_by_id_only"]
            rows = session.execute(stmt, (tx_id,))

        row = rows.one()
        if not row:
            return None
        return self._row_to_transaction(row)

    def list_all_transactions(self, limit: int = 500) -> List[Dict[str, Any]]:
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")
        limit = max(1, min(limit, 5000))
        cql = (
            "SELECT account_id, event_date, event_ts, tx_id, transfer_id, direction, "
            "counterparty_account_id, amount, currency, merchant, status, extra_json "
            f"FROM tx_by_account_day LIMIT {limit}"
        )
        statement = SimpleStatement(cql) if SimpleStatement else cql
        rows = session.execute(statement)
        items = [self._row_to_transaction(row) for row in rows]
        items.sort(
            key=lambda x: (
                str(x.get("event_date") or ""),
                str(x.get("event_ts") or ""),
            ),
            reverse=True,
        )
        return items[:limit]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _row_to_transaction(row) -> Dict[str, Any]:
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
        description = None
        type_val = None
        transfer_id_value = getattr(row, "transfer_id", None)
        direction_value = getattr(row, "direction", None)
        counterparty_value = getattr(row, "counterparty_account_id", None)
        if isinstance(extra_json, dict):
            client_tx_id = extra_json.get("client_tx_id") or extra_json.get("clientTxId")
            transaction_type = extra_json.get("transaction_type") or extra_json.get("direction")
            description = extra_json.get("description")
            type_val = extra_json.get("form_type") or transaction_type
            transfer_id_value = transfer_id_value or extra_json.get("transfer_id")
            direction_value = direction_value or extra_json.get("direction")
            counterparty_value = counterparty_value or extra_json.get("counterparty_account_id")

        if not type_val:
            type_val = direction_value

        return {
            "account_id": getattr(row, "account_id", None),
            "event_ts": iso_ts,
            "event_date": iso_date,
            "tx_id": str(getattr(row, "tx_id", None)),
            "transfer_id": str(transfer_id_value) if transfer_id_value else None,
            "direction": direction_value,
            "counterparty_account_id": counterparty_value,
            "client_tx_id": client_tx_id,
            "amount": amount_value,
            "currency": getattr(row, "currency", None),
            "merchant": getattr(row, "merchant", None),
            "status": getattr(row, "status", None),
            "transaction_type": transaction_type,
            "type": type_val,
            "description": description,
            "extra_json": extra_json,
        }


__all__ = ["TransactionService"]
