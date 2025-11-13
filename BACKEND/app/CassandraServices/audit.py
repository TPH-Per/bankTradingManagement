from datetime import date, datetime
from typing import Any, Dict, Optional

from .base import (
    CassandraSessionManager,
    PreparedStatementType,
    _cassandra_ts,
    _json_dumps,
    logger,
)


class AuditService:
    """
    Persists API call logs into the pldt_audit keyspace.
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}

    def prepare(self) -> None:
        session = self.sessions.session_audit
        if not session:
            return
        if "insert_audit" not in self._prepared:
            self._prepared["insert_audit"] = session.prepare(
                """
                INSERT INTO api_calls (
                    day, ts, endpoint, method, status, account_id, client_ip, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

    def log_api_call(
        self,
        day: Optional[date],
        ts: datetime,
        endpoint: str,
        method: str,
        status_code: int,
        account_id: Optional[str],
        client_ip: Optional[str],
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        session = self.sessions.session_audit
        if not session:
            return
        self.prepare()
        stmt = self._prepared.get("insert_audit")
        if not stmt:
            return

        payload = _json_dumps(extra_json)
        ts = _cassandra_ts(ts)
        day_value = ts.date() if day is None else day
        try:
            session.execute(
                stmt,
                (
                    day_value,
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


__all__ = ["AuditService"]
