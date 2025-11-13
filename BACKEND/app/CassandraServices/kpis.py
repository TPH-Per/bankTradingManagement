from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from .base import (
    CassandraSessionManager,
    CassandraUnavailable,
    PreparedStatementType,
    logger,
)


class KPIService:
    """
    Handles daily KPI read/write operations.
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}
        self._schema_ready = False

    def prepare(self) -> None:
        session = self.sessions.session_rt
        if not session:
            return
        if not self._schema_ready:
            self._schema_ready = self._ensure_schema(session)
            if not self._schema_ready:
                logger.warning(
                    "kpi_daily table is unavailable; KPI endpoints will be disabled until the schema exists."
                )
                return
        try:
            if "upsert_kpi_daily" not in self._prepared:
                self._prepared["upsert_kpi_daily"] = session.prepare(
                    """
                    INSERT INTO kpi_daily (event_date, metric, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """
                )
            if "select_kpi_daily" not in self._prepared:
                self._prepared["select_kpi_daily"] = session.prepare(
                    """
                    SELECT metric, value, updated_at
                    FROM kpi_daily
                    WHERE event_date=? AND metric=?
                    """
                )
            if "select_kpi_daily_all" not in self._prepared:
                self._prepared["select_kpi_daily_all"] = session.prepare(
                    """
                    SELECT metric, value, updated_at
                    FROM kpi_daily
                    WHERE event_date=?
                    """
                )
        except Exception:
            logger.exception("Failed to prepare KPI statements; will retry later.")
            self._prepared.clear()
            self._schema_ready = False

    def _ensure_schema(self, session) -> bool:
        ddl = """
        CREATE TABLE IF NOT EXISTS kpi_daily (
            event_date date,
            metric text,
            value double,
            updated_at timestamp,
            PRIMARY KEY ((event_date), metric)
        )
        """
        try:
            session.execute(ddl)
            return True
        except Exception:
            logger.exception(
                "Failed to ensure kpi_daily table exists. "
                "Please run the schema migration (see db.cql)."
            )
            return False

    def _require_stmt(self, key: str) -> PreparedStatementType:
        stmt = self._prepared.get(key)
        if not stmt:
            raise CassandraUnavailable("kpi_daily table not configured in Cassandra.")
        return stmt

    def upsert_kpi(self, event_date: date, metric: str, value: float) -> Dict[str, Any]:
        session = self.sessions.session_rt
        self.prepare()
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._require_stmt("upsert_kpi_daily")
        now_ts = datetime.now(timezone.utc)
        session.execute(stmt, (event_date, metric, float(value), now_ts))
        return {
            "event_date": event_date.isoformat(),
            "metric": metric,
            "value": float(value),
            "updated_at": now_ts.isoformat(),
        }

    def get_kpi(self, event_date: date, metric: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.session_rt
        self.prepare()
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._require_stmt("select_kpi_daily")
        row = session.execute(stmt, (event_date, metric)).one()
        if not row:
            return None
        return {
            "event_date": event_date.isoformat(),
            "metric": row.metric,
            "value": float(row.value) if row.value is not None else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def list_kpis(self, event_date: date) -> List[Dict[str, Any]]:
        session = self.sessions.session_rt
        self.prepare()
        if not session:
            raise CassandraUnavailable("Cassandra session not available.")
        stmt = self._require_stmt("select_kpi_daily_all")
        rows = session.execute(stmt, (event_date,))
        return [
            {
                "event_date": event_date.isoformat(),
                "metric": row.metric,
                "value": float(row.value) if row.value is not None else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]


__all__ = ["KPIService"]
