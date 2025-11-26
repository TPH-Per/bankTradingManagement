from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .base import (
    CassandraSessionManager,
    CassandraUnavailable,
    PreparedStatementType,
    SimpleStatement,
    _cassandra_ts,
    logger,
)


class BalanceSnapshotService:
    """
    Handles daily balance snapshot operations (balance_daily_snapshots table).
    Used for account statements (Sao kê).
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}

    def prepare(self) -> None:
        """Prepare all statements for balance snapshot operations."""
        session = self.sessions.session_rt
        if not session:
            return

        if "upsert_snapshot" not in self._prepared:
            self._prepared["upsert_snapshot"] = session.prepare(
                """
                INSERT INTO balance_daily_snapshots (
                    account_id,
                    day,
                    balance_open,
                    balance_close,
                    total_debit,
                    total_credit,
                    num_tx,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        if "select_snapshot" not in self._prepared:
            self._prepared["select_snapshot"] = session.prepare(
                """
                SELECT account_id, day, balance_open, balance_close,
                       total_debit, total_credit, num_tx, updated_at
                FROM balance_daily_snapshots
                WHERE account_id = ? AND day = ?
                """
            )

        if "select_snapshot_range" not in self._prepared:
            self._prepared["select_snapshot_range"] = session.prepare(
                """
                SELECT account_id, day, balance_open, balance_close,
                       total_debit, total_credit, num_tx, updated_at
                FROM balance_daily_snapshots
                WHERE account_id = ? AND day >= ? AND day <= ?
                """
            )

    def create_snapshot(
        self,
        account_id: str,
        day: date,
        balance_open: float,
        balance_close: float,
        total_debit: float,
        total_credit: float,
        num_tx: int,
    ) -> Dict[str, Any]:
        """
        Create or update a daily balance snapshot.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("upsert_snapshot")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("upsert_snapshot")

        now = _cassandra_ts(datetime.now(timezone.utc))

        session.execute(
            stmt,
            (
                account_id,
                day,
                Decimal(str(balance_open)),
                Decimal(str(balance_close)),
                Decimal(str(total_debit)),
                Decimal(str(total_credit)),
                num_tx,
                now,
            ),
        )

        logger.info(
            f"Created balance snapshot for {account_id} on {day}: "
            f"open={balance_open}, close={balance_close}, tx={num_tx}"
        )

        return {
            "account_id": account_id,
            "day": day.isoformat(),
            "balance_open": balance_open,
            "balance_close": balance_close,
            "total_debit": total_debit,
            "total_credit": total_credit,
            "num_tx": num_tx,
            "updated_at": now.isoformat(),
        }

    def get_snapshot(self, account_id: str, day: date) -> Optional[Dict[str, Any]]:
        """
        Get a specific daily snapshot.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_snapshot")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_snapshot")

        row = session.execute(stmt, (account_id, day)).one()
        if not row:
            return None

        return {
            "account_id": row.account_id,
            "day": row.day.isoformat() if isinstance(row.day, date) else str(row.day),
            "balance_open": float(row.balance_open),
            "balance_close": float(row.balance_close),
            "total_debit": float(row.total_debit),
            "total_credit": float(row.total_credit),
            "num_tx": row.num_tx,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def get_snapshot_range(
        self, account_id: str, date_from: date, date_to: date
    ) -> List[Dict[str, Any]]:
        """
        Get all snapshots for an account within a date range.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_snapshot_range")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_snapshot_range")

        rows = session.execute(stmt, (account_id, date_from, date_to))
        results = []
        for row in rows:
            results.append({
                "account_id": row.account_id,
                "day": row.day.isoformat() if isinstance(row.day, date) else str(row.day),
                "balance_open": float(row.balance_open),
                "balance_close": float(row.balance_close),
                "total_debit": float(row.total_debit),
                "total_credit": float(row.total_credit),
                "num_tx": row.num_tx,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            })

        # Sort by day descending (newest first)
        results.sort(key=lambda x: x.get("day", ""), reverse=True)
        return results


__all__ = ["BalanceSnapshotService"]

