from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base import (
    CassandraSessionManager,
    CassandraUnavailable,
    PreparedStatementType,
    _cassandra_ts,
    _json_dumps,
    _json_loads,
    logger,
)


class P2PTransactionService:
    """
    Service for P2P (peer-to-peer) transaction tables:
    - p2p_tx_by_account_pair_month: conversation-style history between two accounts
    - tx_by_account_pair_day: directional transactions from A to B by day
    - p2p_tx_by_customer_pair_month: conversation-style history between two customers
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}

    def prepare(self) -> None:
        """Prepare all statements for P2P transaction tables."""
        session = self.sessions.session_rt
        if not session:
            return

        # Insert into p2p_tx_by_account_pair_month
        if "insert_p2p_account_pair_month" not in self._prepared:
            self._prepared["insert_p2p_account_pair_month"] = session.prepare(
                """
                INSERT INTO p2p_tx_by_account_pair_month (
                    party1_account_id, party2_account_id, month_yyyymm,
                    event_ts, tx_id, transfer_id,
                    from_account, to_account,
                    amount, currency, status, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Select from p2p_tx_by_account_pair_month
        if "select_p2p_account_pair_month" not in self._prepared:
            self._prepared["select_p2p_account_pair_month"] = session.prepare(
                """
                SELECT party1_account_id, party2_account_id, month_yyyymm,
                       event_ts, tx_id, transfer_id,
                       from_account, to_account,
                       amount, currency, status, extra_json
                FROM p2p_tx_by_account_pair_month
                WHERE party1_account_id = ?
                  AND party2_account_id = ?
                  AND month_yyyymm = ?
                LIMIT ?
                """
            )

        # Insert into tx_by_account_pair_day
        if "insert_tx_account_pair_day" not in self._prepared:
            self._prepared["insert_tx_account_pair_day"] = session.prepare(
                """
                INSERT INTO tx_by_account_pair_day (
                    from_account, to_account, event_date,
                    event_ts, tx_id, transfer_id,
                    amount, currency, status, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Select from tx_by_account_pair_day
        if "select_tx_account_pair_day" not in self._prepared:
            self._prepared["select_tx_account_pair_day"] = session.prepare(
                """
                SELECT from_account, to_account, event_date,
                       event_ts, tx_id, transfer_id,
                       amount, currency, status, extra_json
                FROM tx_by_account_pair_day
                WHERE from_account = ?
                  AND to_account = ?
                  AND event_date = ?
                LIMIT ?
                """
            )

        # Insert into p2p_tx_by_customer_pair_month
        if "insert_p2p_customer_pair_month" not in self._prepared:
            self._prepared["insert_p2p_customer_pair_month"] = session.prepare(
                """
                INSERT INTO p2p_tx_by_customer_pair_month (
                    party1_customer_id, party2_customer_id, month_yyyymm,
                    event_ts, tx_id, transfer_id,
                    from_customer_id, to_customer_id,
                    from_account, to_account,
                    amount, currency, status, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Select from p2p_tx_by_customer_pair_month
        if "select_p2p_customer_pair_month" not in self._prepared:
            self._prepared["select_p2p_customer_pair_month"] = session.prepare(
                """
                SELECT party1_customer_id, party2_customer_id, month_yyyymm,
                       event_ts, tx_id, transfer_id,
                       from_customer_id, to_customer_id,
                       from_account, to_account,
                       amount, currency, status, extra_json
                FROM p2p_tx_by_customer_pair_month
                WHERE party1_customer_id = ?
                  AND party2_customer_id = ?
                  AND month_yyyymm = ?
                LIMIT ?
                """
            )

    def record_p2p_transaction(
        self,
        from_account: str,
        to_account: str,
        from_customer_id: Optional[str],
        to_customer_id: Optional[str],
        event_ts: datetime,
        tx_id: UUID,
        transfer_id: Optional[UUID],
        amount: float,
        currency: str = "VND",
        status: str = "SETTLED",
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a P2P transaction to all relevant tables.
        
        Args:
            from_account: Source account ID
            to_account: Destination account ID
            from_customer_id: Source customer ID (optional)
            to_customer_id: Destination customer ID (optional)
            event_ts: Transaction timestamp
            tx_id: Transaction ID
            transfer_id: Transfer ID
            amount: Transaction amount
            currency: Currency code
            status: Transaction status
            extra_json: Additional metadata
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Ensure statements are prepared
        if not self._prepared:
            self.prepare()

        # Convert timestamp and calculate month
        ts = _cassandra_ts(event_ts)
        month_yyyymm = int(event_ts.strftime("%Y%m"))
        event_date = event_ts.date()
        extra_json_str = _json_dumps(extra_json)

        # Convert amount to Decimal for Cassandra
        from decimal import Decimal
        amount_decimal = Decimal(str(amount))

        # 1. Write to p2p_tx_by_account_pair_month
        party1_account = min(from_account, to_account)
        party2_account = max(from_account, to_account)
        
        stmt = self._prepared.get("insert_p2p_account_pair_month")
        if stmt:
            session.execute(
                stmt,
                (
                    party1_account,
                    party2_account,
                    month_yyyymm,
                    ts,
                    tx_id,
                    transfer_id,
                    from_account,
                    to_account,
                    amount_decimal,
                    currency,
                    status,
                    extra_json_str,
                ),
            )

        # 2. Write to tx_by_account_pair_day (directional)
        stmt = self._prepared.get("insert_tx_account_pair_day")
        if stmt:
            session.execute(
                stmt,
                (
                    from_account,
                    to_account,
                    event_date,
                    ts,
                    tx_id,
                    transfer_id,
                    amount_decimal,
                    currency,
                    status,
                    extra_json_str,
                ),
            )

        # 3. Write to p2p_tx_by_customer_pair_month (if customer IDs available)
        if from_customer_id and to_customer_id:
            party1_customer = min(from_customer_id, to_customer_id)
            party2_customer = max(from_customer_id, to_customer_id)
            
            stmt = self._prepared.get("insert_p2p_customer_pair_month")
            if stmt:
                session.execute(
                    stmt,
                    (
                        party1_customer,
                        party2_customer,
                        month_yyyymm,
                        ts,
                        tx_id,
                        transfer_id,
                        from_customer_id,
                        to_customer_id,
                        from_account,
                        to_account,
                        amount_decimal,
                        currency,
                        status,
                        extra_json_str,
                    ),
                )

    def get_account_pair_history(
        self,
        account_id1: str,
        account_id2: str,
        month_yyyymm: int,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get P2P transaction history between two accounts for a specific month.
        
        Args:
            account_id1: First account ID
            account_id2: Second account ID
            month_yyyymm: Month in YYYYMM format (e.g., 202511)
            limit: Maximum number of records to return
            
        Returns:
            List of transaction records
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Ensure statements are prepared
        if not self._prepared:
            self.prepare()

        # Normalize account order
        party1 = min(account_id1, account_id2)
        party2 = max(account_id1, account_id2)

        stmt = self._prepared.get("select_p2p_account_pair_month")
        if not stmt:
            return []

        rows = session.execute(stmt, (party1, party2, month_yyyymm, limit))
        
        results = []
        for row in rows:
            results.append({
                "party1_account_id": row.party1_account_id,
                "party2_account_id": row.party2_account_id,
                "month_yyyymm": row.month_yyyymm,
                "event_ts": row.event_ts.isoformat() if row.event_ts else None,
                "tx_id": str(row.tx_id) if row.tx_id else None,
                "transfer_id": str(row.transfer_id) if row.transfer_id else None,
                "from_account": row.from_account,
                "to_account": row.to_account,
                "amount": float(row.amount) if row.amount else 0.0,
                "currency": row.currency,
                "status": row.status,
                "extra_json": _json_loads(row.extra_json),
            })
        
        return results

    def get_directional_history(
        self,
        from_account: str,
        to_account: str,
        event_date: str,  # YYYY-MM-DD format
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get directional transaction history from account A to account B on a specific date.
        
        Args:
            from_account: Source account ID
            to_account: Destination account ID
            event_date: Date in YYYY-MM-DD format
            limit: Maximum number of records to return
            
        Returns:
            List of transaction records
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Ensure statements are prepared
        if not self._prepared:
            self.prepare()

        stmt = self._prepared.get("select_tx_account_pair_day")
        if not stmt:
            return []

        # Parse date string
        from datetime import date
        event_date_obj = date.fromisoformat(event_date)

        rows = session.execute(stmt, (from_account, to_account, event_date_obj, limit))
        
        results = []
        for row in rows:
            results.append({
                "from_account": row.from_account,
                "to_account": row.to_account,
                "event_date": row.event_date.isoformat() if row.event_date else None,
                "event_ts": row.event_ts.isoformat() if row.event_ts else None,
                "tx_id": str(row.tx_id) if row.tx_id else None,
                "transfer_id": str(row.transfer_id) if row.transfer_id else None,
                "amount": float(row.amount) if row.amount else 0.0,
                "currency": row.currency,
                "status": row.status,
                "extra_json": _json_loads(row.extra_json),
            })
        
        return results

    def get_customer_pair_history(
        self,
        customer_id1: str,
        customer_id2: str,
        month_yyyymm: int,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get P2P transaction history between two customers for a specific month.
        
        Args:
            customer_id1: First customer ID
            customer_id2: Second customer ID
            month_yyyymm: Month in YYYYMM format (e.g., 202511)
            limit: Maximum number of records to return
            
        Returns:
            List of transaction records
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Ensure statements are prepared
        if not self._prepared:
            self.prepare()

        # Normalize customer order
        party1 = min(customer_id1, customer_id2)
        party2 = max(customer_id1, customer_id2)

        stmt = self._prepared.get("select_p2p_customer_pair_month")
        if not stmt:
            return []

        rows = session.execute(stmt, (party1, party2, month_yyyymm, limit))
        
        results = []
        for row in rows:
            results.append({
                "party1_customer_id": row.party1_customer_id,
                "party2_customer_id": row.party2_customer_id,
                "month_yyyymm": row.month_yyyymm,
                "event_ts": row.event_ts.isoformat() if row.event_ts else None,
                "tx_id": str(row.tx_id) if row.tx_id else None,
                "transfer_id": str(row.transfer_id) if row.transfer_id else None,
                "from_customer_id": row.from_customer_id,
                "to_customer_id": row.to_customer_id,
                "from_account": row.from_account,
                "to_account": row.to_account,
                "amount": float(row.amount) if row.amount else 0.0,
                "currency": row.currency,
                "status": row.status,
                "extra_json": _json_loads(row.extra_json),
            })
        
        return results
