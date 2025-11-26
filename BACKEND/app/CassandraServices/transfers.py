from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid

from .base import (
    CassandraSessionManager,
    CassandraUnavailable,
    PreparedStatementType,
    SimpleStatement,
    _cassandra_ts,
    _json_dumps,
    _json_loads,
    logger,
)


class TransferService:
    """
    Encapsulates operations on transfer tables (transfers_by_id, transfer_dedup).
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}

    def prepare(self) -> None:
        """Prepare all statements for transfer operations."""
        session = self.sessions.session_rt
        if not session:
            return

        if "insert_transfer" not in self._prepared:
            self._prepared["insert_transfer"] = session.prepare(
                """
                INSERT INTO transfers_by_id (
                    transfer_id,
                    from_account,
                    to_account,
                    amount,
                    currency,
                    created_at,
                    status,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        if "insert_transfer_dedup" not in self._prepared:
            self._prepared["insert_transfer_dedup"] = session.prepare(
                """
                INSERT INTO transfer_dedup (
                    from_account,
                    client_transfer_id,
                    transfer_id,
                    created_at
                ) VALUES (?, ?, ?, ?) IF NOT EXISTS
                """
            )

        if "select_transfer_dedup" not in self._prepared:
            self._prepared["select_transfer_dedup"] = session.prepare(
                """
                SELECT transfer_id, created_at
                FROM transfer_dedup
                WHERE from_account = ? AND client_transfer_id = ?
                """
            )

    def create_transfer(
        self,
        from_account: str,
        to_account: str,
        amount: float,
        currency: str = "VND",
        status: str = "SETTLED",
        client_transfer_id: Optional[str] = None,
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a transfer record in transfers_by_id and transfer_dedup tables.
        
        Args:
            from_account: Sender account ID
            to_account: Receiver account ID
            amount: Transfer amount
            currency: Currency code (default: VND)
            status: Transfer status (default: SETTLED)
            client_transfer_id: Client-provided transfer ID for deduplication
            extra_json: Additional metadata
            
        Returns:
            Dict with transfer information
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Generate transfer_id if not provided
        transfer_id = uuid.uuid4()
        now = _cassandra_ts(datetime.now(timezone.utc))
        
        # Use client_transfer_id for deduplication, or generate one
        if not client_transfer_id:
            client_transfer_id = str(uuid.uuid4())

        # Check for duplicate transfer using transfer_dedup
        dedup_stmt = self._prepared.get("select_transfer_dedup")
        if dedup_stmt:
            existing = session.execute(dedup_stmt, (from_account, client_transfer_id)).one()
            if existing:
                # Transfer already exists, return existing transfer_id
                existing_transfer_id = existing.transfer_id
                logger.info(f"Duplicate transfer detected: {client_transfer_id} -> {existing_transfer_id}")
                return {
                    "status": "duplicate",
                    "transfer_id": str(existing_transfer_id),
                    "client_transfer_id": client_transfer_id,
                }

        # Insert into transfer_dedup first (for idempotency)
        dedup_insert_stmt = self._prepared.get("insert_transfer_dedup")
        if dedup_insert_stmt:
            result = session.execute(
                dedup_insert_stmt,
                (from_account, client_transfer_id, transfer_id, now)
            )
            applied = result.one()
            if applied and not applied.applied:  # type: ignore[attr-defined]
                # Dedup insert failed (already exists)
                existing_transfer_id = getattr(applied, "transfer_id", None)
                if existing_transfer_id:
                    logger.info(f"Duplicate transfer detected via dedup: {client_transfer_id} -> {existing_transfer_id}")
                    return {
                        "status": "duplicate",
                        "transfer_id": str(existing_transfer_id),
                        "client_transfer_id": client_transfer_id,
                    }

        # Insert into transfers_by_id
        insert_stmt = self._prepared.get("insert_transfer")
        if not insert_stmt:
            self.prepare()
            insert_stmt = self._prepared.get("insert_transfer")

        extra_json_str = _json_dumps(extra_json) if extra_json else None

        session.execute(
            insert_stmt,
            (
                transfer_id,
                from_account,
                to_account,
                Decimal(str(amount)),
                currency,
                now,
                status,
                extra_json_str,
            ),
        )

        logger.info(f"Created transfer {transfer_id} from {from_account} to {to_account}: {amount} {currency}")

        return {
            "status": "success",
            "transfer_id": str(transfer_id),
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "currency": currency,
            "status": status,
            "created_at": now.isoformat(),
            "client_transfer_id": client_transfer_id,
        }

    def get_transfer_by_id(self, transfer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transfer by transfer_id.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Use simple statement since we don't prepare SELECT by ID
        cql = "SELECT transfer_id, from_account, to_account, amount, currency, created_at, status, extra_json FROM transfers_by_id WHERE transfer_id = ?"
        if SimpleStatement:
            stmt = SimpleStatement(cql)
        else:
            stmt = cql
        
        row = session.execute(stmt, (uuid.UUID(transfer_id),)).one()
        if not row:
            return None

        extra_json = None
        if row.extra_json:
            extra_json = _json_loads(row.extra_json)

        return {
            "transfer_id": str(row.transfer_id),
            "from_account": row.from_account,
            "to_account": row.to_account,
            "amount": float(row.amount),
            "currency": row.currency,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "status": row.status,
            "extra_json": extra_json,
        }

    def list_transfers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all transfers, ordered by created_at descending.
        Note: Uses ALLOW FILTERING to scan all partitions (use with caution in production).
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Note: transfers_by_id has transfer_id as PRIMARY KEY, so we need ALLOW FILTERING
        # to scan all rows. For production, consider a materialized view or separate table.
        cql = f"SELECT transfer_id, from_account, to_account, amount, currency, created_at, status, extra_json FROM transfers_by_id LIMIT {limit} ALLOW FILTERING"
        if SimpleStatement:
            stmt = SimpleStatement(cql)
        else:
            stmt = cql
        
        try:
            rows = session.execute(stmt)
            results = []
            for row in rows:
                extra_json = None
                if row.extra_json:
                    extra_json = _json_loads(row.extra_json)
                
                results.append({
                    "transfer_id": str(row.transfer_id),
                    "from_account": row.from_account,
                    "to_account": row.to_account,
                    "amount": float(row.amount),
                    "currency": row.currency,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "status": row.status,
                    "extra_json": extra_json,
                })
            
            # Sort by created_at descending
            results.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            return results
        except Exception as e:
            logger.exception(f"Failed to list transfers: {e}")
            # Return empty list if query fails
            return []

