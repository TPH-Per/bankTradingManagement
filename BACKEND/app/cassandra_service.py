import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from .CassandraServices import (
    AuditService,
    CassandraConfig,
    CassandraSessionManager,
    CassandraUnavailable,
    KPIService,
    TransactionService,
    load_config,
)

logger = logging.getLogger("bankTrading.cassandra.service")


class CassandraService:
    """
    Facade that wires concrete Cassandra service modules into a single object.
    """

    def __init__(self, config: Optional[CassandraConfig] = None):
        self.sessions = CassandraSessionManager(config or load_config())
        self.config = self.sessions.config
        self.transactions = TransactionService(self.sessions)
        self.kpis = KPIService(self.sessions)
        self.audit = AuditService(self.sessions)

        self.sessions.connect()
        self._prepare_services()

    # ------------------------------------------------------------------ helpers
    def _prepare_services(self) -> None:
        services = [
            ("transactions", self.transactions),
            ("kpis", self.kpis),
            ("audit", self.audit),
        ]
        for name, service in services:
            try:
                service.prepare()
            except Exception:
                logger.exception("Failed to initialize Cassandra %s service; continuing in degraded mode.", name)

    @property
    def cluster(self):
        return self.sessions.cluster

    @property
    def session_rt(self):
        return self.sessions.session_rt

    @property
    def session_audit(self):
        return self.sessions.session_audit

    # ------------------------------------------------------------------ status
    def available(self) -> bool:
        return self.sessions.available()

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
        return self.transactions.record_transaction(
            account_id=account_id,
            client_tx_id=client_tx_id,
            event_ts=event_ts,
            tx_id=tx_id,
            amount=amount,
            currency=currency,
            merchant=merchant,
            status=status,
            extra_json=extra_json,
            transaction_type=transaction_type,
        )

    def list_transactions(self, account_id: str, event_date: date, limit: int) -> List[Dict[str, Any]]:
        return self.transactions.list_transactions(account_id, event_date, limit)

    def list_transactions_range(
        self,
        account_id: str,
        start_date: date,
        end_date: date,
        limit: int,
    ) -> List[Dict[str, Any]]:
        return self.transactions.list_transactions_range(account_id, start_date, end_date, limit)

    def get_transaction_by_id(
        self,
        tx_id: str,
        account_id: Optional[str] = None,
        event_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.transactions.get_transaction_by_id(
            tx_id=tx_id,
            account_id=account_id,
            event_date=event_date,
        )

    def list_all_transactions(self, limit: int = 500) -> List[Dict[str, Any]]:
        return self.transactions.list_all_transactions(limit)

    # ------------------------------------------------------------------ KPI helpers
    def upsert_kpi(self, event_date: date, metric: str, value: float) -> Dict[str, Any]:
        return self.kpis.upsert_kpi(event_date, metric, value)

    def get_kpi(self, event_date: date, metric: str) -> Optional[Dict[str, Any]]:
        return self.kpis.get_kpi(event_date, metric)

    def list_kpis(self, event_date: date) -> List[Dict[str, Any]]:
        return self.kpis.list_kpis(event_date)

    # ------------------------------------------------------------------ Audit logging
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
        self.audit.log_api_call(
            day=day,
            ts=ts,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            account_id=account_id,
            client_ip=client_ip,
            extra_json=extra_json,
        )


# Singleton helper for FastAPI
cassandra_service = CassandraService()
