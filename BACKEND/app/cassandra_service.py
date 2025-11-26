import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from .CassandraServices import (
    AccountService,
    AuditService,
    BalanceSnapshotService,
    CassandraConfig,
    CassandraSessionManager,
    CassandraUnavailable,
    P2PTransactionService,
    TransactionService,
    TransferService,
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
        self.accounts = AccountService(self.sessions)
        self.transactions = TransactionService(self.sessions)
        self.audit = AuditService(self.sessions)
        self.transfers = TransferService(self.sessions)
        self.balance_snapshots = BalanceSnapshotService(self.sessions)
        self.p2p_transactions = P2PTransactionService(self.sessions)

        self.sessions.connect()
        self._prepare_services()

    # ------------------------------------------------------------------ helpers
    def _prepare_services(self) -> None:
        services = [
            ("accounts", self.accounts),
            ("transactions", self.transactions),
            ("audit", self.audit),
            ("transfers", self.transfers),
            ("balance_snapshots", self.balance_snapshots),
            ("p2p_transactions", self.p2p_transactions),
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

    # ------------------------------------------------------------------ Account operations
    def create_account(
        self,
        account_id: str,
        customer_id: str,
        currency: str = "VND",
        status: str = "ACTIVE",
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.accounts.create_account(
            account_id=account_id,
            customer_id=customer_id,
            currency=currency,
            status=status,
            extra_json=extra_json,
        )

    def get_account_by_id(self, account_id: str) -> Optional[Dict[str, Any]]:
        return self.accounts.get_account_by_id(account_id)

    def get_accounts_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        return self.accounts.get_accounts_by_customer(customer_id)

    def get_customer_by_identity(self, national_id: str) -> Optional[Dict[str, Any]]:
        return self.accounts.get_customer_by_identity(national_id)

    def update_account(
        self,
        account_id: str,
        status: Optional[str] = None,
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.accounts.update_account(
            account_id=account_id,
            status=status,
            extra_json=extra_json,
        )



    def get_account_balance(self, account_id: str) -> Optional[Dict[str, Any]]:
        return self.accounts.get_balance(account_id)

    def update_account_balance(
        self,
        account_id: str,
        amount_delta: float,
        operation: str = "add",
    ) -> Dict[str, Any]:
        return self.accounts.update_balance(account_id, amount_delta, operation)

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

    # ------------------------------------------------------------------ Account delegates
    def get_account_by_id(self, account_id: str):
        """Delegate to accounts service"""
        return self.accounts.get_account_by_id(account_id)
    
    def get_accounts_by_customer(self, customer_id: str):
        """Delegate to accounts service"""
        return self.accounts.get_accounts_by_customer(customer_id)
    
    def get_customer_by_identity(self, national_id: str):
        """Delegate to accounts service"""
        return self.accounts.get_customer_by_identity(national_id)
    
    def get_account_balance(self, account_id: str):
        """Delegate to accounts service"""
        return self.accounts.get_balance(account_id)
    
    def update_account_balance(self, account_id: str, amount_delta: float, operation: str = "add"):
        """Delegate to accounts service"""
        return self.accounts.update_balance(account_id, amount_delta, operation)
    
    def create_account(self, account_data):
        """Delegate to accounts service"""
        return self.accounts.create_account(account_data)


# Singleton helper for FastAPI
cassandra_service = CassandraService()
