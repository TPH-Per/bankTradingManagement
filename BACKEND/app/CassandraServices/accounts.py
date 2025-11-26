from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


class AccountService:
    """
    Encapsulates operations on account tables (accounts_by_id, accounts_by_customer).
    """

    def __init__(self, sessions: CassandraSessionManager):
        self.sessions = sessions
        self._prepared: Dict[str, PreparedStatementType] = {}

    # ------------------------------------------------------------------ preparation
    def prepare(self) -> None:
        session = self.sessions.session_rt
        if not session:
            return

        # Insert into accounts_by_id
        if "insert_account_by_id" not in self._prepared:
            self._prepared["insert_account_by_id"] = session.prepare(
                """
                INSERT INTO accounts_by_id (
                    account_id,
                    customer_id,
                    currency,
                    status,
                    opened_at,
                    updated_at,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Insert into accounts_by_customer
        if "insert_account_by_customer" not in self._prepared:
            self._prepared["insert_account_by_customer"] = session.prepare(
                """
                INSERT INTO accounts_by_customer (
                    customer_id,
                    account_id,
                    opened_at,
                    status
                ) VALUES (?, ?, ?, ?)
                """
            )

        # Select account by ID
        if "select_account_by_id" not in self._prepared:
            self._prepared["select_account_by_id"] = session.prepare(
                """
                SELECT account_id, customer_id, currency, status, opened_at, updated_at, extra_json
                FROM accounts_by_id
                WHERE account_id=?
                """
            )

        # Select accounts by customer
        if "select_accounts_by_customer" not in self._prepared:
            self._prepared["select_accounts_by_customer"] = session.prepare(
                """
                SELECT customer_id, account_id, opened_at, status
                FROM accounts_by_customer
                WHERE customer_id=?
                """
            )

        # Update account
        if "update_account" not in self._prepared:
            self._prepared["update_account"] = session.prepare(
                """
                UPDATE accounts_by_id
                SET status=?, updated_at=?, extra_json=?
                WHERE account_id=?
                """
            )

        # Note: We don't prepare select_all_accounts because LIMIT cannot use bound parameters
        # It will be created as a SimpleStatement in list_all_accounts method

        # Insert into customers_by_identity
        if "insert_customer_by_identity" not in self._prepared:
            self._prepared["insert_customer_by_identity"] = session.prepare(
                """
                INSERT INTO customers_by_identity (
                    national_id,
                    customer_id,
                    full_name,
                    dob,
                    phone,
                    status,
                    created_at,
                    updated_at,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Select customer by identity
        if "select_customer_by_identity" not in self._prepared:
            self._prepared["select_customer_by_identity"] = session.prepare(
                """
                SELECT national_id, customer_id, full_name, dob, phone, status, created_at, updated_at, extra_json
                FROM customers_by_identity
                WHERE national_id=?
                """
            )

        # Insert into customers_by_email
        if "insert_customer_by_email" not in self._prepared:
            self._prepared["insert_customer_by_email"] = session.prepare(
                """
                INSERT INTO customers_by_email (
                    email,
                    customer_id,
                    full_name,
                    phone,
                    national_id,
                    status,
                    created_at,
                    updated_at,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Select customer by email
        if "select_customer_by_email" not in self._prepared:
            self._prepared["select_customer_by_email"] = session.prepare(
                """
                SELECT email, customer_id, full_name, phone, national_id, status, created_at, updated_at, extra_json
                FROM customers_by_email
                WHERE email=?
                """
            )

        # Insert into customers_by_phone
        if "insert_customer_by_phone" not in self._prepared:
            self._prepared["insert_customer_by_phone"] = session.prepare(
                """
                INSERT INTO customers_by_phone (
                    phone,
                    customer_id,
                    full_name,
                    email,
                    national_id,
                    status,
                    created_at,
                    updated_at,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )

        # Select customer by phone
        if "select_customer_by_phone" not in self._prepared:
            self._prepared["select_customer_by_phone"] = session.prepare(
                """
                SELECT phone, customer_id, full_name, email, national_id, status, created_at, updated_at, extra_json
                FROM customers_by_phone
                WHERE phone=?
                """
            )

        # Prepare balance statements
        self._prepare_balance_statements(session)

    # ------------------------------------------------------------------ operations
    def create_account(
        self,
        account_id: str,
        customer_id: str,
        currency: str = "VND",
        status: str = "ACTIVE",
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new account in both accounts_by_id and accounts_by_customer tables.
        
        Note: This method does NOT validate if customer_id exists.
        Accounts can be created with any customer_id value.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        now = _cassandra_ts(datetime.now(timezone.utc))
        extra_json_str = _json_dumps(extra_json)

        # Insert into accounts_by_id
        stmt = self._prepared.get("insert_account_by_id")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("insert_account_by_id")

        session.execute(
            stmt,
            (
                account_id,
                customer_id,
                currency,
                status,
                now,
                now,
                extra_json_str,
            ),
        )



        # Insert into customers_by_identity if national_id is present
        if extra_json and extra_json.get("national_id"):
            stmt_identity = self._prepared.get("insert_customer_by_identity")
            if not stmt_identity:
                self.prepare()
                stmt_identity = self._prepared.get("insert_customer_by_identity")
            
            national_id = extra_json.get("national_id")
            full_name = extra_json.get("full_name")
            phone = extra_json.get("phone")
            dob_str = extra_json.get("dob")
            dob = None
            if dob_str:
                try:
                    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                except ValueError:
                    pass
            
            session.execute(
                stmt_identity,
                (
                    national_id,
                    customer_id,
                    full_name,
                    dob,
                    phone,
                    status,
                    now,
                    now,
                    extra_json_str
                )
            )

        # Insert into customers_by_email if email is present
        if extra_json and extra_json.get("email"):
            stmt_email = self._prepared.get("insert_customer_by_email")
            if not stmt_email:
                self.prepare()
                stmt_email = self._prepared.get("insert_customer_by_email")
            
            email = extra_json.get("email")
            full_name = extra_json.get("full_name")
            phone = extra_json.get("phone")
            national_id = extra_json.get("national_id")
            
            session.execute(
                stmt_email,
                (
                    email,
                    customer_id,
                    full_name,
                    phone,
                    national_id,
                    status,
                    now,
                    now,
                    extra_json_str
                )
            )

        # Insert into customers_by_phone if phone is present
        if extra_json and extra_json.get("phone"):
            phone_val = extra_json.get("phone")
            logger.info(f"Inserting into customers_by_phone: {phone_val}")
            stmt_phone = self._prepared.get("insert_customer_by_phone")
            if not stmt_phone:
                self.prepare()
                stmt_phone = self._prepared.get("insert_customer_by_phone")
            
            phone = extra_json.get("phone")
            full_name = extra_json.get("full_name")
            email = extra_json.get("email")
            national_id = extra_json.get("national_id")
            
            session.execute(
                stmt_phone,
                (
                    phone,
                    customer_id,
                    full_name,
                    email,
                    national_id,
                    status,
                    now,
                    now,
                    extra_json_str
                )
            )

        # Insert into accounts_by_customer
        stmt_customer = self._prepared.get("insert_account_by_customer")
        if not stmt_customer:
            self.prepare()
            stmt_customer = self._prepared.get("insert_account_by_customer")
        
        session.execute(
            stmt_customer,
            (
                customer_id,
                account_id,
                now,
                status
            )
        )

        return {
            "account_id": account_id,
            "customer_id": customer_id,
            "currency": currency,
            "status": status,
            "opened_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "extra_json": extra_json,
        }

    def get_account_by_id(self, account_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an account by account_id.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_account_by_id")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_account_by_id")

        rows = session.execute(stmt, (account_id,))
        row = rows.one()
        if not row:
            return None

        return {
            "account_id": row.account_id,
            "customer_id": row.customer_id,
            "currency": row.currency,
            "status": row.status,
            "opened_at": row.opened_at.isoformat() if row.opened_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "extra_json": _json_loads(row.extra_json),
        }

    def get_accounts_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all accounts for a given customer_id.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_accounts_by_customer")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_accounts_by_customer")

        rows = session.execute(stmt, (customer_id,))
        results = []
        for row in rows:
            results.append({
                "customer_id": row.customer_id,
                "account_id": row.account_id,
                "opened_at": row.opened_at.isoformat() if row.opened_at else None,
                "status": row.status,
            })
        return results

    def get_customer_by_identity(self, national_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer by national_id (CCCD/CMND).
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_customer_by_identity")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_customer_by_identity")

        rows = session.execute(stmt, (national_id,))
        row = rows.one()
        if not row:
            return None

        return {
            "national_id": row.national_id,
            "customer_id": row.customer_id,
            "full_name": row.full_name,
            "dob": row.dob.isoformat() if row.dob else None,
            "phone": row.phone,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "extra_json": _json_loads(row.extra_json),
        }

    def get_customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer by email.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_customer_by_email")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_customer_by_email")

        rows = session.execute(stmt, (email,))
        row = rows.one()
        if not row:
            return None

        return {
            "email": row.email,
            "customer_id": row.customer_id,
            "full_name": row.full_name,
            "phone": row.phone,
            "national_id": row.national_id,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "extra_json": _json_loads(row.extra_json),
        }

    def get_customer_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer by phone.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_customer_by_phone")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("select_customer_by_phone")

        rows = session.execute(stmt, (phone,))
        row = rows.one()
        if not row:
            return None

        return {
            "phone": row.phone,
            "customer_id": row.customer_id,
            "full_name": row.full_name,
            "email": row.email,
            "national_id": row.national_id,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "extra_json": _json_loads(row.extra_json),
        }

    def update_account(
        self,
        account_id: str,
        status: Optional[str] = None,
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update account status and/or extra_json.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Get current account to preserve existing values
        current = self.get_account_by_id(account_id)
        if not current:
            return None

        # Merge extra_json if provided
        updated_extra = current.get("extra_json") or {}
        if extra_json:
            updated_extra.update(extra_json)

        updated_status = status if status is not None else current.get("status")
        now = _cassandra_ts(datetime.now(timezone.utc))
        extra_json_str = _json_dumps(updated_extra) if updated_extra else None

        stmt = self._prepared.get("update_account")
        if not stmt:
            self.prepare()
            stmt = self._prepared.get("update_account")

        session.execute(
            stmt,
            (
                updated_status,
                now,
                extra_json_str,
                account_id,
            ),
        )

        return self.get_account_by_id(account_id)


    def list_all_accounts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all accounts (limited).
        Note: Uses ALLOW FILTERING to scan all partitions (use with caution in production).
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Cassandra doesn't support bound parameters for LIMIT, so we use SimpleStatement
        # with string formatting. Since accounts_by_id has account_id as PRIMARY KEY,
        # we need ALLOW FILTERING to scan all rows.
        cql = f"SELECT account_id, customer_id, currency, status, opened_at, updated_at, extra_json FROM accounts_by_id LIMIT {limit} ALLOW FILTERING"
        if SimpleStatement:
            stmt = SimpleStatement(cql)
        else:
            stmt = cql
        
        try:
            rows = session.execute(stmt)
            results = []
            for row in rows:
                results.append({
                    "account_id": row.account_id,
                    "customer_id": row.customer_id,
                    "currency": row.currency,
                    "status": row.status,
                    "opened_at": row.opened_at.isoformat() if row.opened_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "extra_json": _json_loads(row.extra_json),
                })
            return results
        except Exception as e:
            logger.exception(f"Failed to list all accounts: {e}")
            # Return empty list if query fails
            return []

    # ------------------------------------------------------------------ Balance operations
    def get_balance(self, account_id: str) -> Optional[Dict[str, Any]]:
        """
        Get account balance from account_balances table.
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        stmt = self._prepared.get("select_balance")
        if not stmt:
            self._prepare_balance_statements(session)
            stmt = self._prepared.get("select_balance")

        rows = session.execute(stmt, (account_id,))
        row = rows.one()
        if not row:
            return None

        return {
            "account_id": row.account_id,
            "balance": float(row.balance) if row.balance else 0.0,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def update_balance(
        self,
        account_id: str,
        amount_delta: float,
        operation: str = "add",  # "add", "subtract", or "set"
    ) -> Dict[str, Any]:
        """
        Update account balance. If balance doesn't exist, it will be created.
        
        Args:
            account_id: Account ID
            amount_delta: Amount to add/subtract or set
            operation: "add", "subtract", or "set"
        """
        session = self.sessions.session_rt
        if not session:
            raise CassandraUnavailable("Cassandra session not available")

        # Get current balance
        current_balance = self.get_balance(account_id)
        current_value = current_balance.get("balance", 0.0) if current_balance else 0.0

        # Calculate new balance
        if operation == "set":
            new_balance = amount_delta
        elif operation == "subtract":
            new_balance = current_value - abs(amount_delta)
        else:  # "add" (default)
            new_balance = current_value + abs(amount_delta)

        now = _cassandra_ts(datetime.now(timezone.utc))

        # Update balance
        stmt = self._prepared.get("upsert_balance")
        if not stmt:
            self._prepare_balance_statements(session)
            stmt = self._prepared.get("upsert_balance")

        from decimal import Decimal
        session.execute(
            stmt,
            (
                account_id,
                Decimal(str(new_balance)),
                now,
            ),
        )

        return {
            "account_id": account_id,
            "balance": new_balance,
            "previous_balance": current_value,
            "updated_at": now.isoformat(),
        }

    def _prepare_balance_statements(self, session) -> None:
        """Prepare balance-related statements."""
        if "select_balance" not in self._prepared:
            self._prepared["select_balance"] = session.prepare(
                """
                SELECT account_id, balance, updated_at
                FROM account_balances
                WHERE account_id=?
                """
            )

        if "upsert_balance" not in self._prepared:
            self._prepared["upsert_balance"] = session.prepare(
                """
                INSERT INTO account_balances (account_id, balance, updated_at)
                VALUES (?, ?, ?)
                """
            )

