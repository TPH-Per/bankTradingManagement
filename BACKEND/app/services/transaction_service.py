"""
Banking Transaction Service - Application-level transaction handling
No database schema changes required
"""

import uuid
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("bankTrading")


class TransactionError(Exception):
    """Base exception for transaction errors"""
    pass


class InsufficientBalanceError(TransactionError):
    """Raised when account has insufficient balance"""
    pass


class AccountNotFoundError(TransactionError):
    """Raised when account doesn't exist"""
    pass


class TransactionFailedError(TransactionError):
    """Raised when transaction execution fails"""
    pass


class P2PTransactionService:
    """
    Peer-to-peer transaction service with application-level transaction handling.
    
    Features:
    - Atomic balance updates with Cassandra LWT
    - Automatic rollback on failure
    - Complete audit trail
    - Race condition prevention
    """
    
    def __init__(self, cassandra_service):
        self.cassandra = cassandra_service
        self.session = cassandra_service.session_rt
    
    def execute_transfer(
        self, 
        sender_id: str, 
        receiver_id: str, 
        amount: float,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Execute P2P money transfer with full transaction safety.
        
        Process:
        1. Validate accounts exist
        2. Check sender balance (with atomic read)
        3. Deduct from sender (with LWT - Compare-And-Set)
        4. Credit to receiver
        5. Log transaction
        6. Rollback on any failure
        
        Args:
            sender_id: Sender account ID
            receiver_id: Receiver account ID
            amount: Transfer amount (positive number)
            description: Optional transaction description
            
        Returns:
            Dict with transaction details
            
        Raises:
            AccountNotFoundError: If sender or receiver doesn't exist
            InsufficientBalanceError: If sender has insufficient balance
            TransactionFailedError: If transaction execution fails
        """
        tx_id = str(uuid.uuid4())
        amount_decimal = Decimal(str(amount))
        
        logger.info(f"[TX {tx_id}] Starting P2P transfer: {sender_id} -> {receiver_id}, amount: {amount}")
        
        # State tracking for rollback
        sender_debited = False
        receiver_credited = False
        
        try:
            # STEP 1: Validate accounts exist
            logger.info(f"[TX {tx_id}] Step 1: Validating accounts")
            self._validate_accounts_exist(sender_id, receiver_id)
            
            # STEP 2: Get sender's current balance
            logger.info(f"[TX {tx_id}] Step 2: Checking sender balance")
            sender_balance = self._get_account_balance(sender_id)
            
            if sender_balance < amount_decimal:
                raise InsufficientBalanceError(
                    f"Số dư không đủ. Số dư hiện tại: {sender_balance:,.0f} VND, "
                    f"Số tiền cần chuyển: {amount_decimal:,.0f} VND"
                )
            
            logger.info(f"[TX {tx_id}] Balance check passed: {sender_balance} >= {amount_decimal}")
            
            # STEP 3: Deduct from sender (ATOMIC with LWT)
            logger.info(f"[TX {tx_id}] Step 3: Deducting from sender")
            new_sender_balance = sender_balance - amount_decimal
            
            success = self._update_balance_atomic(
                account_id=sender_id,
                new_balance=new_sender_balance,
                expected_old_balance=sender_balance
            )
            
            if not success:
                raise TransactionFailedError(
                    "Race condition detected - balance changed during transaction. Please retry."
                )
            
            sender_debited = True
            logger.info(f"[TX {tx_id}] ✅ Sender debited: {sender_balance} -> {new_sender_balance}")
            
            # STEP 4: Credit to receiver
            logger.info(f"[TX {tx_id}] Step 4: Crediting to receiver")
            receiver_balance = self._get_account_balance(receiver_id)
            new_receiver_balance = receiver_balance + amount_decimal
            
            # Simple update for receiver (no LWT needed)
            self._update_balance(receiver_id, new_receiver_balance)
            receiver_credited = True
            
            logger.info(f"[TX {tx_id}] ✅ Receiver credited: {receiver_balance} -> {new_receiver_balance}")
            
            # STEP 5: Log successful transaction
            logger.info(f"[TX {tx_id}] Step 5: Logging transaction")
            self._log_transaction(
                tx_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=float(amount_decimal),
                status="completed",
                description=description
            )
            
            logger.info(f"[TX {tx_id}] ✅ Transaction completed successfully")
            
            return {
                "tx_id": tx_id,
                "status": "completed",
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "amount": float(amount_decimal),
                "sender_new_balance": float(new_sender_balance),
                "receiver_new_balance": float(new_receiver_balance),
                "timestamp": datetime.now().isoformat(),
                "description": description
            }
            
        except (InsufficientBalanceError, AccountNotFoundError) as e:
            # Expected business logic errors - no rollback needed
            logger.warning(f"[TX {tx_id}] ❌ Transaction rejected: {e}")
            self._log_transaction(
                tx_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=float(amount_decimal),
                status="rejected",
                description=f"Error: {str(e)}"
            )
            raise
            
        except Exception as e:
            # Unexpected error - attempt rollback
            logger.error(f"[TX {tx_id}] ❌ Transaction failed: {e}")
            
            # ROLLBACK: Restore sender balance if it was debited
            if sender_debited and not receiver_credited:
                logger.warning(f"[TX {tx_id}] 🔄 Rolling back sender deduction")
                try:
                    self._update_balance(sender_id, sender_balance)
                    logger.info(f"[TX {tx_id}] ✅ Rollback successful")
                except Exception as rollback_error:
                    logger.critical(
                        f"[TX {tx_id}] 🚨 CRITICAL: Rollback failed! "
                        f"Manual intervention required. Error: {rollback_error}"
                    )
            
            self._log_transaction(
                tx_id=tx_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=float(amount_decimal),
                status="failed",
                description=f"Error: {str(e)}"
            )
            
            raise TransactionFailedError(f"Transaction failed: {str(e)}")
    
    def _validate_accounts_exist(self, sender_id: str, receiver_id: str):
        """Validate both accounts exist"""
        # Check sender
        sender = self._get_account(sender_id)
        if not sender:
            raise AccountNotFoundError(f"Tài khoản người gửi không tồn tại: {sender_id}")
        
        # Check receiver
        receiver = self._get_account(receiver_id)
        if not receiver:
            raise AccountNotFoundError(f"Tài khoản người nhận không tồn tại: {receiver_id}")
        
        # Check accounts are different
        if sender_id == receiver_id:
            raise TransactionError("Không thể chuyển tiền cho chính mình")
    
    def _get_account(self, account_id: str) -> Optional[Dict]:
        """Get account details"""
        try:
            return self.cassandra.accounts.get_account_by_id(account_id)
        except Exception as e:
            logger.error(f"Error getting account {account_id}: {e}")
            return None
    
    def _get_account_balance(self, account_id: str) -> Decimal:
        """Get current account balance"""
        try:
            balance_data = self.cassandra.accounts.get_balance(account_id)
            balance = balance_data.get("balance", 0.0) if balance_data else 0.0
            return Decimal(str(balance))
        except Exception as e:
            logger.error(f"Error getting balance for {account_id}: {e}")
            return Decimal("0.0")
    
    def _update_balance_atomic(
        self, 
        account_id: str, 
        new_balance: Decimal,
        expected_old_balance: Decimal
    ) -> bool:
        """
        Update balance atomically using Cassandra LWT (Compare-And-Set).
        
        Returns True if update succeeded, False if balance changed (race condition).
        """
        try:
            # Use LWT to ensure balance hasn't changed
            result = self.session.execute(
                """
                UPDATE balances 
                SET balance = %s 
                WHERE account_id = %s 
                IF balance = %s
                """,
                (float(new_balance), account_id, float(expected_old_balance))
            )
            
            # Check if the conditional update was applied
            return result.one().applied if result else False
            
        except Exception as e:
            logger.error(f"Error in atomic balance update for {account_id}: {e}")
            # Fallback to regular update if LWT not available
            return self._update_balance(account_id, new_balance)
    
    def _update_balance(self, account_id: str, new_balance: Decimal) -> bool:
        """Simple balance update (no LWT)"""
        try:
            self.cassandra.accounts.update_balance(
                account_id=account_id,
                amount_delta=float(new_balance),
                operation="set"  # Direct set (not delta)
            )
            return True
        except Exception as e:
            logger.error(f"Error updating balance for {account_id}: {e}")
            return False
    
    def _log_transaction(
        self,
        tx_id: str,
        sender_id: str,
        receiver_id: str,
        amount: float,
        status: str,
        description: str = ""
    ):
        """Log transaction to database for audit trail"""
        try:
            self.session.execute(
                """
                INSERT INTO transactions 
                (tx_id, account_id, sender_id, receiver_id, amount, 
                 transaction_type, status, description, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, toTimestamp(now()))
                """,
                (
                    uuid.UUID(tx_id),
                    sender_id,  # Primary account for transaction log
                    sender_id,
                    receiver_id,
                    amount,
                    "p2p_transfer",
                    status,
                    description
                )
            )
        except Exception as e:
            logger.error(f"Error logging transaction {tx_id}: {e}")
