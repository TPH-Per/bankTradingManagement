"""
Script to seed Cassandra with sample transactions for M5P model testing.
Creates 30 days of realistic transaction data.
"""
import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
import sys

# Try to connect to Cassandra
try:
    from cassandra.cluster import Cluster
    from cassandra.query import SimpleStatement
except ImportError:
    print("ERROR: cassandra-driver not installed. Run: pip install cassandra-driver")
    sys.exit(1)

# Configuration
CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
KEYSPACE = "bank_trading"

# Sample data parameters
NUM_ACCOUNTS = 5
NUM_DAYS = 30  # At least 7 days needed for ML features

def generate_account_id():
    return f"ACC-{uuid.uuid4().hex[:8].upper()}"

def connect_cassandra():
    print(f"Connecting to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT}...")
    cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
    session = cluster.connect(KEYSPACE)
    print("Connected successfully!")
    return cluster, session

def create_sample_accounts(session, account_ids):
    print(f"\n[1/3] Creating {len(account_ids)} sample accounts...")
    
    # Schema: account_id, customer_id, currency, status, opened_at, updated_at, extra_json
    insert_account = """
    INSERT INTO accounts_by_id (account_id, customer_id, currency, status, opened_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    prepared = session.prepare(insert_account)
    
    for acc_id in account_ids:
        customer_id = f"CUST-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.utcnow()
        session.execute(prepared, (acc_id, customer_id, "VND", "ACTIVE", now, now))
        print(f"  Created account: {acc_id}")
    
    # Initialize balances - Schema: account_id, balance, reserved, updated_at
    insert_balance = """
    INSERT INTO balances (account_id, balance, reserved, updated_at)
    VALUES (?, ?, ?, ?)
    """
    prepared_bal = session.prepare(insert_balance)
    
    for acc_id in account_ids:
        initial_balance = Decimal(random.randint(10000000, 100000000))
        session.execute(prepared_bal, (acc_id, initial_balance, Decimal(0), datetime.utcnow()))
        print(f"  Set balance for {acc_id}: {initial_balance:,.0f} VND")

def create_sample_transactions(session, account_ids):
    print(f"\n[2/3] Creating {NUM_DAYS} days of transactions...")
    
    # Schema based on db.txt:
    # account_id, event_date, event_ts, tx_id, transfer_id, sender_id, receiver_id,
    # direction, counterparty_account_id, amount, currency, merchant,
    # transaction_type, status, description, extra_json, created_at
    insert_tx = """
    INSERT INTO transactions (
        account_id, event_date, event_ts, tx_id,
        direction, amount, currency, transaction_type, status, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    prepared = session.prepare(insert_tx)
    
    total_tx = 0
    today = datetime.utcnow().date()
    
    for day_offset in range(NUM_DAYS, 0, -1):
        tx_date = today - timedelta(days=day_offset)
        day_of_week = tx_date.weekday()
        
        # Fewer transactions on weekends
        if day_of_week >= 5:  # Saturday, Sunday
            tx_per_account = random.randint(1, 3)
        else:
            tx_per_account = random.randint(3, 8)
        
        for acc_id in account_ids:
            for _ in range(tx_per_account):
                tx_id = uuid.uuid4()
                
                # Random time during the day
                hour = random.randint(8, 20)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                event_ts = datetime.combine(tx_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute, seconds=second)
                
                # Direction and amount
                is_credit = random.choice([True, False])
                direction = "CREDIT" if is_credit else "DEBIT"
                
                # Amount varies by day of week (higher on paydays - 15th, 25th, end of month)
                base_amount = random.randint(100000, 50000000)
                if tx_date.day in [15, 25] or tx_date.day >= 28:
                    base_amount = int(base_amount * random.uniform(1.5, 3.0))
                
                amount = Decimal(base_amount)
                tx_type = "cash_in" if is_credit else "cash_out"
                
                session.execute(prepared, (
                    acc_id,
                    tx_date,
                    event_ts,
                    tx_id,
                    direction,
                    amount,
                    "VND",
                    tx_type,
                    "completed",
                    event_ts
                ))
                total_tx += 1
        
        print(f"  Day {tx_date}: {tx_per_account * len(account_ids)} transactions")
    
    print(f"\n  Total transactions created: {total_tx}")
    return total_tx

def verify_data(session):
    print("\n[3/3] Verifying data...")
    
    # Count transactions
    result = session.execute("SELECT COUNT(*) FROM transactions")
    tx_count = result.one()[0]
    print(f"  Total transactions: {tx_count}")
    
    # Count accounts
    result = session.execute("SELECT COUNT(*) FROM accounts_by_id")
    acc_count = result.one()[0]
    print(f"  Total accounts: {acc_count}")
    
    # Sample transaction
    result = session.execute("SELECT account_id, event_date, direction, amount FROM transactions LIMIT 5")
    print("\n  Sample transactions:")
    for row in result:
        print(f"    {row.account_id} | {row.event_date} | {row.direction} | {row.amount:,.0f} VND")
    
    return tx_count

def main():
    print("=" * 60)
    print("  CASSANDRA SAMPLE DATA SEEDER")
    print("  Creates 30 days of realistic transaction data")
    print("=" * 60)
    
    cluster = None
    try:
        cluster, session = connect_cassandra()
        
        # Generate account IDs
        account_ids = [generate_account_id() for _ in range(NUM_ACCOUNTS)]
        
        # Create accounts and transactions
        create_sample_accounts(session, account_ids)
        total_tx = create_sample_transactions(session, account_ids)
        verify_data(session)
        
        print("\n" + "=" * 60)
        print("  SUCCESS! Sample data created.")
        print(f"  - {NUM_ACCOUNTS} accounts")
        print(f"  - {total_tx} transactions over {NUM_DAYS} days")
        print("  - You can now use the M5P predict API")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if cluster:
            cluster.shutdown()

if __name__ == "__main__":
    main()
