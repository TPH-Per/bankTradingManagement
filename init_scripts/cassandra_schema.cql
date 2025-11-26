-- ====================================================================
-- Bank Trading Management - Cassandra Schema
-- ====================================================================
-- This script creates the complete schema for the banking system
-- Run this after starting a fresh Cassandra instance
-- ====================================================================

-- Create keyspace
CREATE KEYSPACE IF NOT EXISTS pldt_rt 
WITH replication = {
  'class': 'SimpleStrategy', 
  'replication_factor': 1
};

USE pldt_rt;

-- ====================================================================
-- Accounts & Customers
-- ====================================================================

CREATE TABLE IF NOT EXISTS accounts_by_id (
    account_id text PRIMARY KEY,
    customer_id text,
    account_type text,
    balance decimal,
    created_at timestamp,
    updated_at timestamp,
    status text,
    extra_json text
);

CREATE TABLE IF NOT EXISTS accounts_by_customer (
    customer_id text,
    account_id text,
    account_type text,
    balance decimal,
    created_at timestamp,
    status text,
    PRIMARY KEY (customer_id, account_id)
);

CREATE TABLE IF NOT EXISTS customers_by_email (
    email text PRIMARY KEY,
    customer_id text,
    full_name text,
    phone text,
    national_id text,
    status text,
    created_at timestamp,
    updated_at timestamp,
    extra_json text
);

CREATE TABLE IF NOT EXISTS customers_by_phone (
    phone text PRIMARY KEY,
    customer_id text,
    full_name text,
    email text,
    national_id text,
    status text,
    created_at timestamp,
    updated_at timestamp,
    extra_json text
);

CREATE TABLE IF NOT EXISTS customers_by_identity (
    national_id text PRIMARY KEY,
    customer_id text,
    full_name text,
    email text,
    phone text,
    status text,
    created_at timestamp,
    updated_at timestamp,
    extra_json text
);

-- ====================================================================
-- Transactions
-- ====================================================================

CREATE TABLE IF NOT EXISTS tx_by_account_day (
    account_id text,
    day text,
    tx_id timeuuid,
    tx_type text,
    amount decimal,
    balance_after decimal,
    description text,
    created_at timestamp,
    extra_json text,
    PRIMARY KEY ((account_id, day), tx_id)
) WITH CLUSTERING ORDER BY (tx_id DESC);

CREATE TABLE IF NOT EXISTS transfers_by_id (
    tx_id text PRIMARY KEY,
    from_account text,
    to_account text,
    amount decimal,
    description text,
    status text,
    created_at timestamp,
    completed_at timestamp,
    extra_json text
);

-- ====================================================================
-- P2P Transfers
-- ====================================================================

CREATE TABLE IF NOT EXISTS p2p_tx_by_account_pair_month (
    account_pair text,
    month text,
    tx_id timeuuid,
    from_account text,
    to_account text,
    amount decimal,
    created_at timestamp,
    PRIMARY KEY ((account_pair, month), tx_id)
) WITH CLUSTERING ORDER BY (tx_id DESC);

CREATE TABLE IF NOT EXISTS transfer_dedup (
    idempotency_key text PRIMARY KEY,
    tx_id text,
    status text,
    created_at timestamp,
    expires_at timestamp
);

-- ====================================================================
-- Balances
-- ====================================================================

CREATE TABLE IF NOT EXISTS account_balances (
    account_id text PRIMARY KEY,
    balance decimal,
    reserved decimal,
    last_updated timestamp
);

-- ====================================================================
-- Audit Log
-- ====================================================================
CREATE KEYSPACE IF NOT EXISTS pldt_audit 
WITH replication = {
  'class': 'SimpleStrategy', 
  'replication_factor': 1
};

USE pldt_audit;

CREATE TABLE IF NOT EXISTS api_calls (
    call_id timeuuid PRIMARY KEY,
    endpoint text,
    method text,
    user_id text,
    ip_address text,
    request_body text,
    response_status int,
    duration_ms int,
    created_at timestamp
);
