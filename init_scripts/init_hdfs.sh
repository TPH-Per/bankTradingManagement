#!/bin/bash
# HDFS Initialization Script
# This script creates the necessary directories in HDFS for the bank trading system

set -e

echo "Waiting for HDFS NameNode to be ready..."
sleep 30

# Wait for NameNode to be available
until hdfs dfsadmin -report 2>/dev/null; do
    echo "Waiting for HDFS NameNode..."
    sleep 5
done

echo "HDFS NameNode is ready!"

# Create base directory structure
echo "Creating HDFS directory structure..."

# Base directory
hdfs dfs -mkdir -p /banktrading

# Archived data directories
hdfs dfs -mkdir -p /banktrading/archived/transactions
hdfs dfs -mkdir -p /banktrading/archived/audit_logs
hdfs dfs -mkdir -p /banktrading/archived/balance_snapshots

# Analytics directories
hdfs dfs -mkdir -p /banktrading/analytics/daily_aggregates
hdfs dfs -mkdir -p /banktrading/analytics/monthly_kpis
hdfs dfs -mkdir -p /banktrading/analytics/channel_performance

# Model directories
hdfs dfs -mkdir -p /banktrading/models/training_data
hdfs dfs -mkdir -p /banktrading/models/model_artifacts

# Streaming directory (for future use)
hdfs dfs -mkdir -p /banktrading/streaming/transactions

# Compliance directory
hdfs dfs -mkdir -p /banktrading/compliance/reports

# Set permissions (allow all for dev/test)
hdfs dfs -chmod -R 777 /banktrading

echo "HDFS directory structure created successfully!"
echo "Base path: hdfs://namenode:9000/banktrading"

# Display directory structure
echo ""
echo "Directory structure:"
hdfs dfs -ls -R /banktrading

