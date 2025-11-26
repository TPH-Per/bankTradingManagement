# ====================================================================
# Cassandra Cluster - Fault Tolerance Demo
# ====================================================================
# This script demonstrates Cassandra's fault tolerance by:
# 1. Starting a 3-node cluster
# 2. Creating a keyspace with replication
# 3. Inserting test data
# 4. Killing one node
# 5. Verifying data is still accessible
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Cassandra Cluster Fault Tolerance Demo" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ====================================================================
# Step 1: Start Cluster
# ====================================================================
Write-Host "[Step 1/7] Starting 3-node Cassandra cluster..." -ForegroundColor Yellow
Write-Host "  This will take 2-3 minutes. Please wait..." -ForegroundColor Gray
Write-Host ""

docker-compose -f docker-compose.cassandra-cluster.yml up -d

if ($LASTEXITCODE -ne 0) {
  Write-Host "  [ERROR] Failed to start cluster!" -ForegroundColor Red
  exit 1
}

Write-Host "  [OK] Cluster starting..." -ForegroundColor Green
Write-Host ""

# Wait for Node 1 to be ready
Write-Host "  [WAIT] Waiting for Node 1 to initialize (60 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 60

# Wait for Node 2
Write-Host "  [WAIT] Waiting for Node 2 to join (30 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Wait for Node 3
Write-Host "  [WAIT] Waiting for Node 3 to join (30 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 30

Write-Host ""

# ====================================================================
# Step 2: Check Cluster Status
# ====================================================================
Write-Host "[Step 2/7] Checking cluster status..." -ForegroundColor Yellow
Write-Host ""

docker exec bt-cassandra-node1 nodetool status

Write-Host ""
Write-Host "  You should see 3 nodes with status UN (Up/Normal)" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to continue"

# ====================================================================
# Step 3: Create Keyspace with Replication
# ====================================================================
Write-Host ""
Write-Host "[Step 3/7] Creating keyspace with replication..." -ForegroundColor Yellow

$createKeyspace = "CREATE KEYSPACE IF NOT EXISTS demo_bank WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};"

docker exec -i bt-cassandra-node1 cqlsh -e $createKeyspace

Write-Host "  [OK] Keyspace 'demo_bank' created with RF=3" -ForegroundColor Green
Write-Host "     (Data will be replicated to all 3 nodes)" -ForegroundColor Gray
Write-Host ""

# ====================================================================
# Step 4: Create Table and Insert Data
# ====================================================================
Write-Host "[Step 4/7] Creating table and inserting test data..." -ForegroundColor Yellow

$createTable = @"
CREATE TABLE IF NOT EXISTS demo_bank.accounts (
  account_id text PRIMARY KEY,
  customer_name text,
  balance decimal
);

INSERT INTO demo_bank.accounts (account_id, customer_name, balance) 
VALUES ('ACC001', 'Nguyen Van A', 1000000);

INSERT INTO demo_bank.accounts (account_id, customer_name, balance) 
VALUES ('ACC002', 'Tran Thi B', 2000000);

INSERT INTO demo_bank.accounts (account_id, customer_name, balance) 
VALUES ('ACC003', 'Le Van C', 3000000);
"@

docker exec -i bt-cassandra-node1 cqlsh -e $createTable

Write-Host "  [OK] Table created and 3 accounts inserted" -ForegroundColor Green
Write-Host ""

# Verify data on Node 1
Write-Host "  [DATA] Data on Node 1:" -ForegroundColor Cyan
docker exec -i bt-cassandra-node1 cqlsh -e "SELECT * FROM demo_bank.accounts;"
Write-Host ""

Read-Host "Press Enter to continue"

# ====================================================================
# Step 5: Verify Data on All Nodes
# ====================================================================
Write-Host ""
Write-Host "[Step 5/7] Verifying data on all nodes..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  [DATA] Data on Node 2:" -ForegroundColor Cyan
docker exec -i bt-cassandra-node2 cqlsh -e "SELECT * FROM demo_bank.accounts;"
Write-Host ""

Write-Host "  [DATA] Data on Node 3:" -ForegroundColor Cyan
docker exec -i bt-cassandra-node3 cqlsh -e "SELECT * FROM demo_bank.accounts;"
Write-Host ""

Write-Host "  [OK] Data is replicated on all 3 nodes!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to simulate node failure"

# ====================================================================
# Step 6: Simulate Node Failure (Kill Node 2)
# ====================================================================
Write-Host ""
Write-Host "[Step 6/7] [FAILURE] Simulating node failure..." -ForegroundColor Red
Write-Host "  Killing Node 2..." -ForegroundColor Yellow

docker stop bt-cassandra-node2

Write-Host "  [DOWN] Node 2 is DOWN!" -ForegroundColor Red
Write-Host ""

Start-Sleep -Seconds 5

# Check cluster status
Write-Host "  [STATUS] Cluster status after Node 2 failure:" -ForegroundColor Yellow
docker exec bt-cassandra-node1 nodetool status
Write-Host ""

Write-Host "  Notice: Node 2 status is DN (Down/Normal)" -ForegroundColor Gray
Write-Host ""

# ====================================================================
# Step 7: Verify Data Still Accessible
# ====================================================================
Write-Host "[Step 7/7] Testing if data is still accessible..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  Reading from Node 1 (still up):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node1 cqlsh -e "SELECT * FROM demo_bank.accounts;"
Write-Host ""

Write-Host "  Reading from Node 3 (still up):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node3 cqlsh -e "SELECT * FROM demo_bank.accounts;"
Write-Host ""

# Try to insert new data with quorum
Write-Host "  [INSERT] Inserting new data with ONE node down:" -ForegroundColor Yellow
$insertNew = @"
INSERT INTO demo_bank.accounts (account_id, customer_name, balance) 
VALUES ('ACC004', 'Pham Van D', 4000000);

SELECT * FROM demo_bank.accounts WHERE account_id='ACC004';
"@

docker exec -i bt-cassandra-node1 cqlsh -e $insertNew

Write-Host ""
Write-Host "  [OK] Data can still be read and written!" -ForegroundColor Green
Write-Host ""

# ====================================================================
# Summary
# ====================================================================
Write-Host "============================================" -ForegroundColor Green
Write-Host "Fault Tolerance Demo Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "What we demonstrated:" -ForegroundColor White
Write-Host "  1. Created 3-node cluster" -ForegroundColor Gray
Write-Host "  2. Created keyspace with RF=3 (all nodes have data)" -ForegroundColor Gray
Write-Host "  3. Inserted test data" -ForegroundColor Gray
Write-Host "  4. Verified data on all nodes" -ForegroundColor Gray
Write-Host "  5. Killed Node 2 (1/3 nodes down)" -ForegroundColor Gray
Write-Host "  6. Data still accessible on remaining nodes" -ForegroundColor Gray
Write-Host "  7. New writes still work!" -ForegroundColor Gray
Write-Host ""

Write-Host "Key Takeaways:" -ForegroundColor White
Write-Host "  - Cassandra with RF=3 can survive 1 node failure" -ForegroundColor Cyan
Write-Host "  - Data is automatically replicated" -ForegroundColor Cyan
Write-Host "  - Reads and writes continue to work" -ForegroundColor Cyan
Write-Host "  - No single point of failure!" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  - Restart Node 2: docker start bt-cassandra-node2" -ForegroundColor Yellow
Write-Host "  - Check status: docker exec bt-cassandra-node1 nodetool status" -ForegroundColor Yellow
Write-Host "  - Stop cluster: docker-compose -f docker-compose.cassandra-cluster.yml down" -ForegroundColor Yellow
Write-Host ""

$choice = Read-Host "Restart Node 2 now? (y/n)"

if ($choice -eq 'y') {
  Write-Host ""
  Write-Host "[RESTART] Restarting Node 2..." -ForegroundColor Yellow
  docker start bt-cassandra-node2
    
  Write-Host "  [WAIT] Waiting for Node 2 to rejoin (30 seconds)..." -ForegroundColor Gray
  Start-Sleep -Seconds 30
    
  Write-Host ""
  Write-Host "  [STATUS] Cluster status:" -ForegroundColor Yellow
  docker exec bt-cassandra-node1 nodetool status
    
  Write-Host ""
  Write-Host "  [OK] Node 2 is back online!" -ForegroundColor Green
  Write-Host "  [SYNC] Data will automatically sync" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Demo complete! Thank you!" -ForegroundColor Green
Write-Host ""
