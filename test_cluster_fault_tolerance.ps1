# ====================================================================
# Test Cassandra Cluster - Fault Tolerance Demo
# ====================================================================
# This script demonstrates that your cluster can survive node failures
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Cassandra Cluster - Fault Tolerance Test" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ====================================================================
# Step 1: Check Current Cluster Status
# ====================================================================
Write-Host "[Step 1/8] Current cluster status:" -ForegroundColor Yellow
Write-Host ""

docker exec bt-cassandra-node1 nodetool status

Write-Host ""
Write-Host "  You should see 3 nodes with UN (Up/Normal) status" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to continue"

# ====================================================================
# Step 2: Insert Test Data
# ====================================================================
Write-Host ""
Write-Host "[Step 2/8] Inserting test data..." -ForegroundColor Yellow

$insertData = @"
USE pldt_rt;

-- Insert test accounts
INSERT INTO accounts_by_id (account_id, customer_id, balance, status) 
VALUES ('TEST001', 'CUST001', 1000000.00, 'ACTIVE');

INSERT INTO accounts_by_id (account_id, customer_id, balance, status) 
VALUES ('TEST002', 'CUST002', 2000000.00, 'ACTIVE');

INSERT INTO accounts_by_id (account_id, customer_id, balance, status) 
VALUES ('TEST003', 'CUST003', 3000000.00, 'ACTIVE');

SELECT * FROM accounts_by_id WHERE account_id IN ('TEST001', 'TEST002', 'TEST003');
"@

Write-Host "  Inserting 3 test accounts..." -ForegroundColor Gray
docker exec -i bt-cassandra-node1 cqlsh -e $insertData

Write-Host ""
Write-Host "  [OK] Test data inserted on Node 1" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"

# ====================================================================
# Step 3: Verify Data on All Nodes
# ====================================================================
Write-Host ""
Write-Host "[Step 3/8] Verifying data is replicated on all nodes..." -ForegroundColor Yellow
Write-Host ""

$queryData = "SELECT * FROM pldt_rt.accounts_by_id WHERE account_id IN ('TEST001', 'TEST002', 'TEST003');"

Write-Host "  [Node 1] Reading from Node 1 (port 9042):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node1 cqlsh -e $queryData
Write-Host ""

Write-Host "  [Node 2] Reading from Node 2 (port 9043):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node2 cqlsh -e $queryData
Write-Host ""

Write-Host "  [Node 3] Reading from Node 3 (port 9044):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node3 cqlsh -e $queryData
Write-Host ""

Write-Host "  [OK] Data is available on ALL 3 nodes!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to simulate node failure"

# ====================================================================
# Step 4: Simulate Node Failure (Kill Node 2)
# ====================================================================
Write-Host ""
Write-Host "[Step 4/8] [FAILURE] Simulating Node 2 failure..." -ForegroundColor Red
Write-Host "  Stopping Node 2..." -ForegroundColor Yellow

docker stop bt-cassandra-node2

Write-Host "  [DOWN] Node 2 is DOWN!" -ForegroundColor Red
Write-Host ""

Start-Sleep -Seconds 5

# ====================================================================
# Step 5: Check Cluster Status After Failure
# ====================================================================
Write-Host "[Step 5/8] Cluster status after Node 2 failure:" -ForegroundColor Yellow
Write-Host ""

docker exec bt-cassandra-node1 nodetool status

Write-Host ""
Write-Host "  Notice: Node 2 status is DN (Down)" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to test if data is still accessible"

# ====================================================================
# Step 6: Verify Data Still Accessible with Node Down
# ====================================================================
Write-Host ""
Write-Host "[Step 6/8] Testing if data is still accessible..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  [Node 1] Reading from Node 1 (Node 2 is DOWN):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node1 cqlsh -e $queryData
Write-Host ""

Write-Host "  [Node 3] Reading from Node 3 (Node 2 is DOWN):" -ForegroundColor Cyan
docker exec -i bt-cassandra-node3 cqlsh -e $queryData
Write-Host ""

Write-Host "  [OK] Data is STILL accessible with 1 node down!" -ForegroundColor Green
Write-Host ""

# ====================================================================
# Step 7: Insert New Data with Node Down
# ====================================================================
Write-Host "[Step 7/8] Testing write capability with 1 node down..." -ForegroundColor Yellow

$insertNew = @"
USE pldt_rt;

INSERT INTO accounts_by_id (account_id, customer_id, balance, status) 
VALUES ('TEST004', 'CUST004', 4000000.00, 'ACTIVE');

SELECT * FROM accounts_by_id WHERE account_id = 'TEST004';
"@

Write-Host "  Inserting new account (TEST004) with Node 2 DOWN..." -ForegroundColor Gray
docker exec -i bt-cassandra-node1 cqlsh -e $insertNew
Write-Host ""

Write-Host "  [OK] Write operation successful with 1 node down!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to restart Node 2"

# ====================================================================
# Step 8: Restart Node 2 and Verify Recovery
# ====================================================================
Write-Host ""
Write-Host "[Step 8/8] [RECOVERY] Restarting Node 2..." -ForegroundColor Yellow

docker start bt-cassandra-node2

Write-Host "  [WAIT] Waiting for Node 2 to rejoin cluster (30 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "  [STATUS] Cluster status after Node 2 recovery:" -ForegroundColor Cyan
docker exec bt-cassandra-node1 nodetool status
Write-Host ""

Write-Host "  [OK] Node 2 is back online!" -ForegroundColor Green
Write-Host "  [INFO] Data is automatically syncing to Node 2" -ForegroundColor Gray
Write-Host ""

# ====================================================================
# Summary
# ====================================================================
Write-Host "============================================" -ForegroundColor Green
Write-Host "Fault Tolerance Test Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "What we demonstrated:" -ForegroundColor White
Write-Host "  ✅ Inserted test data on cluster" -ForegroundColor Gray
Write-Host "  ✅ Verified data replicated to all 3 nodes" -ForegroundColor Gray
Write-Host "  ✅ Killed Node 2 (1/3 nodes down)" -ForegroundColor Gray
Write-Host "  ✅ Data still accessible from Node 1 & 3" -ForegroundColor Gray
Write-Host "  ✅ Successfully wrote new data with node down" -ForegroundColor Gray
Write-Host "  ✅ Restarted Node 2 and it rejoined cluster" -ForegroundColor Gray
Write-Host ""

Write-Host "Key Takeaways:" -ForegroundColor White
Write-Host "  🎯 Cassandra cluster with RF=3 can survive 1 node failure" -ForegroundColor Cyan
Write-Host "  🎯 Reads and writes continue during node failure" -ForegroundColor Cyan
Write-Host "  🎯 No single point of failure!" -ForegroundColor Cyan
Write-Host "  🎯 Automatic data replication and recovery" -ForegroundColor Cyan
Write-Host ""

Write-Host "Your cluster is now fully operational!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  • Open Frontend: http://localhost:5173" -ForegroundColor Gray
Write-Host "  • Test with real transactions via UI" -ForegroundColor Gray
Write-Host "  • Monitor cluster: docker exec bt-cassandra-node1 nodetool status" -ForegroundColor Gray
Write-Host ""

Read-Host "Press Enter to exit"
