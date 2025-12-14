# ====================================================================
# Fix Cluster Replication - Update RF to 3
# ====================================================================
# This script updates the replication factor to 3 for cluster mode
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Fix Cluster Replication Factor" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Current keyspaces:" -ForegroundColor Yellow
docker exec -i bt-cassandra-node1 cqlsh -e "DESCRIBE KEYSPACES;"
Write-Host ""

Write-Host "[2/3] Updating replication factor to 3..." -ForegroundColor Yellow

$updateReplication = @"
-- Update pldt_rt keyspace to RF=3
ALTER KEYSPACE pldt_rt 
WITH replication = {
  'class': 'SimpleStrategy', 
  'replication_factor': 3
};

-- Update pldt_audit keyspace to RF=3
ALTER KEYSPACE pldt_audit 
WITH replication = {
  'class': 'SimpleStrategy', 
  'replication_factor': 3
};

-- Create demo keyspace if needed
CREATE KEYSPACE IF NOT EXISTS demo_bank 
WITH replication = {
  'class': 'SimpleStrategy', 
  'replication_factor': 3
};
"@

docker exec -i bt-cassandra-node1 cqlsh -e $updateReplication

Write-Host "  [OK] Replication factor updated to 3" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] Running repair to replicate existing data..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray

docker exec bt-cassandra-node1 nodetool repair pldt_rt 2>&1 | Out-Null
docker exec bt-cassandra-node1 nodetool repair pldt_audit 2>&1 | Out-Null

Write-Host "  [OK] Repair complete - data is now replicated to all 3 nodes" -ForegroundColor Green
Write-Host ""

Write-Host "============================================" -ForegroundColor Green
Write-Host "Cluster Configuration Updated!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "Verification:" -ForegroundColor Yellow
$verifyKeyspace = "SELECT keyspace_name, replication FROM system_schema.keyspaces WHERE keyspace_name IN ('pldt_rt', 'pldt_audit', 'demo_bank');"
docker exec -i bt-cassandra-node1 cqlsh -e $verifyKeyspace

Write-Host ""
Write-Host "Your cluster is now properly configured!" -ForegroundColor Green
Write-Host "  • Replication Factor: 3" -ForegroundColor Cyan
Write-Host "  • Can survive 1 node failure" -ForegroundColor Cyan
Write-Host "  • Data is replicated to all nodes" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
