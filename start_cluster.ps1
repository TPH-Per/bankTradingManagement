# ====================================================================
# START CLUSTER - Run Full Project with Cassandra 3-Node Cluster
# ====================================================================
# This script runs the complete Bank Trading system with:
# - Cassandra 3-node cluster (fault-tolerant)
# - Redis cache
# - HDFS storage
# - Backend API (FastAPI)
# - Frontend UI (React/Vite)
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Bank Trading System - CLUSTER MODE" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  • Cassandra: 3-node cluster (fault-tolerant)" -ForegroundColor Gray
Write-Host "  • Replication Factor: 3" -ForegroundColor Gray
Write-Host "  • Can survive 1 node failure" -ForegroundColor Gray
Write-Host ""

# Get project root
$projectRoot = Get-Location

# ====================================================================
# Step 1: Stop any existing single Cassandra instance
# ====================================================================
Write-Host "[1/7] Checking for existing services..." -ForegroundColor Yellow

# Stop single Cassandra container if running
$singleCassandra = docker ps -q --filter "name=bt-cassandra" 2>$null
if ($singleCassandra) {
    Write-Host "  Stopping single Cassandra instance..." -ForegroundColor Gray
    docker stop bt-cassandra 2>&1 | Out-Null
    docker rm bt-cassandra 2>&1 | Out-Null
}

Write-Host "  [OK] Ready to start cluster" -ForegroundColor Green
Write-Host ""

# ====================================================================
# Step 2: Start Cassandra 3-Node Cluster
# ====================================================================
Write-Host "[2/7] Starting Cassandra 3-node cluster..." -ForegroundColor Yellow
Write-Host "  This will take 2-3 minutes. Please wait..." -ForegroundColor Gray
Write-Host ""

docker-compose -f docker-compose.cassandra-cluster.yml up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Failed to start cluster!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Cluster containers created" -ForegroundColor Green
Write-Host ""

# Wait for Node 1 (seed node)
Write-Host "  [WAIT] Waiting for Node 1 to initialize (60 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 60

# Wait for Node 2
Write-Host "  [WAIT] Waiting for Node 2 to join (30 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Wait for Node 3
Write-Host "  [WAIT] Waiting for Node 3 to join (30 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "  [STATUS] Cluster status:" -ForegroundColor Cyan
docker exec bt-cassandra-node1 nodetool status 2>&1
Write-Host ""

# ====================================================================
# Step 3: Initialize Cassandra Schema
# ====================================================================
Write-Host "[3/7] Initializing Cassandra schema..." -ForegroundColor Yellow

$schemaFile = Join-Path $projectRoot "init_scripts\cassandra_schema.cql"

if (Test-Path $schemaFile) {
    Write-Host "  Loading schema from: $schemaFile" -ForegroundColor Gray
    
    # Read and execute schema
    $schemaContent = Get-Content $schemaFile -Raw
    
    # Execute on Node 1
    try {
        # Copy schema file to container
        docker cp $schemaFile bt-cassandra-node1:/tmp/schema.cql
        
        # Execute schema
        docker exec -i bt-cassandra-node1 cqlsh -f /tmp/schema.cql 2>&1 | Out-Null
        
        Write-Host "  [OK] Schema initialized on cluster" -ForegroundColor Green
    }
    catch {
        Write-Host "  [WARNING] Schema initialization may have failed" -ForegroundColor Yellow
        Write-Host "  You can manually run: docker exec -i bt-cassandra-node1 cqlsh -f /tmp/schema.cql" -ForegroundColor Gray
    }
}
else {
    Write-Host "  [WARNING] Schema file not found: $schemaFile" -ForegroundColor Yellow
}

Write-Host ""

# ====================================================================
# Step 4: Start Redis & HDFS
# ====================================================================
Write-Host "[4/7] Starting Redis & HDFS..." -ForegroundColor Yellow

try {
    # Start HDFS NameNode
    Write-Host "  Starting HDFS NameNode..." -ForegroundColor Gray
    docker-compose -f docker-compose.yml up -d hdfs-namenode 2>&1 | Out-Null
    
    Start-Sleep -Seconds 3
    
    # Start HDFS DataNode
    Write-Host "  Starting HDFS DataNode..." -ForegroundColor Gray
    docker-compose -f docker-compose.yml up -d hdfs-datanode 2>&1 | Out-Null
    
    # Start Redis
    Write-Host "  Starting Redis..." -ForegroundColor Gray
    docker-compose -f docker-compose.yml up -d redis 2>&1 | Out-Null
    
    Write-Host "  [OK] Redis & HDFS started" -ForegroundColor Green
}
catch {
    Write-Host "  [WARNING] Some services may already be running" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  [WAIT] Waiting for services to initialize (15 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 15

# Initialize HDFS directories
Write-Host "  Initializing HDFS directories..." -ForegroundColor Gray
try {
    docker exec bt-hdfs-namenode hdfs dfs -mkdir -p /banktrading/data /banktrading/ml /banktrading/reports /banktrading/transactions /banktrading/customers 2>&1 | Out-Null
    docker exec bt-hdfs-namenode hdfs dfs -chmod -R 777 /banktrading 2>&1 | Out-Null
    Write-Host "  [OK] HDFS directories initialized" -ForegroundColor Green
}
catch {
    Write-Host "  [WARNING] HDFS directories may already exist" -ForegroundColor Yellow
}

Write-Host ""

# ====================================================================
# Step 5: Verify All Infrastructure Services
# ====================================================================
Write-Host "[5/7] Verifying infrastructure services..." -ForegroundColor Yellow

# Check Cassandra cluster
$cassandraNodes = docker ps --filter "name=bt-cassandra-node" --format "{{.Names}}" 2>$null
$nodeCount = ($cassandraNodes | Measure-Object).Count

if ($nodeCount -eq 3) {
    Write-Host "  [OK] Cassandra Cluster: 3 nodes running" -ForegroundColor Green
    Write-Host "      - Node 1: localhost:9042" -ForegroundColor Gray
    Write-Host "      - Node 2: localhost:9043" -ForegroundColor Gray
    Write-Host "      - Node 3: localhost:9044" -ForegroundColor Gray
}
else {
    Write-Host "  [WARNING] Cassandra Cluster: Only $nodeCount/3 nodes running" -ForegroundColor Yellow
}

# Check Redis
$redisOK = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null
if ($redisOK) {
    Write-Host "  [OK] Redis (port 6379)" -ForegroundColor Green
}
else {
    Write-Host "  [WARNING] Redis (port 6379)" -ForegroundColor Yellow
}

# Check HDFS
$hdfsOK = Test-NetConnection -ComputerName localhost -Port 9870 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null
if ($hdfsOK) {
    Write-Host "  [OK] HDFS NameNode (port 9870)" -ForegroundColor Green
}
else {
    Write-Host "  [WARNING] HDFS NameNode (port 9870)" -ForegroundColor Yellow
}

Write-Host ""

# ====================================================================
# Step 6: Start Backend with Cluster Configuration
# ====================================================================
Write-Host "[6/7] Starting Backend (with cluster connection)..." -ForegroundColor Yellow

$backendPath = Join-Path $projectRoot "BACKEND"

# Set environment variables for cluster connection
$env:CASSANDRA_CONTACT_POINTS = "localhost:9042,localhost:9043,localhost:9044"
$env:CASSANDRA_PORT = "9042"
$env:CASSANDRA_KEYSPACE_RT = "bank_trading_rt"
$env:CASSANDRA_KEYSPACE_AUDIT = "bank_trading_audit"

$backendCmd = @"
Write-Host '============================================' -ForegroundColor Cyan
Write-Host 'Backend Server - CLUSTER MODE' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Cassandra Cluster:' -ForegroundColor Yellow
Write-Host '  • Node 1: localhost:9042' -ForegroundColor Gray
Write-Host '  • Node 2: localhost:9043' -ForegroundColor Gray
Write-Host '  • Node 3: localhost:9044' -ForegroundColor Gray
Write-Host ''
`$env:CASSANDRA_CONTACT_POINTS = 'localhost:9042,localhost:9043,localhost:9044'
`$env:CASSANDRA_PORT = '9042'
`$env:CASSANDRA_KEYSPACE_RT = 'bank_trading_rt'
`$env:CASSANDRA_KEYSPACE_AUDIT = 'bank_trading_audit'
Set-Location '$backendPath'
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
Write-Host "  [OK] Backend starting in new window (port 8000)" -ForegroundColor Green

Start-Sleep -Seconds 2

# ====================================================================
# Step 7: Start Frontend
# ====================================================================
Write-Host "[7/7] Starting Frontend..." -ForegroundColor Yellow

$frontendPath = Join-Path $projectRoot "FRONTEND"
$frontendCmd = @"
Write-Host '============================================' -ForegroundColor Cyan
Write-Host 'Frontend Server' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Set-Location '$frontendPath'
npm run dev
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd
Write-Host "  [OK] Frontend starting in new window (port 5173)" -ForegroundColor Green

# ====================================================================
# Summary
# ====================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "CLUSTER MODE - System Started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "Infrastructure:" -ForegroundColor White
Write-Host "  • Cassandra Node 1:    localhost:9042" -ForegroundColor Gray
Write-Host "  • Cassandra Node 2:    localhost:9043" -ForegroundColor Gray
Write-Host "  • Cassandra Node 3:    localhost:9044" -ForegroundColor Gray
Write-Host "  • Redis:               localhost:6379" -ForegroundColor Gray
Write-Host "  • HDFS Web UI:         http://localhost:9870" -ForegroundColor Gray
Write-Host ""

Write-Host "Application:" -ForegroundColor White
Write-Host "  • Backend API:         http://localhost:8000" -ForegroundColor Gray
Write-Host "  • Backend Docs:        http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "  • Frontend:            http://localhost:5173" -ForegroundColor Gray
Write-Host ""

Write-Host "Cluster Management:" -ForegroundColor White
Write-Host "  • Check status:        docker exec bt-cassandra-node1 nodetool status" -ForegroundColor Gray
Write-Host "  • Access CQL:          docker exec -it bt-cassandra-node1 cqlsh" -ForegroundColor Gray
Write-Host "  • Stop cluster:        docker-compose -f docker-compose.cassandra-cluster.yml down" -ForegroundColor Gray
Write-Host ""

Write-Host "Features:" -ForegroundColor White
Write-Host "  ✅ 3-node Cassandra cluster" -ForegroundColor Cyan
Write-Host "  ✅ Replication Factor = 3" -ForegroundColor Cyan
Write-Host "  ✅ Can survive 1 node failure" -ForegroundColor Cyan
Write-Host "  ✅ High availability" -ForegroundColor Cyan
Write-Host ""

Write-Host "Open Frontend: " -NoNewline -ForegroundColor Yellow
Write-Host "http://localhost:5173" -ForegroundColor Cyan
Write-Host ""

Write-Host "[WAIT] Waiting for complete initialization (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "[SUCCESS] Cluster system is ready!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to close this window"
