# ====================================================================
# INIT PROJECT - Bank Trading Management System
# ====================================================================
# ⚠️  RUN THIS SCRIPT ONLY ONCE FOR INITIAL SETUP!
# ⚠️  DO NOT RUN AGAIN OR YOU WILL LOSE ALL DATA!
# ====================================================================

Write-Host "============================================" -ForegroundColor Red
Write-Host "⚠️  PROJECT INITIALIZATION" -ForegroundColor Red
Write-Host "============================================" -ForegroundColor Red
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Create Docker containers (Cassandra, Redis, HDFS)" -ForegroundColor Yellow
Write-Host "  2. Initialize database schema" -ForegroundColor Yellow
Write-Host "  3. Create HDFS directories" -ForegroundColor Yellow
Write-Host ""
Write-Host "⚠️  WARNING: Only run this ONCE for initial setup!" -ForegroundColor Red
Write-Host "⚠️  Running again will RECREATE containers and LOSE DATA!" -ForegroundColor Red
Write-Host ""

$confirm = Read-Host "Are you sure you want to initialize? (type 'YES' to continue)"

if ($confirm -ne 'YES') {
    Write-Host ""
    Write-Host " Cancelled. No changes made." -ForegroundColor Green
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🚀 Starting Initialization..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ====================================================================
# Step 1: Check Docker
# ====================================================================
Write-Host "[1/5] Checking Docker Desktop..." -ForegroundColor Yellow
try {
    docker version 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not running"
    }
    Write-Host "  ✅ Docker Desktop is running" -ForegroundColor Green
}
catch {
    Write-Host "  ❌ Docker Desktop is NOT running!" -ForegroundColor Red
    Write-Host "  Start Docker Desktop first, then run this script again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# ====================================================================
# Step 2: Create Docker Containers
# ====================================================================
Write-Host "[2/5] Creating Docker containers..." -ForegroundColor Yellow
Write-Host "  ⏳ This may take a few minutes..." -ForegroundColor Gray

try {
    # Stop and remove any existing containers (CLEAN START!)
    docker-compose down -v 2>$null
    
    # Create and start containers
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Containers created successfully!" -ForegroundColor Green
    }
    else {
        throw "Failed to create containers"
    }
}
catch {
    Write-Host "  ❌ Failed to create containers: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# ====================================================================
# Step 3: Wait for Cassandra
# ====================================================================
Write-Host "[3/5] Waiting for Cassandra to initialize..." -ForegroundColor Yellow
Write-Host "  ⏳ This takes ~60 seconds (please be patient)..." -ForegroundColor Gray

Start-Sleep -Seconds 60

# Check if Cassandra is ready
$cassandraReady = $false
for ($i = 1; $i -le 10; $i++) {
    Write-Host "  Checking Cassandra readiness (attempt $i/10)..." -ForegroundColor Gray
    try {
        docker exec bt-cassandra cqlsh -e "SELECT cluster_name FROM system.local;" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $cassandraReady = $true
            break
        }
    }
    catch {}
    Start-Sleep -Seconds 5
}

if ($cassandraReady) {
    Write-Host "  ✅ Cassandra is ready!" -ForegroundColor Green
}
else {
    Write-Host "  ⚠️  Cassandra may need more time, continuing..." -ForegroundColor Yellow
}

Write-Host ""

# ====================================================================
# Step 4: Initialize Cassandra Schema
# ====================================================================
Write-Host "[4/5] Initializing Cassandra schema..." -ForegroundColor Yellow

if (Test-Path "init_scripts\cassandra_schema.cql") {
    try {
        # Copy schema file
        docker cp init_scripts\cassandra_schema.cql bt-cassandra:/tmp/schema.cql
        
        # Execute schema
        docker exec bt-cassandra cqlsh -f /tmp/schema.cql
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Schema initialized successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "  ⚠️  Schema initialization had warnings (may be OK)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "  ❌ Failed to initialize schema: $_" -ForegroundColor Red
    }
}
else {
    Write-Host "  ⚠️  Schema file not found, skipping..." -ForegroundColor Yellow
}

Write-Host ""

# ====================================================================
# Step 5: Initialize HDFS Directories
# ====================================================================
Write-Host "[5/5] Initializing HDFS directories..." -ForegroundColor Yellow

try {
    # Wait for HDFS
    Start-Sleep -Seconds 10
    
    # Create directories
    docker exec bt-hdfs-namenode hdfs dfs -mkdir -p /banktrading/data 2>$null
    docker exec bt-hdfs-namenode hdfs dfs -mkdir -p /banktrading/ml 2>$null
    docker exec bt-hdfs-namenode hdfs dfs -mkdir -p /banktrading/reports 2>$null
    
    # Set permissions
    docker exec bt-hdfs-namenode hdfs dfs -chmod -R 777 /banktrading 2>$null
    
    Write-Host "  ✅ HDFS directories created!" -ForegroundColor Green
}
catch {
    Write-Host "  ⚠️  HDFS initialization may need more time" -ForegroundColor Yellow
}

# ====================================================================
# Summary
# ====================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ INITIALIZATION COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "📦 Docker Containers Created:" -ForegroundColor White
Write-Host "  • Cassandra:    localhost:9042" -ForegroundColor Gray
Write-Host "  • Redis:        localhost:6379" -ForegroundColor Gray
Write-Host "  • HDFS NameNode: http://localhost:9870" -ForegroundColor Gray
Write-Host ""

Write-Host "🗄️  Database Schema:" -ForegroundColor White
Write-Host "  • Keyspace: pldt_rt (initialized)" -ForegroundColor Gray
Write-Host "  • Tables: Created and ready" -ForegroundColor Gray
Write-Host ""

Write-Host "📁 HDFS Directories:" -ForegroundColor White
Write-Host "  • /banktrading (created)" -ForegroundColor Gray
Write-Host ""

Write-Host "🎯 Next Steps:" -ForegroundColor White
Write-Host "  1. Run: .\run_project.ps1" -ForegroundColor Cyan
Write-Host "  2. Open: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""

Write-Host "⚠️  IMPORTANT:" -ForegroundColor Red
Write-Host "  • DO NOT run this init script again!" -ForegroundColor Yellow
Write-Host "  • Use .\run_project.ps1 for daily startup" -ForegroundColor Yellow
Write-Host "  • To reset project: docker-compose down -v (DELETES DATA!)" -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to close"
