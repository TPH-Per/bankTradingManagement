# ====================================================================
# Initialize Cassandra Schema
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🗄️  Initializing Cassandra Schema" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Cassandra is running
Write-Host "[1/3] Checking Cassandra..." -ForegroundColor Yellow
$cassandraRunning = docker ps --filter "name=bt-cassandra" --format "{{.Names}}" 2>$null

if (-not $cassandraRunning) {
    Write-Host "  ❌ Cassandra is not running!" -ForegroundColor Red
    Write-Host "  Please start Cassandra first:" -ForegroundColor Yellow
    Write-Host "    docker-compose up -d cassandra" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  ✅ Cassandra is running" -ForegroundColor Green
Write-Host ""

# Copy schema file to container
Write-Host "[2/3] Copying schema file..." -ForegroundColor Yellow
$schemaFile = "init_scripts\cassandra_schema.cql"

if (-not (Test-Path $schemaFile)) {
    Write-Host "  ❌ Schema file not found: $schemaFile" -ForegroundColor Red
    exit 1
}

docker cp $schemaFile bt-cassandra:/tmp/schema.cql

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Schema file copied to container" -ForegroundColor Green
}
else {
    Write-Host "  ❌ Failed to copy schema file" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Execute schema
Write-Host "[3/3] Creating schema..." -ForegroundColor Yellow
Write-Host "  This may take 10-20 seconds..." -ForegroundColor Gray
Write-Host ""

docker exec bt-cassandra cqlsh -f /tmp/schema.cql

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  ✅ Schema created successfully!" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "  ⚠️  Some errors occurred, but schema may be partially created" -ForegroundColor Yellow
}

Write-Host ""

# Verify
Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ Schema Initialization Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "📊 Verifying tables..." -ForegroundColor White
Write-Host ""

$tables = docker exec bt-cassandra cqlsh -e "DESCRIBE TABLES;" 2>$null

if ($tables) {
    Write-Host "Tables created:" -ForegroundColor Green
    Write-Host $tables -ForegroundColor Gray
}
else {
    Write-Host "Could not verify tables (this is OK if schema just created)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor White
Write-Host "  1. Restart backend to connect to new schema" -ForegroundColor Gray
Write-Host "  2. Check health: http://localhost:8000/health/detailed" -ForegroundColor Gray
Write-Host "  3. Import sample data (if needed)" -ForegroundColor Gray
Write-Host ""

Read-Host "Press Enter to close"
