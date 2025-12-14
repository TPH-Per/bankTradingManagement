# setup_hadoop_windows.ps1
# Script to download and setup Hadoop binaries for PySpark on Windows

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "SETUP HADOOP FOR PYSPARK ON WINDOWS" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$HadoopHome = "C:\hadoop"
$HadoopBin = "$HadoopHome\bin"

# Create directories
Write-Host "`n[1/5] Creating directories..." -ForegroundColor Yellow
if (-not (Test-Path $HadoopHome)) {
    New-Item -ItemType Directory -Path $HadoopHome -Force | Out-Null
}
if (-not (Test-Path $HadoopBin)) {
    New-Item -ItemType Directory -Path $HadoopBin -Force | Out-Null
}

# Download winutils.exe and hadoop.dll for Hadoop 3.3.1
Write-Host "[2/5] Downloading Hadoop binaries (winutils.exe, hadoop.dll)..." -ForegroundColor Yellow

$baseUrl = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.1/bin"
$files = @("winutils.exe", "hadoop.dll", "hdfs.dll")

foreach ($file in $files) {
    $url = "$baseUrl/$file"
    $dest = "$HadoopBin\$file"
    
    Write-Host "  Downloading $file..." -NoNewline
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        Write-Host " OK" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
    }
}

# Set environment variables
Write-Host "[3/5] Setting HADOOP_HOME environment variable..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", $HadoopHome, "User")
$env:HADOOP_HOME = $HadoopHome

Write-Host "[4/5] Adding to PATH..." -ForegroundColor Yellow
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$HadoopBin*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$HadoopBin", "User")
}
$env:PATH = "$env:PATH;$HadoopBin"

# Verify installation
Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow
Write-Host ""

$allFound = $true
foreach ($file in $files) {
    $path = "$HadoopBin\$file"
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        Write-Host "  [OK] $file ($size bytes)" -ForegroundColor Green
    }
    else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
        $allFound = $false
    }
}

Write-Host ""
Write-Host "HADOOP_HOME = $env:HADOOP_HOME" -ForegroundColor Cyan

if ($allFound) {
    Write-Host "`n[SUCCESS] Hadoop setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: Close this terminal and open a NEW PowerShell window"
    Write-Host "Then run:" -ForegroundColor Yellow
    Write-Host "  cd C:\temp\bankTradingManagement-master\bankTradingManagement-master\BACKEND"
    Write-Host "  python spark-etl.py --mode local --local-base data"
}
else {
    Write-Host "`n[WARNING] Some files are missing. Please download manually:" -ForegroundColor Yellow
    Write-Host "  https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.1/bin"
}
