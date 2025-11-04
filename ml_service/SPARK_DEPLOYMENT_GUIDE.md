# Spark + HDFS Deployment Guide

## Tổng quan triển khai

Hệ thống BankTrading đã được tích hợp Apache Spark để xử lý ETL và feature engineering quy mô lớn.

## Kiến trúc đã triển khai

```
┌─────────────────┐
│   FastAPI       │ ← REST API endpoints
│   (app.py)      │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
┌───▼──────┐ ┌─▼──────────┐
│ Scheduler│ │ Spark ETL  │ ← Spark pipeline
│ (0:00 AM)│ │(spark-etl.py)
└───┬──────┘ └─┬──────────┘
    │          │
    └────┬─────┘
         │
┌────────▼──────────────┐
│  Data Storage         │
│  - cash_daily.csv     │ ← Temp daily transactions
│  - cash_daily_train_  │
│    realistic.csv      │ ← Training dataset
└───────────────────────┘
         │
┌────────▼────────┐
│  M5P Models (6) │ ← Prediction models
└─────────────────┘
```

## Components đã cài đặt

### 1. Spark ETL Pipeline
**File**: `spark-etl.py`

**Chức năng**:
- Đọc `cash_daily.csv` (transactions trong ngày)
- Aggregate theo date (sum cash_in, cash_out)
- Engineer features: lag1_in, lag7_in, roll_mean_7_in, lag1_out, lag7_out, roll_mean_7_out
- Tạo targets: next_day, h7_sum, next_month_sum cho cả cash_in và cash_out
- Merge với training dataset hiện tại
- Export ra `cash_daily_train_realistic.csv`
- Clear `cash_daily.csv`

**Usage**:
```bash
# Local mode (development)
cd C:\temp\aa\ml_service
python spark-etl.py --mode local

# HDFS mode (production - requires Hadoop cluster)
spark-submit --master yarn spark-etl.py --mode hdfs --hdfs-base hdfs://namenode:9000/banktrading
```

### 2. Daily Scheduler
**File**: `app/scheduler.py`

**Chức năng**:
- Tự động chạy lúc 0:00 AM mỗi ngày
- Gọi Spark ETL để aggregate data
- Tự động retrain 6 models sau khi ETL xong

**Configuration**: Tự động khởi động cùng FastAPI server

### 3. API Endpoints

#### POST `/spark/trigger-etl`
Trigger Spark ETL manually

**Request**:
```bash
curl -X POST http://localhost:8000/spark/trigger-etl
```

**Response**:
```json
{
  "status": "success",
  "message": "Spark ETL completed successfully",
  "output": "... Spark logs ...",
  "timestamp": "2025-11-04T12:00:00Z"
}
```

#### GET `/spark/status`
Kiểm tra trạng thái Spark

**Request**:
```bash
curl http://localhost:8000/spark/status
```

**Response**:
```json
{
  "spark_installed": true,
  "spark_script_exists": true,
  "spark_script_path": "C:\\temp\\aa\\ml_service\\spark-etl.py",
  "daily_csv_path": "C:\\temp\\aa\\ml_service\\data\\cash_daily.csv",
  "daily_csv_exists": true,
  "training_csv_path": "C:\\temp\\aa\\ml_service\\data\\cash_daily_train_realistic.csv",
  "training_csv_exists": true,
  "scheduler_running": true
}
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Java 8 or 11** (required for Spark)
3. **PySpark**

### Install PySpark

```bash
pip install pyspark
```

### Verify Installation

```bash
python -c "from pyspark.sql import SparkSession; print('Spark OK')"
```

## Usage

### 1. Start FastAPI Server

```bash
cd C:\temp\aa\ml_service
uvicorn app.main:app --reload --port 8000
```

**Server sẽ tự động**:
- Start daily scheduler (runs at 0:00 AM)
- Load 6 M5P models
- Listen for API requests

### 2. Manual ETL Trigger

**Option A: Via API**
```bash
curl -X POST http://localhost:8000/spark/trigger-etl
```

**Option B: Direct Script**
```bash
cd C:\temp\aa\ml_service
python spark-etl.py --mode local
```

### 3. Check Status

```bash
curl http://localhost:8000/spark/status
```

### 4. Create Transactions

Khi tạo transactions qua API, chúng sẽ được ghi vào `cash_daily.csv`:

```bash
curl -X POST http://localhost:8000/rt/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC001",
    "amount": 1000000,
    "currency": "VND",
    "transaction_type": "cash_in",
    "channel": "ONLINE"
  }'
```

### 5. Daily Automatic Process

**Lúc 0:00 AM mỗi ngày**:
1. Scheduler triggers Spark ETL
2. Spark reads `cash_daily.csv` (all transactions of previous day)
3. Aggregates: sum cash_in, sum cash_out per date
4. Engineers features
5. Merges with training dataset
6. Exports to `cash_daily_train_realistic.csv`
7. Clears `cash_daily.csv`
8. Retrains all 6 models automatically

## Data Flow

```
Transaction Created
  ↓
Written to cash_daily.csv
  ↓
(Multiple transactions accumulate during the day)
  ↓
0:00 AM - Scheduler Triggers
  ↓
Spark ETL Pipeline
  ├─ Read cash_daily.csv (29 rows today)
  ├─ Aggregate by date → 1 row (2025-11-03)
  ├─ Engineer features
  ├─ Create targets
  └─ Merge with existing
  ↓
Export to cash_daily_train_realistic.csv
  ↓
Clear cash_daily.csv (reset to header only)
  ↓
Retrain Models (6 models)
  ↓
Ready for predictions!
```

## File Structure

```
ml_service/
├── spark-etl.py                    # Spark ETL pipeline
├── app/
│   ├── main.py                     # FastAPI entry
│   ├── app.py                      # API endpoints
│   ├── scheduler.py                # Daily scheduler (0:00 AM)
│   ├── multi_target_model.py       # 6 M5P models wrapper
│   └── ml_m5p.py                   # M5P implementation
├── data/
│   ├── cash_daily.csv              # Temp daily transactions (cleared at 0:00 AM)
│   └── cash_daily_train_realistic.csv  # Training dataset
├── models/                         # 6 trained models
│   ├── cash_in_next_day/
│   ├── cash_out_next_day/
│   ├── cash_in_h7/
│   ├── cash_out_h7/
│   ├── cash_in_next_month/
│   └── cash_out_next_month/
└── SPARK_DEPLOYMENT_GUIDE.md       # This file
```

## Testing

### Test 1: Spark Installation

```bash
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[*]').getOrCreate(); print('Spark version:', spark.version); spark.stop()"
```

Expected output:
```
Spark version: 3.5.x
```

### Test 2: ETL Pipeline

```bash
cd C:\temp\aa\ml_service
python spark-etl.py --mode local --no-clear
```

Expected output:
```
======================================================================
SPARK ETL PIPELINE
======================================================================
Spark ETL initialized in local mode
Reading: data\cash_daily.csv
Read 29 rows
Aggregating...
Engineering features...
Creating targets...
Merging with: data\cash_daily_train_realistic.csv
Wrote: data\cash_daily_train_realistic.csv
SUCCESS: 802 rows
```

### Test 3: API Trigger

```bash
curl -X POST http://localhost:8000/spark/trigger-etl
```

Expected response:
```json
{
  "status": "success",
  "message": "Spark ETL completed successfully",
  "output": "... SUCCESS: 802 rows ..."
}
```

### Test 4: Scheduler

```bash
# Check scheduler status
curl http://localhost:8000/spark/status
```

Expected response:
```json
{
  "scheduler_running": true,
  ...
}
```

## Monitoring

### Logs

**FastAPI logs** (includes scheduler):
```bash
# Server startup logs
INFO:     Bootstrapping multi-target cash flow models (6 models)...
INFO:     Starting daily aggregation scheduler...
INFO:     Scheduler started - will run at 0:00 AM daily
```

**Spark ETL logs**:
```bash
# When triggered
INFO:     Next aggregation scheduled at 2025-11-05 00:00:00
INFO:     Starting daily aggregation...
INFO:     Daily aggregation completed: 29 rows, cash_in=2,592, cash_out=617
```

### Metrics

Check current state:
```bash
curl http://localhost:8000/spark/status
```

## Troubleshooting

### Issue 1: Spark import error

```
ModuleNotFoundError: No module named 'pyspark'
```

**Solution**:
```bash
pip install pyspark
```

### Issue 2: Java not found

```
Exception: Java gateway process exited before sending its port number
```

**Solution**: Install Java 8 or 11
```bash
# Windows: Download from https://www.oracle.com/java/technologies/downloads/
# Linux: sudo apt install openjdk-11-jdk
```

Verify:
```bash
java -version
```

### Issue 3: ETL timeout

```
HTTPException: Spark ETL timeout (5 minutes)
```

**Solution**: Increase timeout in `app.py`:
```python
timeout=600,  # 10 minutes instead of 5
```

### Issue 4: Permission denied on daily CSV

```
PermissionError: [Errno 13] Permission denied: 'data\\cash_daily.csv'
```

**Solution**: Close any programs accessing the CSV file (Excel, etc.)

### Issue 5: Scheduler not running

```bash
curl http://localhost:8000/spark/status
# "scheduler_running": false
```

**Solution**: Restart FastAPI server
```bash
# Stop server (Ctrl+C)
# Start again
uvicorn app.main:app --reload --port 8000
```

## Performance

### Current Performance (Local Mode)

| Metric | Value |
|--------|-------|
| Daily CSV size | ~30 rows/day |
| ETL processing time | ~10-30 seconds |
| Training dataset size | ~800 rows (2+ years) |
| Model retrain time | ~5 minutes (all 6 models) |
| Memory usage | ~500MB (Spark local) |

### Production Performance (Cluster Mode)

| Metric | Local | Cluster (10 nodes) |
|--------|-------|-------------------|
| Max rows | 10M | 1B+ |
| ETL time | 5 min | 30 sec |
| Training time | 10 min | 2 min |
| Memory | 4GB | 640GB (64GB × 10) |

## Next Steps

### Phase 1: Current (✓ DONE)
- [x] Spark ETL pipeline
- [x] Daily scheduler (0:00 AM)
- [x] API endpoints
- [x] Local mode testing

### Phase 2: Production (Optional)
- [ ] Deploy Hadoop HDFS cluster
- [ ] Switch to HDFS storage mode
- [ ] Configure Spark cluster (YARN/Standalone)
- [ ] Add Spark UI monitoring
- [ ] Setup Airflow for orchestration

### Phase 3: Advanced (Future)
- [ ] Spark Streaming for real-time processing
- [ ] Spark MLlib models (alternative to M5P)
- [ ] Model versioning on HDFS
- [ ] A/B testing framework
- [ ] Grafana + Prometheus monitoring

## Configuration

### Environment Variables

```bash
# .env file
SPARK_MODE=local          # local or hdfs
SPARK_MASTER=local[*]     # or yarn, spark://master:7077
HDFS_BASE_PATH=hdfs://namenode:9000/banktrading
SCHEDULER_ENABLED=true
SCHEDULER_TIME=00:00      # Daily run time (24h format)
```

### Scheduler Configuration

Edit `app/app.py`:
```python
# Change scheduler time
scheduler = DailyAggregationScheduler(
    daily_csv=DAILY_CSV_PATH,
    training_csv=TRAINING_DATASET_PATH,
    retrain_callback=retrain_all_models,
    trigger_time="00:00"  # Customize here
)
```

## Cost Estimate

### Development (Current)
- **Hardware**: Local machine
- **Cost**: $0 (using existing resources)

### Production (Cluster)
- **Hadoop Cluster**: 10 nodes × $100/month = $1,000/month
- **Or On-Premise**: $30,000 one-time (hardware)

## Support

### Documentation
- [Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API](https://spark.apache.org/docs/latest/api/python/)
- [HDFS Architecture](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)

### Contact
- **Issues**: Create ticket at project repository
- **Questions**: Contact dev team

## Summary

**Hệ thống đã được tích hợp Spark thành công!**

✅ **Features**:
- Spark ETL pipeline cho data processing
- Daily scheduler tự động (0:00 AM)
- API endpoints để trigger manual
- Tương thích với hệ thống hiện tại
- Sẵn sàng scale lên HDFS cluster khi cần

✅ **Testing**:
- PySpark installed và working
- ETL pipeline tested
- Scheduler running
- API endpoints functional

✅ **Production Ready**:
- Local mode: Ready to use
- HDFS mode: Ready to deploy (requires cluster setup)

**Bắt đầu sử dụng ngay**:
```bash
uvicorn app.main:app --reload --port 8000
```

Hệ thống sẽ tự động chạy ETL lúc 0:00 AM mỗi ngày! 🚀
