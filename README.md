# bankTrading ML Service v2.3

A machine learning-powered cash flow prediction service for banking applications, built with FastAPI, scikit-learn, and pandas.

> **Repository**: https://github.com/TPH-Per/bankTradingManagement

## Features

- **Real-time Transaction Processing**: Handle banking transactions with automatic currency conversion
- **Cash Flow Prediction**: Predict next-day, 7-day, and monthly cash inflows and outflows using M5P model trees
- **Multi-target Models**: Six specialized models for different prediction horizons:
  - `cash_in_next_day` - Next day cash inflow prediction
  - `cash_out_next_day` - Next day cash outflow prediction
  - `cash_in_h7` - Next 7 days cash inflow sum
  - `cash_out_h7` - Next 7 days cash outflow sum
  - `cash_in_next_month` - Next calendar month cash inflow sum
  - `cash_out_next_month` - Next calendar month cash outflow sum
- **Automatic Model Retraining**: Models automatically retrain with new transaction data
- **Daily Aggregation Scheduler**: Automated daily data aggregation at midnight
- **Cassandra Integration**: Full CRUD operations with optional Cassandra database support
- **RESTful API**: Comprehensive API endpoints for transactions, predictions, and model management
- **Web-based GUI**: User-friendly interface for transaction management, model training, and predictions

## Project Structure

```
ml_service/
├── app/
│   ├── __init__.py
│   ├── app.py          # Main FastAPI application
│   ├── cassandra_service.py  # Cassandra database integration
│   ├── dual_m5p.py     # Dual cash flow model wrapper
│   ├── main.py         # Application entry point
│   ├── ml_m5p.py       # Core M5P model implementation
│   ├── multi_target_model.py  # Multi-target model management
│   ├── scheduler.py     # Daily aggregation scheduler
│   └── requirements.txt
├── data/
│   ├── cash_daily.csv           # Temporary daily transaction data
│   └── cash_daily_train_realistic.csv  # Training dataset
├── models/
│   ├── cash_in/        # Cash inflow models
│   ├── cash_out/       # Cash outflow models
│   └── m5p_schema.json # Model schema
├── Dockerfile
└── docker-compose.ml-only.yml
```

## API Endpoints

### Health Checks
- `GET /health` - Service health status
- `GET /healthz/liveness` - Liveness probe
- `GET /healthz/readiness` - Readiness probe

### Real-time Transactions
- `POST /rt/transactions` - Create a new transaction
- `POST /rt/transactions/bulk` - Create multiple transactions
- `GET /rt/transactions` - List transactions for an account and date
- `GET /rt/transactions/all` - List all recent transactions
- `GET /rt/transactions/range` - List transactions in a date range
- `GET /rt/transactions/by-id/{tx_id}` - Get transaction by ID

### Machine Learning
- `POST /ml/m5p/train` - Train the M5P model
- `POST /ml/m5p/train/upload` - Upload CSV and train model
- `GET /ml/m5p/status` - Get model status
- `GET /ml/m5p/schema` - Get model schema
- `GET /ml/m5p/feature-mapping` - Get feature mapping
- `GET /ml/prepare-features` - Prepare features for prediction
- `POST /ml/m5p/predict` - Make a single prediction
- `POST /ml/m5p/predict/batch` - Make batch predictions
- `POST /ml/predict/all` - Predict all 6 targets at once
- `POST /ml/predict/cash-in` - Predict all cash-in targets
- `POST /ml/predict/cash-out` - Predict all cash-out targets
- `GET /ml/status` - Get multi-model status
- `GET /ml/m5p/rules` - Export decision rules
- `GET /ml/m5p/feature-importances` - Get feature importances
- `POST /ml/m5p/load` - Load model from disk

### KPI Management
- `POST /kpi/daily/upsert` - Create or update daily KPI
- `GET /kpi/daily/get` - Get daily KPI
- `GET /kpi/daily/list` - List KPIs in date range

### Data Utilities
- `POST /data/aggregate/daily` - Manually trigger daily aggregation
- `GET /data/preview` - Preview data file

## Getting Started

### Prerequisites
- Python 3.9+ (tested with 3.11.7)
- pip package manager

### Installation

1. Navigate to the app directory:
   ```bash
   cd ml_service/app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Service

#### Direct Execution (Windows)
```bash
cd ml_service/app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Docker (Optional)
```bash
docker-compose -f docker-compose.ml-only.yml up
```

### Accessing the API
- **API Documentation**: http://localhost:8000/docs
- **Base URL**: http://localhost:8000

## Environment Variables

- `MODEL_DIR` - Path to models directory (default: ./models)
- `DATA_DIR` - Path to data directory (default: ./data)
- `M5P_DATASET` - Training dataset path (default: ./data/cash_daily_train_realistic.csv)
- `M5P_FORCE_RETRAIN` - Force model retraining on startup (default: 0)
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `RELOAD` - Enable auto-reload (default: false)

## Data Format

### Transaction Data
The service expects transaction data in CSV format with the following columns:
- `date` - Transaction date (ISO format)
- `cash_in` - Cash inflow amount
- `cash_out` - Cash outflow amount
- `channel` - Transaction channel
- `day_of_week` - Day of week (0-6)
- `month` - Month (1-12)
- `quarter` - Quarter (1-4)
- `balance` - Account balance

### Feature Engineering
The service automatically engineers features including:
- Lag features (1-day and 7-day)
- Rolling means
- Date-based features (weekend, month end, payday)
- Channel information

## Model Architecture

The service uses M5P model trees, which combine:
1. **Decision Tree Structure**: For splitting data based on feature values
2. **Linear Regression Models**: At leaf nodes for precise predictions
3. **Smoothing**: To prevent overfitting and improve generalization
4. **Pruning**: To optimize tree complexity

## Cassandra Integration

The service supports optional Cassandra database integration for:
- Transaction storage and deduplication
- API call logging and auditing
- KPI storage and retrieval

To enable Cassandra:
1. Set up a Cassandra cluster
2. Configure connection details in `cassandra_service.py`
3. The service will automatically use Cassandra when available

## GUI Interface

A web-based GUI is available in `gui.html` and `index-bootstrap.html` that provides:
- Transaction creation and management
- Model training and prediction
- KPI management
- Real-time status monitoring

To use the GUI:
1. Serve the file through a web server
2. Ensure the ML service is running on http://localhost:8000
3. Access the GUI through your web browser

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Version History

- **v2.3** - Enhanced multi-target models and improved GUI
- **v2.2** - Added Cassandra integration and daily scheduler
- **v2.1** - Initial M5P model implementation