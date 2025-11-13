# How the Cash Flow Prediction System Works

## Summary

**GOOD NEWS**: The models ARE reading cash_in and cash_out data correctly! The system is working as designed.

## Test Results

```
OK - CSV loaded successfully
   Rows: 365
   Columns: ['date', 'cash_in', 'cash_out', 'channel', 'balance', 'day_of_week', 'month', 'quarter']

OK - Multi-target model initialized
   Models created:
   - cash_in_next_day ✓
   - cash_out_next_day ✓
   - cash_in_h7 ✓
   - cash_out_h7 ✓
   - cash_in_next_month ✓
   - cash_out_next_month ✓

OK - Bootstrap completed (all 6 models loaded successfully)
```

## How Same-Day Transaction Aggregation Works

### Code Location: `app.py` lines 245-351

When you create a transaction, the system automatically:

1. **Extracts transaction info**:
   - Date from `event_date` or `event_ts`
   - Direction (cash_in or cash_out)
   - Amount converted to VND

2. **Aggregates by date**:
   ```python
   # If date already exists in CSV, ADD to existing values
   if (df["date"] == event_date_str).any():
       idx = df.index[df["date"] == event_date_str][0]
       current_cash_in = df.at[idx, "cash_in"]
       current_cash_out = df.at[idx, "cash_out"]

       # ADD new transaction amount
       if direction == "cash_out":
           current_cash_out += amount_vnd
       else:
           current_cash_in += amount_vnd
   ```

3. **Saves to CSV**:
   ```python
   df.to_csv("data/cash_daily_train_realistic.csv", index=False)
   ```

4. **Automatically retrains all 6 models** (lines 333-349):
   ```python
   multi_stats = multi_model.retrain_all(dataset_csv=str(dataset_path))
   ```

### Example Flow

**Scenario**: Create 3 transactions on 2024-01-15

```
Transaction 1:
- Date: 2024-01-15
- Type: cash_in
- Amount: 5,000,000 VND
→ CSV row: date=2024-01-15, cash_in=5,000,000, cash_out=0

Transaction 2:
- Date: 2024-01-15
- Type: cash_in
- Amount: 3,000,000 VND
→ CSV row: date=2024-01-15, cash_in=8,000,000, cash_out=0  (AGGREGATED!)

Transaction 3:
- Date: 2024-01-15
- Type: cash_out
- Amount: 2,000,000 VND
→ CSV row: date=2024-01-15, cash_in=8,000,000, cash_out=2,000,000  (AGGREGATED!)
```

**Result**: ONE row per date in CSV with summed values.

## Why Predictions Require More Features

### The CSV has these columns:
```csv
date,cash_in,cash_out,channel,balance,day_of_week,month,quarter
```

### But predictions need these engineered features:
```json
{
  "cash_in_d0": 5000000,      // Today's cash in
  "cash_out_d0": 3000000,     // Today's cash out
  "cash_net_d0": 2000000,     // Net = cash_in - cash_out
  "lag1_in": 4500000,         // Cash in from 1 day ago
  "lag7_in": 6000000,         // Cash in from 7 days ago
  "roll_mean_7_in": 5500000,  // Average cash in last 7 days
  "dow": 1,                   // Day of week (0=Monday, 6=Sunday)
  "is_weekend": 0,            // 1 if weekend, 0 otherwise
  "is_month_end": 0,          // 1 if day >= 25
  "is_payday": 0,             // 1 if day 15 or >= 25
  "channel": "ATM"            // Transaction channel
}
```

### Why?

The models were **trained** with these engineered features because they improve predictions:
- `lag1_in`, `lag7_in` = Historical patterns
- `roll_mean_7_in` = Trend over last week
- `is_weekend`, `is_payday` = Seasonal patterns

## How to Prepare Features for Prediction

### Option 1: Manual Calculation (For GUI)

You need to calculate features from your recent transaction history:

```python
import pandas as pd
from datetime import datetime

# Load the CSV
df = pd.read_csv("data/cash_daily_train_realistic.csv")

# Get last 7 days
df_recent = df.tail(7)

# Calculate features
features = {
    "cash_in_d0": float(df.iloc[-1]["cash_in"]),
    "cash_out_d0": float(df.iloc[-1]["cash_out"]),
    "cash_net_d0": float(df.iloc[-1]["cash_in"] - df.iloc[-1]["cash_out"]),
    "lag1_in": float(df.iloc[-2]["cash_in"]) if len(df) > 1 else 0.0,
    "lag7_in": float(df.iloc[-8]["cash_in"]) if len(df) > 7 else 0.0,
    "roll_mean_7_in": float(df_recent["cash_in"].mean()),
    "dow": int(df.iloc[-1]["day_of_week"]),
    "is_weekend": int(df.iloc[-1]["day_of_week"] >= 5),
    "is_month_end": int(datetime.now().day >= 25),
    "is_payday": int(datetime.now().day == 15 or datetime.now().day >= 25),
    "channel": "ATM"  # or "ONLINE", "POS", "TRANSFER"
}
```

### Option 2: Create a Helper Endpoint (Recommended)

We should add an endpoint to auto-calculate features from the CSV:

```python
@app.get("/ml/prepare-features")
def prepare_features_from_latest():
    """
    Automatically prepare prediction features from the latest data in CSV.
    """
    dataset_path = DATA_DIR / "cash_daily_train_realistic.csv"
    df = pd.read_csv(dataset_path)

    if len(df) < 7:
        raise HTTPException(400, "Need at least 7 days of data")

    # Calculate all features
    last_row = df.iloc[-1]
    features = {
        "cash_in_d0": float(last_row["cash_in"]),
        "cash_out_d0": float(last_row["cash_out"]),
        "cash_net_d0": float(last_row["cash_in"] - last_row["cash_out"]),
        "lag1_in": float(df.iloc[-2]["cash_in"]),
        "lag7_in": float(df.iloc[-8]["cash_in"]),
        "roll_mean_7_in": float(df.tail(7)["cash_in"].mean()),
        "dow": int(last_row["day_of_week"]),
        "is_weekend": int(last_row["day_of_week"] >= 5),
        "is_month_end": 1 if int(last_row["date"].split("-")[2]) >= 25 else 0,
        "is_payday": 1 if int(last_row["date"].split("-")[2]) in [15, 25, 26, 27, 28, 29, 30, 31] else 0,
        "channel": str(last_row["channel"])
    }

    return features
```

Then users can simply:
```javascript
// 1. Get features automatically
fetch("/ml/prepare-features")
  .then(r => r.json())
  .then(features => {
    // 2. Use features for prediction
    fetch("/ml/predict/cash-in", {
      method: "POST",
      body: JSON.stringify({ features })
    })
  })
```

## Complete Workflow

```
User creates transaction
  ↓
POST /transactions (app.py line 583)
  ↓
_update_training_dataset_from_transaction() (line 245)
  ↓
1. Load CSV
2. Find row for date (or create new row)
3. ADD transaction amount to cash_in or cash_out
4. Save CSV
5. Retrain all 6 models
  ↓
Models updated and ready for prediction!
  ↓
User requests prediction
  ↓
OPTION A: Manual - User provides all engineered features
OPTION B: Auto - System calculates features from CSV
  ↓
POST /ml/predict/cash-in or /ml/predict/cash-out
  ↓
Return: { next_day, h7_sum, next_month_sum }
```

## File Structure

```
ml_service/
├── data/
│   └── cash_daily_train_realistic.csv   ← Daily aggregated data (1 row per date)
├── models/
│   ├── cash_in_next_day/                ← Model artifacts
│   ├── cash_out_next_day/
│   ├── cash_in_h7/
│   ├── cash_out_h7/
│   ├── cash_in_next_month/
│   └── cash_out_next_month/
└── app/
    ├── app.py                           ← Transaction handling + aggregation
    ├── multi_target_model.py            ← 6-model wrapper
    └── ml_m5p.py                        ← M5P training logic
```

## Summary

1. **Data Reading**: ✓ WORKING - Models successfully read cash_in and cash_out from CSV
2. **Transaction Aggregation**: ✓ WORKING - Multiple same-day transactions sum into one CSV row
3. **Auto Retrain**: ✓ WORKING - Models retrain after each transaction
4. **Prediction Issue**: Need to provide engineered features (lag1_in, lag7_in, etc.)

## Solution: Add Feature Preparation Endpoint

We need to add `/ml/prepare-features` endpoint so users don't have to manually calculate:
- lag features
- rolling means
- date-based flags

This will make the GUI work smoothly:
```
User clicks "Dự đoán Cash In"
  ↓
GUI calls /ml/prepare-features
  ↓
GUI calls /ml/predict/cash-in with auto-generated features
  ↓
Display: Next Day, Next 7 Days, Next Month
```

## Next Steps

1. Add `/ml/prepare-features` endpoint to `app.py`
2. Update GUI to use this endpoint before prediction
3. Test complete flow: create transaction → predict
