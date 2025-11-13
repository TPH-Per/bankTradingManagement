# Fix: Cash Out 7-Day Prediction Returning 0

## Problem

User reported: "cash out tổng 7 ngày ra 0, kiểm tra lại"

Cash OUT 7-day prediction was returning 0 VND.

## Root Cause

The cash_out models (cash_out_next_day, cash_out_h7, cash_out_next_month) were trained with **incorrect lag features**.

### What Was Wrong

**Before Fix**:
```
cash_out models were using:
- lag1_in       (cash IN from 1 day ago)  ❌ WRONG
- lag7_in       (cash IN from 7 days ago) ❌ WRONG
- roll_mean_7_in (avg cash IN last 7 days) ❌ WRONG
```

**Should Have Been**:
```
cash_out models should use:
- lag1_out       (cash OUT from 1 day ago)  ✓ CORRECT
- lag7_out       (cash OUT from 7 days ago) ✓ CORRECT
- roll_mean_7_out (avg cash OUT last 7 days) ✓ CORRECT
```

### Why This Caused 0 Predictions

The `cash_out_h7` model learned relationships between **cash_in lag features** and future cash_out values. Since these features are not strongly correlated, the model couldn't make good predictions and returned very low values (close to 0).

## Code Issue

**File**: `ml_m5p.py` line 1205-1212

**Before Fix**:
```python
required_features = list(self.required_features_ or REQUIRED_FEATURES_IN)
# Always used REQUIRED_FEATURES_IN (cash_in lag features)
```

**After Fix**:
```python
# Select appropriate feature set based on target column
if "out" in target_column.lower() or "cash_out" in target_column.lower():
    # Cash out models use lag_out features
    required_features = list(REQUIRED_FEATURES_OUT)
else:
    # Cash in models use lag_in features
    required_features = list(REQUIRED_FEATURES_IN)
```

## Solution Applied

1. **Updated ml_m5p.py** (lines 1205-1212, 1259, 1275):
   - Added logic to select correct feature set based on target column name
   - Cash OUT targets now use `REQUIRED_FEATURES_OUT`
   - Cash IN targets use `REQUIRED_FEATURES_IN`

2. **Retrained all cash_out models**:
   - cash_out_next_day
   - cash_out_h7
   - cash_out_next_month

## Results

### Before Fix
```
CASH OUT Predictions:
   Next Day:      1,494,700 VND
   Next 7 Days:           0 VND  ❌ WRONG
   Next Month:   97,435,229 VND
```

### After Fix
```
CASH OUT Predictions:
   Next Day:      3,127,821 VND
   Next 7 Days:  22,994,650 VND  ✓ CORRECT
   Next Month:  101,951,859 VND
```

### Training Metrics

**cash_out_next_day**:
- RMSE: 609,531.78
- R²: 0.4991 (49.91% variance explained)

**cash_out_h7**:
- RMSE: 2,676,562.23
- R²: 0.1809 (18.09% variance explained)
- Tree depth: 0 (single regression model at root)

**cash_out_next_month**:
- RMSE: 5,953,925.59
- R²: 0.3074 (30.74% variance explained)

## Verification

Verified correct features in schemas:

```bash
cd models/cash_out_h7
cat schema_cash_out_h7.json
```

**After Fix**:
```json
{
  "required_features": [
    "cash_in_d0",
    "cash_out_d0",
    "cash_net_d0",
    "lag1_out",     ✓
    "lag7_out",     ✓
    "roll_mean_7_out", ✓
    "dow",
    "is_weekend",
    "is_month_end",
    "is_payday",
    "channel"
  ]
}
```

## Files Modified

1. **ml_m5p.py**:
   - Line 1205-1212: Feature set selection logic
   - Line 1259: Save correct required_features
   - Line 1275: Save correct features in schema

2. **Models Retrained**:
   - `models/cash_out_next_day/m5p_cash_out_next_day.pkl`
   - `models/cash_out_h7/m5p_cash_out_h7.pkl`
   - `models/cash_out_next_month/m5p_cash_out_next_month.pkl`

3. **Schemas Updated**:
   - `models/cash_out_next_day/schema_cash_out_next_day.json`
   - `models/cash_out_h7/schema_cash_out_h7.json`
   - `models/cash_out_next_month/schema_cash_out_next_month.json`

## How to Verify

Run the test script:
```bash
cd C:\temp\aa\ml_service
python test_feature_preparation.py
```

Expected output:
```
CASH OUT Predictions:
   Next Day:      ~3,000,000 VND
   Next 7 Days:  ~23,000,000 VND  ✓ NOT ZERO
   Next Month:  ~100,000,000 VND
```

## Note on cash_out_h7 Performance

The `cash_out_h7` model has:
- R² = 0.1809 (relatively low)
- Tree depth = 0 (single linear regression)

This suggests that:
1. Cash OUT is harder to predict than cash IN
2. The 7-day sum aggregation smooths out patterns
3. More features or more training data might help

However, the predictions are now **non-zero and reasonable** based on historical patterns.

## Prevention

To prevent this in the future:
1. Always verify model schemas after training
2. Check that lag features match the target type (cash_in → lag_in, cash_out → lag_out)
3. Test predictions immediately after training

## Status

✅ **FIXED** - Cash OUT 7-day predictions now working correctly
