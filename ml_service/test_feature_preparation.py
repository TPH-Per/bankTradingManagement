"""
Test the feature preparation endpoint and complete prediction flow.
"""
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from multi_target_model import MultiTargetCashModel
import pandas as pd

def main():
    print("=" * 60)
    print("Testing Feature Preparation and Prediction Flow")
    print("=" * 60)

    # 1. Load CSV
    csv_path = Path("data/cash_daily_train_realistic.csv")
    df = pd.read_csv(csv_path)
    print(f"\nStep 1: Loaded CSV with {len(df)} rows")

    # 2. Prepare features (simulating the endpoint)
    last_row = df.iloc[-1]

    from datetime import datetime
    try:
        date_obj = datetime.fromisoformat(str(last_row["date"]))
        day_of_month = date_obj.day
    except Exception:
        day_of_month = 15

    features = {
        # Today's values
        "cash_in_d0": float(last_row["cash_in"]),
        "cash_out_d0": float(last_row["cash_out"]),
        "cash_net_d0": float(last_row["cash_in"] - last_row["cash_out"]),

        # Lag features for cash_in
        "lag1_in": float(df.iloc[-2]["cash_in"]) if len(df) > 1 else 0.0,
        "lag7_in": float(df.iloc[-8]["cash_in"]) if len(df) > 7 else 0.0,
        "roll_mean_7_in": float(df.tail(7)["cash_in"].mean()),

        # Lag features for cash_out
        "lag1_out": float(df.iloc[-2]["cash_out"]) if len(df) > 1 else 0.0,
        "lag7_out": float(df.iloc[-8]["cash_out"]) if len(df) > 7 else 0.0,
        "roll_mean_7_out": float(df.tail(7)["cash_out"].mean()),

        # Date features
        "dow": int(last_row["day_of_week"]),
        "is_weekend": int(last_row["day_of_week"] >= 5),
        "is_month_end": 1 if day_of_month >= 25 else 0,
        "is_payday": 1 if day_of_month == 15 or day_of_month >= 25 else 0,

        # Channel
        "channel": str(last_row.get("channel", "DEFAULT"))
    }

    print(f"\nStep 2: Prepared features from date {last_row['date']}")
    print("\nFeatures:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"   {key:20s} = {value:>15,.0f}")
        else:
            print(f"   {key:20s} = {value}")

    # 3. Initialize models
    print(f"\nStep 3: Initializing models...")
    multi_model = MultiTargetCashModel(base_model_dir="models")

    # 4. Bootstrap (load existing models)
    print(f"\nStep 4: Loading models...")
    stats = multi_model.bootstrap(dataset_csv=csv_path, force_retrain=False)
    print("   All models loaded successfully")

    # 5. Make predictions
    print(f"\nStep 5: Making predictions...")
    print("\n" + "=" * 60)
    print("CASH IN Predictions")
    print("=" * 60)

    try:
        cash_in_preds = multi_model.predict_cash_in(features)
        print(f"   Next Day:     {cash_in_preds['next_day']:>15,.0f} VND")
        print(f"   Next 7 Days:  {cash_in_preds['h7_sum']:>15,.0f} VND")
        print(f"   Next Month:   {cash_in_preds['next_month_sum']:>15,.0f} VND")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("CASH OUT Predictions")
    print("=" * 60)

    try:
        cash_out_preds = multi_model.predict_cash_out(features)
        print(f"   Next Day:     {cash_out_preds['next_day']:>15,.0f} VND")
        print(f"   Next 7 Days:  {cash_out_preds['h7_sum']:>15,.0f} VND")
        print(f"   Next Month:   {cash_out_preds['next_month_sum']:>15,.0f} VND")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUCCESS - Complete Flow Working!")
    print("=" * 60)
    print("1. CSV data readable: OK")
    print("2. Feature preparation: OK")
    print("3. Model loading: OK")
    print("4. Cash IN predictions: OK")
    print("5. Cash OUT predictions: OK")
    print("\nThe endpoint /ml/prepare-features will provide these features automatically.")

if __name__ == "__main__":
    main()
