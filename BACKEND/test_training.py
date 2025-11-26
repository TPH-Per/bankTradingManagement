"""
Test script to verify models can read and train on the CSV data.
"""
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from multi_target_model import MultiTargetCashModel
import pandas as pd

def main():
    # 1. Verify CSV can be read
    print("=" * 60)
    print("Step 1: Verifying CSV data can be read...")
    print("=" * 60)

    csv_path = Path("data/cash_daily_train_realistic.csv")
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"OK - CSV loaded successfully")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)

    # 2. Test multi-target model initialization
    print("\n" + "=" * 60)
    print("Step 2: Initializing multi-target model system...")
    print("=" * 60)

    try:
        multi_model = MultiTargetCashModel(base_model_dir="models")
        print("OK - Multi-target model initialized")
        print(f"   Models created:")
        print(f"   - cash_in_next_day")
        print(f"   - cash_out_next_day")
        print(f"   - cash_in_h7")
        print(f"   - cash_out_h7")
        print(f"   - cash_in_next_month")
        print(f"   - cash_out_next_month")
    except Exception as e:
        print(f"ERROR: Failed to initialize models: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test bootstrap (train or load models)
    print("\n" + "=" * 60)
    print("Step 3: Bootstrap models (train or load)...")
    print("=" * 60)

    try:
        stats = multi_model.bootstrap(
            dataset_csv=csv_path,
            force_retrain=False  # Only train if models don't exist
        )
        print("OK - Bootstrap completed")
        print("\nModel statistics:")
        for model_name, model_stats in stats.items():
            print(f"\n{model_name}:")
            for key, value in model_stats.items():
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"ERROR: Bootstrap failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Test predictions
    print("\n" + "=" * 60)
    print("Step 4: Testing predictions...")
    print("=" * 60)

    # Sample features from the last row of data
    last_row = df.iloc[-1]
    test_features = {
        "cash_in": float(last_row["cash_in"]),
        "cash_out": float(last_row["cash_out"]),
        "day_of_week": int(last_row["day_of_week"]),
        "month": int(last_row["month"]),
        "quarter": int(last_row["quarter"]),
    }

    print(f"\nTest features (from last row):")
    for key, value in test_features.items():
        print(f"   {key}: {value}")

    try:
        # Test cash_in predictions
        print("\n--- CASH IN Predictions ---")
        cash_in_preds = multi_model.predict_cash_in(test_features)
        print(f"   Next Day:   {cash_in_preds['next_day']:,.0f} VND")
        print(f"   Next 7 Days: {cash_in_preds['h7_sum']:,.0f} VND")
        print(f"   Next Month:  {cash_in_preds['next_month_sum']:,.0f} VND")

        # Test cash_out predictions
        print("\n--- CASH OUT Predictions ---")
        cash_out_preds = multi_model.predict_cash_out(test_features)
        print(f"   Next Day:   {cash_out_preds['next_day']:,.0f} VND")
        print(f"   Next 7 Days: {cash_out_preds['h7_sum']:,.0f} VND")
        print(f"   Next Month:  {cash_out_preds['next_month_sum']:,.0f} VND")

        print("\nOK - All predictions successful!")

    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("OK - All tests passed!")
    print("   CSV data: readable")
    print("   Models: initialized and trained")
    print("   Predictions: working")
    print("\nThe system is ready to use.")

if __name__ == "__main__":
    main()
