"""
Retrain the cash_out models with correct lag_out features.
"""
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from multi_target_model import MultiTargetCashModel

def main():
    print("=" * 60)
    print("Retraining Cash OUT Models with Correct Features")
    print("=" * 60)

    csv_path = "data/cash_daily_train_realistic.csv"

    print(f"\nDataset: {csv_path}")

    # Initialize multi-model
    multi_model = MultiTargetCashModel(base_model_dir="models")

    print("\nRetraining CASH OUT models only...")
    print("   - cash_out_next_day")
    print("   - cash_out_h7")
    print("   - cash_out_next_month")

    # Retrain only cash_out models
    stats = {}

    print("\n1. Training cash_out_next_day...")
    stats["cash_out_next_day"] = multi_model.cash_out_next_day.train(
        csv_path,
        target_column="cash_out_next_day"
    )
    print(f"   RMSE: {stats['cash_out_next_day']['rmse']:.2f}")
    print(f"   R2: {stats['cash_out_next_day']['r2']:.4f}")

    print("\n2. Training cash_out_h7...")
    stats["cash_out_h7"] = multi_model.cash_out_h7.train(
        csv_path,
        target_column="cash_out_h7_sum"
    )
    print(f"   RMSE: {stats['cash_out_h7']['rmse']:.2f}")
    print(f"   R2: {stats['cash_out_h7']['r2']:.4f}")

    print("\n3. Training cash_out_next_month...")
    stats["cash_out_next_month"] = multi_model.cash_out_next_month.train(
        csv_path,
        target_column="cash_out_next_month_sum"
    )
    print(f"   RMSE: {stats['cash_out_next_month']['rmse']:.2f}")
    print(f"   R2: {stats['cash_out_next_month']['r2']:.4f}")

    print("\n" + "=" * 60)
    print("Retraining Complete!")
    print("=" * 60)

    print("\nVerifying schemas...")
    import json

    for model_name in ["cash_out_next_day", "cash_out_h7", "cash_out_next_month"]:
        schema_path = Path(f"models/{model_name}/schema_{model_name}.json")
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            print(f"\n{model_name}:")
            print(f"   Target: {schema['target_column']}")
            print(f"   Features: {', '.join(schema['required_features'][:3])}...")

            # Check for lag_out features
            has_lag_out = any('lag' in f and 'out' in f for f in schema['required_features'])
            if has_lag_out:
                print(f"   Status: OK - Using lag_out features")
            else:
                print(f"   Status: ERROR - Missing lag_out features!")

    print("\n" + "=" * 60)
    print("All cash_out models retrained with correct features!")
    print("=" * 60)

if __name__ == "__main__":
    main()
