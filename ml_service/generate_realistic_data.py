#!/usr/bin/env python3
"""
Generate realistic cash flow training data for ML models.
Creates 365 days of data with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2023, 1, 1)
NUM_DAYS = 365
CHANNELS = ["ATM", "ONLINE", "POS", "TRANSFER"]

def generate_daily_cash_flow(date, day_idx):
    """Generate realistic cash in/out for a given date."""

    # Base amounts
    base_cash_in = 5_000_000  # 5M VND base
    base_cash_out = 3_000_000  # 3M VND base

    # Day of week effect (0=Monday, 6=Sunday)
    dow = date.weekday()
    is_weekend = dow >= 5

    # Weekend: lower activity
    if is_weekend:
        cash_in_multiplier = 0.6
        cash_out_multiplier = 0.7
    else:
        # Weekday: normal to high
        cash_in_multiplier = 1.0 + (dow * 0.05)  # Gradually increase through week
        cash_out_multiplier = 0.9 + (dow * 0.05)

    # Month effects
    day_of_month = date.day
    is_month_start = day_of_month <= 5
    is_month_end = day_of_month >= 25
    is_payday = day_of_month in [15, 25, 26, 27, 28, 29, 30, 31]

    # Payday: higher cash in
    if is_payday:
        cash_in_multiplier *= 1.5
        cash_out_multiplier *= 1.2

    # Month start/end: higher activity
    if is_month_start or is_month_end:
        cash_in_multiplier *= 1.3
        cash_out_multiplier *= 1.1

    # Seasonal trend (growth over year)
    seasonal_factor = 1.0 + (day_idx / NUM_DAYS) * 0.3  # 30% growth over year

    # Random variation (±20%)
    random_in = np.random.uniform(0.8, 1.2)
    random_out = np.random.uniform(0.8, 1.2)

    # Calculate final amounts
    cash_in = base_cash_in * cash_in_multiplier * seasonal_factor * random_in
    cash_out = base_cash_out * cash_out_multiplier * seasonal_factor * random_out

    # Round to nearest 1000
    cash_in = round(cash_in / 1000) * 1000
    cash_out = round(cash_out / 1000) * 1000

    # Ensure positive
    cash_in = max(100_000, cash_in)
    cash_out = max(50_000, cash_out)

    # Random channel
    channel = np.random.choice(CHANNELS, p=[0.4, 0.3, 0.2, 0.1])

    return {
        "date": date.strftime("%Y-%m-%d"),
        "cash_in": cash_in,
        "cash_out": cash_out,
        "channel": channel
    }

# Generate data
print(f"Generating {NUM_DAYS} days of cash flow data...")
data = []
balance = 0.0

for day_idx in range(NUM_DAYS):
    date = START_DATE + timedelta(days=day_idx)
    row = generate_daily_cash_flow(date, day_idx)

    # Calculate running balance
    balance += row["cash_in"] - row["cash_out"]
    row["balance"] = balance

    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Add computed features (these will be re-computed during training, but good to have)
df["day_of_week"] = pd.to_datetime(df["date"]).dt.weekday
df["month"] = pd.to_datetime(df["date"]).dt.month
df["quarter"] = pd.to_datetime(df["date"]).dt.quarter

# Reorder columns
df = df[["date", "cash_in", "cash_out", "channel", "balance", "day_of_week", "month", "quarter"]]

# Save to CSV
output_path = "data/cash_daily_train_realistic.csv"
df.to_csv(output_path, index=False)

print(f"\nOK Generated {len(df)} rows")
print(f"Saved to: {output_path}")
print(f"\nStatistics:")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Cash In:  Min={df['cash_in'].min():,.0f}, Max={df['cash_in'].max():,.0f}, Avg={df['cash_in'].mean():,.0f}")
print(f"   Cash Out: Min={df['cash_out'].min():,.0f}, Max={df['cash_out'].max():,.0f}, Avg={df['cash_out'].mean():,.0f}")
print(f"   Channels: {df['channel'].value_counts().to_dict()}")
print(f"\nSample data:")
print(df.head(10).to_string(index=False))
print("\nThis dataset is ready for training all 6 models!")
