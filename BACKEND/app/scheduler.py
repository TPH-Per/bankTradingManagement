"""
Daily scheduler for aggregating cash_daily.csv into training dataset.

This module runs a background task that:
1. At 0:00 AM every day, reads cash_daily.csv
2. Aggregates transactions by date
3. Appends to cash_daily_train_realistic.csv
4. Clears cash_daily.csv
5. Triggers model retraining
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DailyAggregationScheduler:
    """
    Scheduler that aggregates daily cash flow data at midnight.
    """

    def __init__(
        self,
        daily_csv: Path,
        training_csv: Path,
        retrain_callback: Optional[callable] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            daily_csv: Path to cash_daily.csv (temporary daily data)
            training_csv: Path to cash_daily_train_realistic.csv (training dataset)
            retrain_callback: Optional callback function to retrain models
        """
        self.daily_csv = daily_csv
        self.training_csv = training_csv
        self.retrain_callback = retrain_callback
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def aggregate_daily_data(self) -> dict:
        """
        Aggregate data from cash_daily.csv and append to training dataset.

        Returns:
            Dict with aggregation stats
        """
        stats = {
            "date": None,
            "rows_processed": 0,
            "cash_in_total": 0.0,
            "cash_out_total": 0.0,
            "status": "skipped",
            "error": None,
        }

        try:
            # Check if daily CSV exists and has data
            if not self.daily_csv.exists():
                logger.info("Daily CSV does not exist, skipping aggregation")
                return stats

            # Read daily CSV
            df_daily = pd.read_csv(self.daily_csv)

            if df_daily.empty:
                logger.info("Daily CSV is empty, skipping aggregation")
                return stats

            stats["rows_processed"] = len(df_daily)

            # Get yesterday's date (since we run at midnight)
            yesterday = datetime.now(timezone.utc).date()
            yesterday_str = yesterday.isoformat()
            stats["date"] = yesterday_str

            # Expected columns in daily CSV
            required_cols = {"date", "cash_in", "cash_out"}
            if not required_cols.issubset(df_daily.columns):
                logger.warning(
                    f"Daily CSV missing required columns. Has: {df_daily.columns.tolist()}"
                )
                # Try to infer from transaction data if available
                if "amount" in df_daily.columns and "transaction_type" in df_daily.columns:
                    df_daily = self._convert_from_transactions(df_daily)
                else:
                    stats["error"] = "Missing required columns"
                    return stats

            # Aggregate by date
            aggregated = df_daily.groupby("date").agg({
                "cash_in": "sum",
                "cash_out": "sum",
            }).reset_index()

            # Get channel (most common for the day)
            if "channel" in df_daily.columns:
                channel_mode = df_daily["channel"].mode()
                channel = channel_mode[0] if len(channel_mode) > 0 else "DEFAULT"
            else:
                channel = "DEFAULT"

            # Process each aggregated date
            for _, row in aggregated.iterrows():
                date_str = str(row["date"])
                cash_in = float(row["cash_in"])
                cash_out = float(row["cash_out"])

                # Parse date to get features
                try:
                    date_obj = pd.to_datetime(date_str).date()
                except Exception:
                    date_obj = yesterday

                # Create new row with all required columns
                new_row = {
                    "date": date_str,
                    "cash_in": cash_in,
                    "cash_out": cash_out,
                    "channel": channel,
                    "balance": cash_in - cash_out,
                    "day_of_week": date_obj.weekday(),
                    "month": date_obj.month,
                    "quarter": (date_obj.month - 1) // 3 + 1,
                }

                # Append to training CSV
                self._append_to_training(new_row)

                stats["cash_in_total"] += cash_in
                stats["cash_out_total"] += cash_out

            stats["status"] = "success"
            logger.info(
                f"Daily aggregation completed: {stats['rows_processed']} rows, "
                f"cash_in={stats['cash_in_total']:,.0f}, cash_out={stats['cash_out_total']:,.0f}"
            )

            # Clear daily CSV
            self._clear_daily_csv()

            # Trigger model retraining
            if self.retrain_callback:
                try:
                    logger.info("Triggering model retraining...")
                    await self.retrain_callback()
                except Exception as e:
                    logger.exception("Model retraining failed")
                    stats["retrain_error"] = str(e)

        except Exception as e:
            logger.exception("Daily aggregation failed")
            stats["status"] = "error"
            stats["error"] = str(e)

        return stats

    def _convert_from_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert transaction format to cash_in/cash_out format.
        """
        # Initialize columns
        df["cash_in"] = 0.0
        df["cash_out"] = 0.0

        # Ensure date column exists
        if "date" not in df.columns:
            if "event_date" in df.columns:
                df["date"] = df["event_date"]
            elif "event_ts" in df.columns:
                df["date"] = pd.to_datetime(df["event_ts"]).dt.date.astype(str)
            else:
                df["date"] = datetime.now(timezone.utc).date().isoformat()

        # Convert amount based on transaction_type
        for idx, row in df.iterrows():
            amount = abs(float(row.get("amount", 0)))
            tx_type = str(row.get("transaction_type", "")).lower()

            if "in" in tx_type or tx_type == "cash_in":
                df.at[idx, "cash_in"] = amount
            else:
                df.at[idx, "cash_out"] = amount

        return df

    def _append_to_training(self, new_row: dict):
        """
        Append a new row to the training CSV, updating if date exists.
        """
        # Load existing training data
        if self.training_csv.exists():
            df_train = pd.read_csv(self.training_csv)
        else:
            df_train = pd.DataFrame()

        # Check if date already exists
        date_str = new_row["date"]
        if "date" in df_train.columns and (df_train["date"] == date_str).any():
            # Update existing row
            idx = df_train.index[df_train["date"] == date_str][0]
            df_train.at[idx, "cash_in"] += new_row["cash_in"]
            df_train.at[idx, "cash_out"] += new_row["cash_out"]
            df_train.at[idx, "balance"] = (
                df_train.at[idx, "cash_in"] - df_train.at[idx, "cash_out"]
            )
            logger.info(f"Updated existing row for date {date_str}")
        else:
            # Append new row
            df_train = pd.concat(
                [df_train, pd.DataFrame([new_row])],
                ignore_index=True
            )
            logger.info(f"Added new row for date {date_str}")

        # Sort by date and save
        df_train = df_train.sort_values("date")
        df_train.to_csv(self.training_csv, index=False)

    def _clear_daily_csv(self):
        """
        Clear the daily CSV file (keep header only).
        """
        try:
            if self.daily_csv.exists():
                # Preserve existing column order/structure
                existing = pd.read_csv(self.daily_csv, nrows=0)
                columns = list(existing.columns)
            else:
                columns = [
                    "date", "cash_in", "cash_out", "channel", "balance",
                    "day_of_week", "month", "quarter"
                ]

            df_empty = pd.DataFrame(columns=columns)
            df_empty.to_csv(self.daily_csv, index=False)
            logger.info(f"Cleared daily CSV: {self.daily_csv}")
        except Exception as e:
            logger.exception("Failed to clear daily CSV")

    async def _wait_until_midnight(self):
        """
        Wait until next midnight (0:00 AM).
        """
        now = datetime.now()
        # Calculate next midnight
        next_midnight = datetime.combine(
            now.date(),
            time(0, 0, 0)
        )

        # If we're past midnight today, go to tomorrow
        if now.time() >= time(0, 0, 0):
            from datetime import timedelta
            next_midnight = next_midnight + timedelta(days=1)

        # Calculate seconds to wait
        wait_seconds = (next_midnight - now).total_seconds()

        logger.info(
            f"Next aggregation scheduled at {next_midnight.isoformat()}, "
            f"waiting {wait_seconds:.0f} seconds"
        )

        await asyncio.sleep(wait_seconds)

    async def run(self):
        """
        Run the scheduler loop.
        """
        self.running = True
        logger.info("Daily aggregation scheduler started")

        while self.running:
            try:
                # Wait until midnight
                await self._wait_until_midnight()

                # Perform aggregation
                logger.info("Starting daily aggregation...")
                stats = await self.aggregate_daily_data()
                logger.info(f"Aggregation stats: {stats}")

            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break
            except Exception as e:
                logger.exception("Scheduler error")
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)

        logger.info("Daily aggregation scheduler stopped")

    def start(self):
        """
        Start the scheduler in background.
        """
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())
            logger.info("Scheduler task created")

    async def stop(self):
        """
        Stop the scheduler.
        """
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def trigger_now(self) -> dict:
        """
        Manually trigger aggregation immediately (for testing).
        """
        logger.info("Manual aggregation triggered")
        return await self.aggregate_daily_data()
