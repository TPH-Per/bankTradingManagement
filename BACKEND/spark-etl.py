#!/usr/bin/env python3
"""
Spark ETL Pipeline for BankTrading Cash Flow Prediction

Usage:
    spark-submit spark-etl.py --mode local
    spark-submit --master yarn spark-etl.py --mode hdfs
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, sum as spark_sum, count, countDistinct, first,
    lag, avg, when, dayofweek, month, quarter,
    to_date, row_number
)


class SparkETLPipeline:
    def __init__(self, mode="local", hdfs_base="hdfs://localhost:9000/banktrading", local_base="data"):
        self.mode = mode
        self.hdfs_base = hdfs_base
        self.local_base = Path(local_base)

        builder = SparkSession.builder \
            .appName("BankTrading ETL") \
            .config("spark.sql.parquet.compression.codec", "snappy")

        if mode == "local":
            builder = builder.master("local[*]")

        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        print(f"Spark ETL initialized in {mode} mode")

    def get_path(self, relative_path):
        return f"{self.hdfs_base}/{relative_path}" if self.mode == "hdfs" else str(self.local_base / relative_path)

    def read_daily_csv(self, filename="cash_daily.csv"):
        path = self.get_path(filename)
        print(f"Reading: {path}")
        df = self.spark.read.csv(path, header=True, inferSchema=True)
        print(f"Read {df.count()} rows")
        return df

    def aggregate_transactions(self, df):
        print("Aggregating...")
        df = df.withColumn("date", to_date(col("date")))

        agg_df = df.groupBy("date").agg(
            spark_sum("cash_in").alias("cash_in"),
            spark_sum("cash_out").alias("cash_out"),
            first("channel").alias("channel"),
            count("*").alias("transaction_count")
        )

        agg_df = agg_df.withColumn("balance", col("cash_in") - col("cash_out")) \
            .withColumn("day_of_week", dayofweek(col("date")) - 1) \
            .withColumn("month", month(col("date"))) \
            .withColumn("quarter", quarter(col("date"))) \
            .withColumn("channel", when(col("channel").isNull(), "DEFAULT").otherwise(col("channel")))

        return agg_df

    def engineer_features(self, df):
        print("Engineering features...")
        window_spec = Window.orderBy("date")

        # Lag features
        df = df.withColumn("lag1_in", lag(col("cash_in"), 1).over(window_spec)) \
            .withColumn("lag7_in", lag(col("cash_in"), 7).over(window_spec)) \
            .withColumn("lag1_out", lag(col("cash_out"), 1).over(window_spec)) \
            .withColumn("lag7_out", lag(col("cash_out"), 7).over(window_spec))

        # Rolling windows
        for window_size in [7, 30]:
            window_rolling = Window.orderBy("date").rowsBetween(-window_size + 1, 0)
            df = df.withColumn(f"roll_mean_{window_size}_in", avg(col("cash_in")).over(window_rolling)) \
                .withColumn(f"roll_mean_{window_size}_out", avg(col("cash_out")).over(window_rolling))

        # Fill nulls
        for col_name in df.columns:
            if "lag" in col_name or "roll" in col_name:
                df = df.withColumn(col_name, when(col(col_name).isNull(), 0.0).otherwise(col(col_name)))

        return df

    def create_targets(self, df):
        print("Creating targets...")
        window_spec = Window.orderBy("date")

        df = df.withColumn("cash_in_next_day", lag(col("cash_in"), -1).over(window_spec)) \
            .withColumn("cash_out_next_day", lag(col("cash_out"), -1).over(window_spec))

        window_7days = Window.orderBy("date").rowsBetween(1, 7)
        df = df.withColumn("cash_in_h7_sum", spark_sum(col("cash_in")).over(window_7days)) \
            .withColumn("cash_out_h7_sum", spark_sum(col("cash_out")).over(window_7days))

        window_30days = Window.orderBy("date").rowsBetween(1, 30)
        df = df.withColumn("cash_in_next_month_sum", spark_sum(col("cash_in")).over(window_30days)) \
            .withColumn("cash_out_next_month_sum", spark_sum(col("cash_out")).over(window_30days))

        return df

    def merge_with_existing(self, new_df, existing_path):
        full_path = self.get_path(existing_path)
        print(f"Merging with: {full_path}")

        try:
            existing_df = self.spark.read.csv(full_path, header=True, inferSchema=True)
            existing_df = existing_df.withColumn("date", to_date(col("date")))

            merged_df = existing_df.union(new_df)
            window_spec = Window.partitionBy("date").orderBy(col("date").desc())
            merged_df = merged_df.withColumn("row_num", row_number().over(window_spec)) \
                .filter(col("row_num") == 1).drop("row_num")

            return merged_df.orderBy("date")
        except:
            print("No existing data, using new data only")
            return new_df.orderBy("date")

    def run_daily_pipeline(self, clear_after=True):
        print("=" * 70)
        print("SPARK ETL PIPELINE")
        print("=" * 70)

        stats = {"status": "started", "rows_processed": 0}

        try:
            df = self.read_daily_csv("cash_daily.csv")
            if df.count() == 0:
                print("No data")
                stats["status"] = "skipped"
                return stats

            df_agg = self.aggregate_transactions(df)
            df_features = self.engineer_features(df_agg)
            df_with_targets = self.create_targets(df_features)
            df_merged = self.merge_with_existing(df_with_targets, "cash_daily_train_realistic.csv")

            stats["rows_processed"] = df_merged.count()

            # Write output
            output_path = str(self.local_base / "cash_daily_train_realistic.csv")
            df_merged.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path + "_temp")

            # Move CSV to final location
            import glob, shutil
            csv_files = glob.glob(output_path + "_temp/*.csv")
            if csv_files:
                shutil.move(csv_files[0], output_path)
                shutil.rmtree(output_path + "_temp")
                print(f"Wrote: {output_path}")

            # Clear daily CSV
            if clear_after and self.mode == "local":
                csv_path = self.local_base / "cash_daily.csv"
                self.spark.read.csv(str(csv_path), header=True).limit(0) \
                    .write.mode("overwrite").option("header", True).csv(str(csv_path))
                print("Cleared daily CSV")

            stats["status"] = "success"
            print(f"SUCCESS: {stats['rows_processed']} rows")

        except Exception as e:
            stats["status"] = "error"
            stats["error"] = str(e)
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        return stats

    def stop(self):
        self.spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Spark ETL for BankTrading")
    parser.add_argument("--mode", choices=["local", "hdfs"], default="local")
    parser.add_argument("--hdfs-base", default="hdfs://localhost:9000/banktrading")
    parser.add_argument("--local-base", default="data")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    pipeline = SparkETLPipeline(args.mode, args.hdfs_base, args.local_base)

    try:
        stats = pipeline.run_daily_pipeline(clear_after=not args.no_clear)
        print(f"\nStats: {stats}")
        sys.exit(0 if stats["status"] == "success" else 1)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
