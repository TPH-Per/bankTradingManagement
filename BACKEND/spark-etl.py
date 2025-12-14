#!/usr/bin/env python3
"""
Spark ETL Pipeline for BankTrading Cash Flow Prediction

Usage:
    # Local CSV mode
    spark-submit spark-etl.py --mode local
    
    # Cassandra mode (read from database)
    spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.12:3.4.1 spark-etl.py --mode cassandra
    
    # HDFS mode
    spark-submit --master yarn spark-etl.py --mode hdfs
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, sum as spark_sum, count, countDistinct, first,
    lag, avg, when, dayofweek, month, quarter,
    to_date, row_number, lit
)


class SparkETLPipeline:
    def __init__(self, mode="local", hdfs_base="hdfs://localhost:9000/banktrading", local_base="data",
                 cassandra_host="localhost", cassandra_keyspace="pldt_rt"):
        self.mode = mode
        self.hdfs_base = hdfs_base
        self.local_base = Path(local_base)
        self.cassandra_host = cassandra_host
        self.cassandra_keyspace = cassandra_keyspace

        builder = SparkSession.builder \
            .appName("BankTrading ETL") \
            .config("spark.sql.parquet.compression.codec", "snappy")

        if mode == "local":
            builder = builder.master("local[*]")
        
        # Add Cassandra connector config
        if mode == "cassandra":
            builder = builder \
                .config("spark.cassandra.connection.host", cassandra_host) \
                .config("spark.cassandra.connection.port", "9042")

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

    def read_from_cassandra(self, days_back=90):
        """
        Read transactions directly from Cassandra.
        Requires spark-cassandra-connector package.
        
        Table schema expected:
        - account_id, event_date, event_ts, tx_id
        - direction (CREDIT/DEBIT), amount, currency
        - transaction_type (deposit/withdrawal/p2p_transfer)
        """
        print(f"Reading from Cassandra: {self.cassandra_keyspace}.transactions (last {days_back} days)")
        
        try:
            # Read transactions table
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(table="transactions", keyspace=self.cassandra_keyspace) \
                .load()
            
            # Filter by date range
            cutoff_date = (datetime.now() - timedelta(days=days_back)).date()
            df = df.filter(col("event_date") >= lit(cutoff_date))
            
            print(f"Read {df.count()} transactions from Cassandra")
            return df
        except Exception as e:
            print(f"Error reading from Cassandra: {e}")
            raise

    def aggregate_cassandra_transactions(self, df):
        """
        Aggregate Cassandra transactions by date.
        - CREDIT (deposit) → cash_in
        - DEBIT (withdrawal) → cash_out
        """
        print("Aggregating Cassandra transactions...")
        
        agg_df = df.groupBy(col("event_date").alias("date")).agg(
            spark_sum(when(col("direction") == "CREDIT", col("amount")).otherwise(0)).alias("cash_in"),
            spark_sum(when(col("direction") == "DEBIT", col("amount")).otherwise(0)).alias("cash_out"),
            count("*").alias("transaction_count"),
            countDistinct("account_id").alias("unique_accounts")
        )
        
        agg_df = agg_df.withColumn("balance", col("cash_in") - col("cash_out")) \
            .withColumn("day_of_week", dayofweek(col("date")) - 1) \
            .withColumn("month", month(col("date"))) \
            .withColumn("quarter", quarter(col("date")))
        
        print(f"Aggregated to {agg_df.count()} daily records")
        return agg_df

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

        stats = {"status": "started", "rows_processed": 0, "mode": self.mode}

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

    def run_cassandra_pipeline(self, days_back=90):
        """
        Full pipeline reading from Cassandra:
        1. Read transactions from Cassandra
        2. Aggregate by date (CREDIT → cash_in, DEBIT → cash_out)
        3. Engineer features
        4. Create prediction targets
        5. Save to CSV for model training
        """
        print("=" * 70)
        print("SPARK ETL PIPELINE (CASSANDRA MODE)")
        print("=" * 70)

        stats = {"status": "started", "rows_processed": 0, "mode": "cassandra"}

        try:
            # Read from Cassandra
            df = self.read_from_cassandra(days_back)
            if df.count() == 0:
                print("No transactions in Cassandra")
                stats["status"] = "skipped"
                return stats

            # Aggregate
            df_agg = self.aggregate_cassandra_transactions(df)
            
            # Engineer features
            df_features = self.engineer_features(df_agg)
            
            # Create targets
            df_with_targets = self.create_targets(df_features)
            
            stats["rows_processed"] = df_with_targets.count()

            # Write output
            output_path = str(self.local_base / "cash_daily_train_realistic.csv")
            df_with_targets.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path + "_temp")

            # Move CSV to final location
            import glob, shutil
            csv_files = glob.glob(output_path + "_temp/*.csv")
            if csv_files:
                shutil.move(csv_files[0], output_path)
                shutil.rmtree(output_path + "_temp")
                print(f"Wrote: {output_path}")

            stats["status"] = "success"
            print(f"SUCCESS: {stats['rows_processed']} daily records from Cassandra")

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
    parser.add_argument("--mode", choices=["local", "hdfs", "cassandra"], default="local",
                       help="Data source mode: local (CSV), hdfs (HDFS CSV), cassandra (Cassandra DB)")
    parser.add_argument("--hdfs-base", default="hdfs://localhost:9000/banktrading")
    parser.add_argument("--local-base", default="data")
    parser.add_argument("--cassandra-host", default="localhost")
    parser.add_argument("--cassandra-keyspace", default="pldt_rt")
    parser.add_argument("--days-back", type=int, default=90, help="Days of history to read from Cassandra")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    pipeline = SparkETLPipeline(
        args.mode, 
        args.hdfs_base, 
        args.local_base,
        args.cassandra_host,
        args.cassandra_keyspace
    )

    try:
        if args.mode == "cassandra":
            stats = pipeline.run_cassandra_pipeline(days_back=args.days_back)
        else:
            stats = pipeline.run_daily_pipeline(clear_after=not args.no_clear)
        
        print(f"\nStats: {stats}")
        sys.exit(0 if stats["status"] == "success" else 1)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

