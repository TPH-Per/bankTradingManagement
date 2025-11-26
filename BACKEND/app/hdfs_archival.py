"""
HDFS Archival Service

Archives old data from Cassandra to HDFS for long-term storage and analytics.
This reduces Cassandra storage costs and enables historical analytics.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, count, min as spark_min, max as spark_max
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logging.warning("PySpark not available. HDFS archival features disabled.")

logger = logging.getLogger(__name__)


class HDFSArchivalService:
    """
    Service for archiving data from Cassandra to HDFS.
    
    Features:
    - Archive old transactions to HDFS
    - Archive audit logs to HDFS
    - Archive balance snapshots to HDFS
    - Support for partitioned storage by date
    """
    
    def __init__(
        self,
        hdfs_base: str = "hdfs://localhost:9000/banktrading",
        cassandra_host: str = "localhost",
        cassandra_keyspace_rt: str = "pldt_rt",
        cassandra_keyspace_audit: str = "pldt_audit"
    ):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required for HDFS archival. Install with: pip install pyspark")
        
        self.hdfs_base = hdfs_base
        self.cassandra_host = cassandra_host
        self.keyspace_rt = cassandra_keyspace_rt
        self.keyspace_audit = cassandra_keyspace_audit
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("HDFS Archival Service") \
            .config("spark.cassandra.connection.host", cassandra_host) \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info(f"HDFS Archival Service initialized. HDFS base: {hdfs_base}")
    
    def archive_transactions(
        self,
        days_to_keep: int = 90,
        delete_after_archive: bool = False
    ) -> Dict[str, Any]:
        """
        Archive transactions older than specified days from Cassandra to HDFS.
        
        Args:
            days_to_keep: Keep transactions newer than this many days in Cassandra
            delete_after_archive: If True, delete archived data from Cassandra (use with caution!)
        
        Returns:
            Dictionary with archival statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        logger.info(f"Archiving transactions older than {cutoff_date.date()}")
        
        try:
            # Read from Cassandra
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(
                    table="tx_by_account_day",
                    keyspace=self.keyspace_rt
                ) \
                .load()
            
            # Filter old data
            old_data = df.filter(col("event_date") < cutoff_date.date())
            row_count = old_data.count()
            
            if row_count == 0:
                logger.info("No transactions to archive")
                return {
                    "status": "success",
                    "rows_archived": 0,
                    "cutoff_date": cutoff_date.date().isoformat()
                }
            
            # Write to HDFS partitioned by date
            output_path = f"{self.hdfs_base}/archived/transactions"
            old_data.write \
                .mode("append") \
                .partitionBy("event_date") \
                .parquet(output_path)
            
            logger.info(f"Archived {row_count} transactions to {output_path}")
            
            # Optionally delete from Cassandra (commented out for safety)
            if delete_after_archive:
                logger.warning("Delete after archive is enabled. This will remove data from Cassandra!")
                # TODO: Implement safe deletion with backup verification
                # old_data.write \
                #     .format("org.apache.spark.sql.cassandra") \
                #     .options(table="tx_by_account_day", keyspace=self.keyspace_rt) \
                #     .mode("append") \
                #     .save()
            
            return {
                "status": "success",
                "rows_archived": row_count,
                "cutoff_date": cutoff_date.date().isoformat(),
                "output_path": output_path
            }
            
        except Exception as e:
            logger.exception(f"Error archiving transactions: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def archive_audit_logs(
        self,
        days_to_keep: int = 30,
        delete_after_archive: bool = False
    ) -> Dict[str, Any]:
        """
        Archive audit logs older than specified days from Cassandra to HDFS.
        
        Args:
            days_to_keep: Keep logs newer than this many days in Cassandra
            delete_after_archive: If True, delete archived data from Cassandra
        
        Returns:
            Dictionary with archival statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        logger.info(f"Archiving audit logs older than {cutoff_date.date()}")
        
        try:
            # Read from Cassandra
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(
                    table="api_calls",
                    keyspace=self.keyspace_audit
                ) \
                .load()
            
            # Filter old data
            old_logs = df.filter(col("day") < cutoff_date.date())
            row_count = old_logs.count()
            
            if row_count == 0:
                logger.info("No audit logs to archive")
                return {
                    "status": "success",
                    "rows_archived": 0,
                    "cutoff_date": cutoff_date.date().isoformat()
                }
            
            # Write to HDFS partitioned by day
            output_path = f"{self.hdfs_base}/archived/audit_logs"
            old_logs.write \
                .mode("append") \
                .partitionBy("day") \
                .parquet(output_path)
            
            logger.info(f"Archived {row_count} audit logs to {output_path}")
            
            return {
                "status": "success",
                "rows_archived": row_count,
                "cutoff_date": cutoff_date.date().isoformat(),
                "output_path": output_path
            }
            
        except Exception as e:
            logger.exception(f"Error archiving audit logs: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def archive_balance_snapshots(
        self,
        days_to_keep: int = 365,
        delete_after_archive: bool = False
    ) -> Dict[str, Any]:
        """
        Archive balance snapshots older than specified days from Cassandra to HDFS.
        
        Args:
            days_to_keep: Keep snapshots newer than this many days in Cassandra
            delete_after_archive: If True, delete archived data from Cassandra
        
        Returns:
            Dictionary with archival statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        logger.info(f"Archiving balance snapshots older than {cutoff_date.date()}")
        
        try:
            # Read from Cassandra
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(
                    table="balance_daily_snapshots",
                    keyspace=self.keyspace_rt
                ) \
                .load()
            
            # Filter old data
            old_snapshots = df.filter(col("day") < cutoff_date.date())
            row_count = old_snapshots.count()
            
            if row_count == 0:
                logger.info("No balance snapshots to archive")
                return {
                    "status": "success",
                    "rows_archived": 0,
                    "cutoff_date": cutoff_date.date().isoformat()
                }
            
            # Write to HDFS partitioned by day
            output_path = f"{self.hdfs_base}/archived/balance_snapshots"
            old_snapshots.write \
                .mode("append") \
                .partitionBy("day") \
                .parquet(output_path)
            
            logger.info(f"Archived {row_count} balance snapshots to {output_path}")
            
            return {
                "status": "success",
                "rows_archived": row_count,
                "cutoff_date": cutoff_date.date().isoformat(),
                "output_path": output_path
            }
            
        except Exception as e:
            logger.exception(f"Error archiving balance snapshots: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_archived_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about archived data in HDFS.
        
        Returns:
            Dictionary with statistics for each archived dataset
        """
        stats = {}
        
        try:
            # Check transactions
            try:
                tx_df = self.spark.read.parquet(
                    f"{self.hdfs_base}/archived/transactions"
                )
                tx_count = tx_df.count()
                date_range = tx_df.agg(
                    spark_min("event_date").alias("min_date"),
                    spark_max("event_date").alias("max_date")
                ).collect()[0]
                
                stats["transactions"] = {
                    "row_count": tx_count,
                    "min_date": str(date_range["min_date"]),
                    "max_date": str(date_range["max_date"])
                }
            except Exception as e:
                stats["transactions"] = {"error": str(e)}
            
            # Check audit logs
            try:
                audit_df = self.spark.read.parquet(
                    f"{self.hdfs_base}/archived/audit_logs"
                )
                audit_count = audit_df.count()
                date_range = audit_df.agg(
                    spark_min("day").alias("min_date"),
                    spark_max("day").alias("max_date")
                ).collect()[0]
                
                stats["audit_logs"] = {
                    "row_count": audit_count,
                    "min_date": str(date_range["min_date"]),
                    "max_date": str(date_range["max_date"])
                }
            except Exception as e:
                stats["audit_logs"] = {"error": str(e)}
            
            # Check balance snapshots
            try:
                balance_df = self.spark.read.parquet(
                    f"{self.hdfs_base}/archived/balance_snapshots"
                )
                balance_count = balance_df.count()
                date_range = balance_df.agg(
                    spark_min("day").alias("min_date"),
                    spark_max("day").alias("max_date")
                ).collect()[0]
                
                stats["balance_snapshots"] = {
                    "row_count": balance_count,
                    "min_date": str(date_range["min_date"]),
                    "max_date": str(date_range["max_date"])
                }
            except Exception as e:
                stats["balance_snapshots"] = {"error": str(e)}
            
        except Exception as e:
            logger.exception(f"Error getting archived data stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def stop(self):
        """Stop Spark session"""
        if hasattr(self, 'spark'):
            self.spark.stop()
            logger.info("HDFS Archival Service stopped")


def create_archival_service(
    hdfs_base: Optional[str] = None,
    cassandra_host: Optional[str] = None
) -> Optional[HDFSArchivalService]:
    """
    Factory function to create HDFS archival service.
    Returns None if PySpark is not available.
    """
    if not SPARK_AVAILABLE:
        logger.warning("PySpark not available. HDFS archival service cannot be created.")
        return None
    
    return HDFSArchivalService(
        hdfs_base=hdfs_base or "hdfs://localhost:9000/banktrading",
        cassandra_host=cassandra_host or "localhost"
    )

