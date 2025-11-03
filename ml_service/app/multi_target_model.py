"""
Multi-target cash flow prediction system.

Trains 4 specialized models:
1. cash_in_next_day - Tomorrow's cash in
2. cash_out_next_day - Tomorrow's cash out
3. cash_in_h7_sum - Total cash in next 7 days
4. cash_out_h7_sum - Total cash out next 7 days
5. cash_in_next_month_sum - Total cash in next calendar month
6. cash_out_next_month_sum - Total cash out next calendar month
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ml_m5p import M5PModelAPI


class MultiTargetCashModel:
    """
    Manages 6 separate M5P models for comprehensive cash flow forecasting.
    """

    def __init__(self, base_model_dir: Union[str, Path] = "models"):
        """
        Initialize all 6 prediction models.

        Args:
            base_model_dir: Base directory for all model artifacts
        """
        base_path = Path(base_model_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Next day predictions
        self.cash_in_next_day = M5PModelAPI(
            model_dir=base_path / "cash_in_next_day",
            model_filename="m5p_cash_in_next_day.pkl",
            schema_filename="schema_cash_in_next_day.json",
        )

        self.cash_out_next_day = M5PModelAPI(
            model_dir=base_path / "cash_out_next_day",
            model_filename="m5p_cash_out_next_day.pkl",
            schema_filename="schema_cash_out_next_day.json",
        )

        # Next 7 days sum predictions
        self.cash_in_h7 = M5PModelAPI(
            model_dir=base_path / "cash_in_h7",
            model_filename="m5p_cash_in_h7.pkl",
            schema_filename="schema_cash_in_h7.json",
        )

        self.cash_out_h7 = M5PModelAPI(
            model_dir=base_path / "cash_out_h7",
            model_filename="m5p_cash_out_h7.pkl",
            schema_filename="schema_cash_out_h7.json",
        )

        # Next month sum predictions
        self.cash_in_next_month = M5PModelAPI(
            model_dir=base_path / "cash_in_next_month",
            model_filename="m5p_cash_in_next_month.pkl",
            schema_filename="schema_cash_in_next_month.json",
        )

        self.cash_out_next_month = M5PModelAPI(
            model_dir=base_path / "cash_out_next_month",
            model_filename="m5p_cash_out_next_month.pkl",
            schema_filename="schema_cash_out_next_month.json",
        )

    def bootstrap(
        self,
        dataset_csv: Union[str, Path],
        force_retrain: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Bootstrap all 6 models on startup.

        Args:
            dataset_csv: Path to training dataset
            force_retrain: Force retrain even if models exist

        Returns:
            Dict with stats for each model
        """
        dataset_path = Path(dataset_csv)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        stats: Dict[str, Dict[str, Any]] = {}

        # Define all models with their target columns
        models = [
            (self.cash_in_next_day, "cash_in_next_day", "cash_in_next_day"),
            (self.cash_out_next_day, "cash_out_next_day", "cash_out_next_day"),
            (self.cash_in_h7, "cash_in_h7_sum", "cash_in_h7"),
            (self.cash_out_h7, "cash_out_h7_sum", "cash_out_h7"),
            (self.cash_in_next_month, "cash_in_next_month_sum", "cash_in_next_month"),
            (self.cash_out_next_month, "cash_out_next_month_sum", "cash_out_next_month"),
        ]

        for model, target_col, stats_key in models:
            if force_retrain or not model.has_persisted_model():
                stats[stats_key] = model.train(str(dataset_path), target_column=target_col)
            else:
                model.load_from_disk()
                path = model._find_existing_model_file()
                stats[stats_key] = {
                    "status": "loaded",
                    "model_path": str(path) if path else None,
                }

        return stats

    def retrain_all(
        self,
        dataset_csv: Union[str, Path],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrain all 6 models with new data.

        Args:
            dataset_csv: Path to training dataset

        Returns:
            Dict with training metrics for each model
        """
        dataset_path = Path(dataset_csv)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        stats: Dict[str, Dict[str, Any]] = {}

        # Retrain all models
        stats["cash_in_next_day"] = self.cash_in_next_day.train(
            str(dataset_path), target_column="cash_in_next_day"
        )
        stats["cash_out_next_day"] = self.cash_out_next_day.train(
            str(dataset_path), target_column="cash_out_next_day"
        )
        stats["cash_in_h7"] = self.cash_in_h7.train(
            str(dataset_path), target_column="cash_in_h7_sum"
        )
        stats["cash_out_h7"] = self.cash_out_h7.train(
            str(dataset_path), target_column="cash_out_h7_sum"
        )
        stats["cash_in_next_month"] = self.cash_in_next_month.train(
            str(dataset_path), target_column="cash_in_next_month_sum"
        )
        stats["cash_out_next_month"] = self.cash_out_next_month.train(
            str(dataset_path), target_column="cash_out_next_month_sum"
        )

        return stats

    def predict_all(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate all 6 predictions at once.

        Args:
            features: Feature dictionary

        Returns:
            Dict with all 6 predictions
        """
        return {
            "cash_in_next_day": max(0.0, self.cash_in_next_day.predict(dict(features))),
            "cash_out_next_day": max(0.0, self.cash_out_next_day.predict(dict(features))),
            "cash_in_h7_sum": max(0.0, self.cash_in_h7.predict(dict(features))),
            "cash_out_h7_sum": max(0.0, self.cash_out_h7.predict(dict(features))),
            "cash_in_next_month_sum": max(0.0, self.cash_in_next_month.predict(dict(features))),
            "cash_out_next_month_sum": max(0.0, self.cash_out_next_month.predict(dict(features))),
        }

    def predict_cash_in(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict all cash_in targets (next_day, h7, next_month).

        Args:
            features: Feature dictionary

        Returns:
            Dict with 3 cash_in predictions
        """
        return {
            "next_day": max(0.0, self.cash_in_next_day.predict(dict(features))),
            "h7_sum": max(0.0, self.cash_in_h7.predict(dict(features))),
            "next_month_sum": max(0.0, self.cash_in_next_month.predict(dict(features))),
        }

    def predict_cash_out(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict all cash_out targets (next_day, h7, next_month).

        Args:
            features: Feature dictionary

        Returns:
            Dict with 3 cash_out predictions
        """
        return {
            "next_day": max(0.0, self.cash_out_next_day.predict(dict(features))),
            "h7_sum": max(0.0, self.cash_out_h7.predict(dict(features))),
            "next_month_sum": max(0.0, self.cash_out_next_month.predict(dict(features))),
        }

    def is_ready(self) -> bool:
        """Check if all models are trained and ready."""
        models = [
            self.cash_in_next_day,
            self.cash_out_next_day,
            self.cash_in_h7,
            self.cash_out_h7,
            self.cash_in_next_month,
            self.cash_out_next_month,
        ]

        return all(
            bool(getattr(m, "is_trained", False) or m.has_persisted_model())
            for m in models
        )
