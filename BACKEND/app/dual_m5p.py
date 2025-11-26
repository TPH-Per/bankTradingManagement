from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .ml_m5p import (
    M5PModelAPI,
    REQUIRED_FEATURES_IN,
    REQUIRED_FEATURES_OUT,
)


class DualCashModelAPI:
    """
    Wrapper that maintains two separate M5P models:
    - cash-in (predicts next-day inflow)
    - cash-out (predicts next-day outflow)
    """

    def __init__(
        self,
        ci_dir: Union[str, Path] = "models/cash_in",
        co_dir: Union[str, Path] = "models/cash_out",
    ) -> None:
        ci_path = Path(ci_dir)
        co_path = Path(co_dir)
        ci_path.mkdir(parents=True, exist_ok=True)
        co_path.mkdir(parents=True, exist_ok=True)

        self.ci = M5PModelAPI(
            model_dir=ci_path,
            model_filename="m5p_cash_in.pkl",
            schema_filename="schema_cash_in.json",
        )
        self.ci.required_features_ = list(REQUIRED_FEATURES_IN)

        self.co = M5PModelAPI(
            model_dir=co_path,
            model_filename="m5p_cash_out.pkl",
            schema_filename="schema_cash_out.json",
        )
        self.co.required_features_ = list(REQUIRED_FEATURES_OUT)

    def bootstrap(
        self,
        dataset_csv: Union[str, Path],
        force_retrain: bool = False,
        ci_target: str = "cash_in_next_day",
        co_target: str = "cash_out_next_day",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Ensure both models are ready. If persisted artifacts exist and
        force_retrain is False, load them. Otherwise, train from the dataset.
        """
        dataset_path = Path(dataset_csv)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        stats: Dict[str, Dict[str, Any]] = {}

        # Cash-In model
        if force_retrain or not self.ci.has_persisted_model():
            stats["cash_in"] = self.ci.train(str(dataset_path), target_column=ci_target)
        else:
            self.ci.load_from_disk()
            path = self.ci._find_existing_model_file()
            stats["cash_in"] = {
                "status": "loaded",
                "model_path": str(path) if path else None,
            }

        # Cash-Out model
        if force_retrain or not self.co.has_persisted_model():
            stats["cash_out"] = self.co.train(str(dataset_path), target_column=co_target)
        else:
            self.co.load_from_disk()
            path = self.co._find_existing_model_file()
            stats["cash_out"] = {
                "status": "loaded",
                "model_path": str(path) if path else None,
            }

        return stats

    def predict_both(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict both cash-in and cash-out for the next day.
        Values are clipped to be non-negative.
        """
        ci_pred = max(0.0, self.ci.predict(dict(features)))
        co_pred = max(0.0, self.co.predict(dict(features)))
        return {
            "cash_in_next_day": ci_pred,
            "cash_out_next_day": co_pred,
        }

    def retrain_both(
        self,
        dataset_csv: Union[str, Path],
        ci_target: str = "cash_in_next_day",
        co_target: str = "cash_out_next_day"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrain both cash-in and cash-out models with the given dataset.
        """
        dataset_path = Path(dataset_csv)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        stats: Dict[str, Dict[str, Any]] = {}

        # Retrain Cash-In model
        stats["cash_in"] = self.ci.train(str(dataset_path), target_column=ci_target)

        # Retrain Cash-Out model
        stats["cash_out"] = self.co.train(str(dataset_path), target_column=co_target)

        return stats

    def predict_cash_in(self, features: Dict[str, Any]) -> float:
        """
        Predict only cash_in_next_day using the cash_in model.
        """
        return max(0.0, self.ci.predict(dict(features)))

    def predict_cash_out(self, features: Dict[str, Any]) -> float:
        """
        Predict only cash_out_next_day using the cash_out model.
        """
        return max(0.0, self.co.predict(dict(features)))

    def is_ready(self) -> bool:
        """
        Check if both models are ready (trained or loaded).
        """
        return bool(getattr(self.ci, "is_trained", False) or self.ci.has_persisted_model()) and \
               bool(getattr(self.co, "is_trained", False) or self.co.has_persisted_model())
