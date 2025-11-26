"""
bankTrading.ml package
Expose the M5P API and core classes.
"""
from .ml_m5p import CompleteM5PRegressor, M5PModelAPI, m5p_model

__all__ = ["CompleteM5PRegressor", "M5PModelAPI", "m5p_model"]
__version__ = "0.1.0"
__author__ = "Thai Phuc Hung"
