
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Sequence
import joblib
import warnings
import os
import json
from pathlib import Path
from dataclasses import dataclass, field

# Core ML libraries
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "m5p_model.pkl"
DEFAULT_SCHEMA_PATH = BASE_DIR / "models" / "m5p_schema.json"
DEFAULT_TARGET_COLUMN = "amount_next_day"
DEFAULT_CURRENCY = "VND"
REQUIRED_FEATURES: Sequence[str] = (
    "cash_in_d0",
    "cash_out_d0",
    "cash_net_d0",
    "lag1_in",
    "lag7_in",
    "roll_mean_7_in",
    "dow",
    "is_weekend",
    "is_month_end",
    "is_payday",
    "channel",
)

# Feature sets for specialized models
REQUIRED_FEATURES_IN: Sequence[str] = (
    "cash_in_d0",
    "cash_out_d0",
    "cash_net_d0",
    "lag1_in",
    "lag7_in",
    "roll_mean_7_in",
    "dow",
    "is_weekend",
    "is_month_end",
    "is_payday",
    "channel",
)

REQUIRED_FEATURES_OUT: Sequence[str] = (
    "cash_in_d0",
    "cash_out_d0",
    "cash_net_d0",
    "lag1_out",
    "lag7_out",
    "roll_mean_7_out",
    "dow",
    "is_weekend",
    "is_month_end",
    "is_payday",
    "channel",
)


@dataclass
class M5PNode:
    """Node in the M5P model tree"""

    node_id: int
    depth: int
    is_leaf: bool

    split_feature: Optional[int] = None
    split_value: Optional[float] = None

    linear_model: Optional[Ridge] = None
    model_features: List[int] = field(default_factory=list)

    n_samples: int = 0
    std_dev: float = 0.0
    mean_target: float = 0.0

    left_child: Optional['M5PNode'] = None
    right_child: Optional['M5PNode'] = None
    parent: Optional['M5PNode'] = None

    pruning_error: float = float('inf')


class M5PTree:
    """
    Core M5P Model Tree - Version 2.2

    All fixes from v2.1 + proper feature name mapping
    """

    def __init__(self,
                 min_samples_leaf: int = 4,
                 max_depth: int = 10,
                 min_std_reduction: float = 0.05,
                 use_pruning: bool = True,
                 use_smoothing: bool = True,
                 smoothing_constant: int = 15,
                 pruning_factor: float = 2.0,
                 pruning_method: str = 'mse',
                 ridge_alpha: float = 0.01,
                 n_split_percentiles: int = 32,
                 random_state: Optional[int] = None):

        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_std_reduction = min_std_reduction
        self.use_pruning = use_pruning
        self.use_smoothing = use_smoothing
        self.smoothing_constant = smoothing_constant
        self.pruning_factor = pruning_factor
        self.pruning_method = pruning_method
        self.ridge_alpha = ridge_alpha
        self.n_split_percentiles = n_split_percentiles
        self.random_state = random_state

        self.root = None
        self.node_count = 0
        self.feature_importances_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _calculate_sdr(self, y: np.ndarray, split_mask: np.ndarray) -> float:
        """Calculate Standard Deviation Reduction (SDR)"""
        n = len(y)
        if n == 0:
            return 0.0

        n_left = np.sum(split_mask)
        n_right = n - n_left

        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return -float('inf')

        sd_original = np.std(y)

        y_left = y[split_mask]
        y_right = y[~split_mask]

        sd_left = np.std(y_left) if n_left > 0 else 0
        sd_right = np.std(y_right) if n_right > 0 else 0

        sd_weighted = (n_left/n * sd_left) + (n_right/n * sd_right)
        sdr = sd_original - sd_weighted

        return sdr

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split using smart percentile thresholds"""
        n_samples, n_features = X.shape

        if n_samples < 2 * self.min_samples_leaf:
            return None, None, -float('inf')

        best_feature = None
        best_value = None
        best_sdr = -float('inf')

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]

            if np.std(feature_values) < 1e-10:
                continue

            # Exclude 0% and 100% percentiles
            percentiles = np.linspace(0, 100, self.n_split_percentiles + 2)[1:-1]
            thresholds = np.unique(np.percentile(feature_values, percentiles))

            for threshold in thresholds:
                split_mask = feature_values <= threshold

                if np.sum(split_mask) < self.min_samples_leaf or                    np.sum(~split_mask) < self.min_samples_leaf:
                    continue

                sdr = self._calculate_sdr(y, split_mask)

                if sdr > best_sdr:
                    best_sdr = sdr
                    best_feature = feature_idx
                    best_value = threshold

        return best_feature, best_value, best_sdr

    def _build_linear_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Ridge, List[int]]:
        """Build Ridge regression model for node"""
        n_samples, n_features = X.shape

        features = list(range(n_features))
        features = [f for f in features if np.std(X[:, f]) > 1e-10]

        if len(features) == 0 or n_samples < len(features) + 1:
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(np.ones((n_samples, 1)), y)
            return model, []

        model = Ridge(alpha=self.ridge_alpha)
        try:
            model.fit(X[:, features], y)
        except Exception as e:
            warnings.warn(f"Ridge fitting failed: {e}. Using constant model.")
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(np.ones((n_samples, 1)), y)
            return model, []

        return model, features

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0, 
                   parent: Optional[M5PNode] = None) -> M5PNode:
        """Recursively build the M5P tree"""
        n_samples = len(y)

        node = M5PNode(
            node_id=self.node_count,
            depth=depth,
            is_leaf=False,
            n_samples=n_samples,
            std_dev=float(np.std(y)) if n_samples > 0 else 0.0,
            mean_target=float(np.mean(y)) if n_samples > 0 else 0.0,
            parent=parent
        )
        self.node_count += 1

        should_stop = (
            depth >= self.max_depth or
            n_samples < 2 * self.min_samples_leaf or
            np.std(y) < 1e-10
        )

        if should_stop:
            node.is_leaf = True
            node.linear_model, node.model_features = self._build_linear_model(X, y)
            return node

        best_feature, best_value, best_sdr = self._find_best_split(X, y)

        if best_feature is None or best_sdr < self.min_std_reduction * np.std(y):
            node.is_leaf = True
            node.linear_model, node.model_features = self._build_linear_model(X, y)
            return node

        node.split_feature = best_feature
        node.split_value = best_value
        node.linear_model, node.model_features = self._build_linear_model(X, y)

        mask = X[:, best_feature] <= best_value
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        if len(y_left) > 0:
            node.left_child = self._build_tree(X_left, y_left, depth + 1, node)
        if len(y_right) > 0:
            node.right_child = self._build_tree(X_right, y_right, depth + 1, node)

        return node

    def _prune_tree(self, node: M5PNode, X: np.ndarray, y: np.ndarray) -> float:
        """Bottom-up pruning"""
        if node.is_leaf:
            if node.linear_model is not None and len(node.model_features) > 0:
                predictions = node.linear_model.predict(X[:, node.model_features])
            else:
                predictions = np.full(len(y), node.mean_target)

            if self.pruning_method == 'mae':
                error = np.mean(np.abs(y - predictions))
            else:
                error = np.mean((y - predictions) ** 2)

            n_params = len(node.model_features) + 1
            node.pruning_error = float(error * (1 + self.pruning_factor * n_params / max(len(y), 1)))
            return float(node.pruning_error)

        mask = X[:, node.split_feature] <= node.split_value
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        left_error = self._prune_tree(node.left_child, X_left, y_left) if node.left_child and len(y_left) > 0 else 0
        right_error = self._prune_tree(node.right_child, X_right, y_right) if node.right_child and len(y_right) > 0 else 0

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        if n_total > 0:
            children_error = (n_left * left_error + n_right * right_error) / n_total
        else:
            children_error = 0

        if node.linear_model is not None and len(node.model_features) > 0:
            predictions = node.linear_model.predict(X[:, node.model_features])
        else:
            predictions = np.full(len(y), node.mean_target)

        if self.pruning_method == 'mae':
            error = np.mean(np.abs(y - predictions))
        else:
            error = np.mean((y - predictions) ** 2)

        n_params = len(node.model_features) + 1
        node_error = error * (1 + self.pruning_factor * n_params / max(len(y), 1))

        if node_error <= children_error:
            node.is_leaf = True
            node.left_child = None
            node.right_child = None
            node.pruning_error = float(node_error)
            return float(node_error)
        else:
            node.pruning_error = children_error
            return children_error

    def _predict_single(self, x: np.ndarray, node: Optional[M5PNode]) -> float:
        """Predict without smoothing"""
        if node is None:
            return 0.0
            
        if node.is_leaf:
            if node.linear_model is not None and len(node.model_features) > 0:
                return float(node.linear_model.predict(x[node.model_features].reshape(1, -1))[0])
            else:
                return float(node.mean_target)

        if node.split_feature is not None and x[node.split_feature] <= node.split_value:
            return self._predict_single(x, node.left_child)
        else:
            return self._predict_single(x, node.right_child)

    def _predict_single_smoothed(self, x: np.ndarray, node: Optional[M5PNode]) -> float:
        """Predict with proper M5P smoothing (weight accumulation)"""
        if node is None:
            return 0.0
            
        # Traverse to leaf, collect path
        path = []
        current = node

        while current is not None and not current.is_leaf:
            path.append(current)
            if current.split_feature is not None and x[current.split_feature] <= current.split_value:
                current = current.left_child
            else:
                current = current.right_child

        # Handle case where current is None
        if current is None:
            return 0.0

        # Get leaf prediction
        if current.linear_model is not None and len(current.model_features) > 0:
            pred = float(current.linear_model.predict(x[current.model_features].reshape(1, -1))[0])
        else:
            pred = float(current.mean_target)

        # Initial weight = number of samples at leaf
        w = current.n_samples

        # Go up path, apply smoothing with accumulating weight
        for parent in reversed(path):
            if parent.linear_model is not None and len(parent.model_features) > 0:
                p_parent = float(parent.linear_model.predict(x[parent.model_features].reshape(1, -1))[0])
            else:
                p_parent = float(parent.mean_target)

            k = self.smoothing_constant
            pred = (w * pred + k * p_parent) / (w + k)
            w += k  # Weight accumulates

        return pred

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the M5P tree"""
        self.node_count = 0
        self.root = self._build_tree(X, y)

        if self.use_pruning:
            self._prune_tree(self.root, X, y)

        self._calculate_feature_importances(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.root is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            if self.use_smoothing:
                predictions[i] = self._predict_single_smoothed(x, self.root)
            else:
                predictions[i] = self._predict_single(x, self.root)

        return predictions

    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances (node-level SDR with data subsets)"""
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        def traverse(node: Optional[M5PNode], X_sub: np.ndarray, y_sub: np.ndarray):
            if node is None or node.is_leaf or len(y_sub) == 0:
                return

            # Calculate SDR using this node's data subset
            if node.split_feature is not None:
                mask = X_sub[:, node.split_feature] <= node.split_value
                sdr = self._calculate_sdr(y_sub, mask)
                importances[node.split_feature] += sdr * len(y_sub)

                # Recurse with data subsets
                if node.left_child:
                    traverse(node.left_child, X_sub[mask], y_sub[mask])
                if node.right_child:
                    traverse(node.right_child, X_sub[~mask], y_sub[~mask])
            else:
                # Handle case where split_feature is None
                if node.left_child:
                    traverse(node.left_child, X_sub, y_sub)
                if node.right_child:
                    traverse(node.right_child, X_sub, y_sub)

        if self.root is not None:
            traverse(self.root, X, y)

        # Normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()

        self.feature_importances_ = importances

    def get_tree_depth(self) -> int:
        """Get tree depth"""
        if self.root is None:
            return 0

        def get_depth(node: M5PNode) -> int:
            if node.is_leaf:
                return node.depth
            left = get_depth(node.left_child) if node.left_child else node.depth
            right = get_depth(node.right_child) if node.right_child else node.depth
            return max(left, right)

        return get_depth(self.root)

    def get_n_leaves(self) -> int:
        """Get number of leaves"""
        if self.root is None:
            return 0

        def count(node: M5PNode) -> int:
            if node.is_leaf:
                return 1
            c = 0
            if node.left_child:
                c += count(node.left_child)
            if node.right_child:
                c += count(node.right_child)
            return c

        return count(self.root)

    def export_rules(self, feature_names: List[str]) -> str:
        """
        Export decision rules with CORRECT feature names

        feature_names should be processed_feature_names_ (AFTER OneHot)
        """
        rules = []

        def traverse(node: Optional[M5PNode], conditions: List[str], depth: int = 0):
            indent = "  " * depth

            if node is None:
                return
                
            if node.is_leaf:
                cond_str = " AND ".join(conditions) if conditions else "ROOT"
                if node.linear_model is not None and len(node.model_features) > 0:
                    coef_str = ", ".join([f"{feature_names[f]}={node.linear_model.coef_[i]:.3f}"
                                           for i, f in enumerate(node.model_features)])
                    rules.append(f"{indent}IF {cond_str}:\n{indent}  → {coef_str} + {node.linear_model.intercept_:.3f}")
                else:
                    rules.append(f"{indent}IF {cond_str}:\n{indent}  → {node.mean_target:.3f} (constant)")
            else:
                if node.split_feature is not None:
                    feat_name = feature_names[node.split_feature]
                    left_cond = f"{feat_name} ≤ {node.split_value:.3f}"
                    right_cond = f"{feat_name} > {node.split_value:.3f}"

                    if node.left_child:
                        traverse(node.left_child, conditions + [left_cond], depth + 1)
                    if node.right_child:
                        traverse(node.right_child, conditions + [right_cond], depth + 1)

        if self.root is not None:
            traverse(self.root, [])
        return "\n".join(rules)

    def print_tree(self, node: Optional[M5PNode] = None, indent: str = ""):
        """Print tree structure"""
        if node is None:
            node = self.root
            print("\nM5P Model Tree (v2.2 - Feature Names Fixed):")
            print("=" * 70)

        if node is None:
            return
            
        if node.is_leaf:
            model_info = f"Ridge({len(node.model_features)} features)" if node.model_features else "Constant"
            print(f"{indent}[LEAF {node.node_id}] {model_info} (n={node.n_samples}, σ={node.std_dev:.3f})")
        else:
            if node.split_feature is not None:
                print(f"{indent}[NODE {node.node_id}] X[{node.split_feature}] ≤ {node.split_value:.3f} (n={node.n_samples})")

            if node.left_child:
                print(f"{indent}  ├─ Left:")
                self.print_tree(node.left_child, indent + "  │  ")

            if node.right_child:
                print(f"{indent}  └─ Right:")
                self.print_tree(node.right_child, indent + "     ")


class CompleteM5PRegressor(BaseEstimator, RegressorMixin):
    """
    Complete M5P Regressor - Version 2.2 (FINAL WITH FEATURE NAME FIX)

    CRITICAL FIX:
    - Stores processed_feature_names_ after OneHot encoding
    - export_rules() uses processed names (matching tree indices)
    - get_feature_importances_with_names() returns dict with correct mapping
    """

    def __init__(self,
                 unpruned: bool = False,
                 unsmoothed: bool = False,
                 min_samples_leaf: int = 4,
                 max_depth: int = 10,
                 min_std_reduction: float = 0.05,
                 smoothing_constant: int = 15,
                 pruning_factor: float = 2.0,
                 pruning_method: str = 'mse',
                 ridge_alpha: float = 0.01,
                 n_split_percentiles: int = 32,
                 handle_categorical: bool = True,
                 handle_missing: bool = True,
                 imputation_strategy: str = 'mean',
                 scale_features: bool = True,
                 random_state: Optional[int] = None,
                 verbose: bool = True):

        self.unpruned = unpruned
        self.unsmoothed = unsmoothed
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_std_reduction = min_std_reduction
        self.smoothing_constant = smoothing_constant
        self.pruning_factor = pruning_factor
        self.pruning_method = pruning_method
        self.ridge_alpha = ridge_alpha
        self.n_split_percentiles = n_split_percentiles
        self.handle_categorical = handle_categorical
        self.handle_missing = handle_missing
        self.imputation_strategy = imputation_strategy
        self.scale_features = scale_features
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.encoder_ = None

        self.cat_feature_names_ = None
        self.num_feature_names_ = None
        self.all_feature_names_ = None
        self.processed_feature_names_ = None  # NEW: Feature names AFTER preprocessing
        self.n_features_in_ = None

    def _log(self, message: str):
        if self.verbose:
            try:
                print(message)
            except UnicodeEncodeError:
                safe = message.encode("ascii", "ignore").decode("ascii")
                print(safe)

    def _identify_categorical_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        if not isinstance(X, pd.DataFrame):
            return [], list(range(X.shape[1]))

        cat_names = []
        num_names = []

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                cat_names.append(col)
            else:
                num_names.append(col)

        return cat_names, num_names

    def _build_processed_feature_names(self) -> List[str]:
        """
        CRITICAL NEW METHOD: Build feature names AFTER preprocessing

        Example:
        Input: num=['age', 'value'], cat=['category'] with values ['A', 'B', 'C']
        Output: ['age', 'value', 'category_A', 'category_B', 'category_C']
        """
        processed_names = list(self.num_feature_names_ or [])

        if (self.encoder_ is not None and 
            self.cat_feature_names_ is not None and 
            len(self.cat_feature_names_) > 0):
            # Get OneHot feature names
            for cat_col_idx, cat_col in enumerate(self.cat_feature_names_ or []):
                # encoder.categories_ contains categories for each feature
                if (hasattr(self.encoder_, 'categories_') and 
                    len(self.encoder_.categories_) > cat_col_idx):
                    categories = self.encoder_.categories_[cat_col_idx]

                    for category in categories:
                        processed_names.append(f"{cat_col}_{category}")

        return processed_names

    def _preprocess_fit(self, X: Union[np.ndarray, pd.DataFrame], 
                       y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        # 1) Đưa về DataFrame & nhận diện loại cột
        if isinstance(X, np.ndarray):
            cols = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)  # type: ignore
        else:
            X_df = X.copy()

        self.all_feature_names_ = list(X_df.columns)
        self.n_features_in_ = X_df.shape[1]
        self.cat_feature_names_, self.num_feature_names_ = self._identify_categorical_features(X_df)

        # 2) Chuẩn hoá numeric -> to_numeric
        if self.num_feature_names_:
            numeric_df = X_df[self.num_feature_names_]
            # Ensure we have a DataFrame, not a Series
            if not isinstance(numeric_df, pd.DataFrame):
                numeric_df = pd.DataFrame(numeric_df)
            X_df[self.num_feature_names_] = numeric_df.apply(pd.to_numeric, errors='coerce')

        # 3) Impute numeric trước, rồi scale numeric
        X_num = np.empty((len(X_df), 0))
        self.num_imputer_ = None
        if self.num_feature_names_:
            self.num_imputer_ = SimpleImputer(strategy=self.imputation_strategy)
            X_num = self.num_imputer_.fit_transform(X_df[self.num_feature_names_])
            if self.scale_features:
                self.scaler_ = StandardScaler()
                X_num = self.scaler_.fit_transform(X_num)

        # 4) Impute categorical (most_frequent hoặc placeholder), rồi OHE
        X_cat = np.empty((len(X_df), 0))
        if self.handle_categorical and self.cat_feature_names_:
            cat_df = X_df[self.cat_feature_names_]
            # Ensure we have a DataFrame, not a Series
            if not isinstance(cat_df, pd.DataFrame):
                cat_df = pd.DataFrame(cat_df)
            X_cat_raw = cat_df.astype('object').fillna('__MISSING__')
            try:
                self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                self.encoder_ = OneHotEncoder(sparse=False, handle_unknown='ignore')  # type: ignore
            X_cat = self.encoder_.fit_transform(X_cat_raw)
            X_cat = np.asarray(X_cat, dtype=np.float64)

        # 5) Ghép lại: [numeric(đã impute+scale), onehot(đã impute)]
        X_num_array = np.asarray(X_num, dtype=np.float64)
        X_cat_array = np.asarray(X_cat, dtype=np.float64)
        if X_cat_array.size > 0 and X_num_array.size > 0:
            X_processed = np.hstack([X_num_array, X_cat_array])
        elif X_num_array.size > 0:
            X_processed = X_num_array
        elif X_cat_array.size > 0:
            X_processed = X_cat_array
        else:
            X_processed = np.asarray(X_df, dtype=np.float64)

        # 6) Đặt tên feature sau preprocess
        self.processed_feature_names_ = self._build_processed_feature_names()
        self._log(f"📊 Processed features: {self.processed_feature_names_}")

        return np.asarray(X_processed, dtype=np.float64)

    def _preprocess_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        # 1) Đưa về DataFrame & đúng thứ tự cột
        if isinstance(X, np.ndarray):
            cols = list(self.all_feature_names_ or [f'feature_{i}' for i in range(X.shape[1])])
            X_df = pd.DataFrame(X, columns=cols)  # type: ignore
        else:
            X_df = X.copy()

        if self.all_feature_names_ is not None:
            X_selected = X_df[self.all_feature_names_]
            X_df = pd.DataFrame(X_selected) if not isinstance(X_selected, pd.DataFrame) else X_selected

        # 2) Kiểm tra thiếu cột
        miss_num = set(self.num_feature_names_ or []) - set(X_df.columns)
        miss_cat = set(self.cat_feature_names_ or []) - set(X_df.columns)
        if miss_num or miss_cat:
            msg = "Missing columns -> "
            if miss_num: msg += f"numeric: {sorted(miss_num)}"
            if miss_cat: msg += f", categorical: {sorted(miss_cat)}".lstrip(",")
            raise ValueError(msg)

        # 3) Numeric: to_numeric -> impute -> scale
        X_num = np.empty((len(X_df), 0))
        if self.num_feature_names_:
            # Ensure we have a DataFrame, not a Series
            numeric_df = X_df[self.num_feature_names_]
            if not isinstance(numeric_df, pd.DataFrame):
                numeric_df = pd.DataFrame(numeric_df)
            numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
            X_df[self.num_feature_names_] = numeric_df
            if self.num_imputer_ is not None:
                X_num = self.num_imputer_.transform(X_df[self.num_feature_names_])
            else:
                X_num = X_df[self.num_feature_names_].values
            if self.scale_features and self.scaler_ is not None:
                X_num = self.scaler_.transform(X_num)

        # 4) Categorical: fillna('__MISSING__') -> OHE.transform
        X_cat = np.empty((len(X_df), 0))
        if self.handle_categorical and self.cat_feature_names_ and self.encoder_ is not None:
            cat_df = X_df[self.cat_feature_names_]
            # Ensure we have a DataFrame, not a Series
            if not isinstance(cat_df, pd.DataFrame):
                cat_df = pd.DataFrame(cat_df)
            X_cat_raw = cat_df.astype('object').fillna('__MISSING__')
            X_cat = self.encoder_.transform(X_cat_raw)
            X_cat = np.asarray(X_cat, dtype=np.float64)

        # 5) Ghép
        X_num_array = np.asarray(X_num, dtype=np.float64)
        X_cat_array = np.asarray(X_cat, dtype=np.float64)
        if X_cat_array.size > 0 and X_num_array.size > 0:
            X_proc = np.hstack([X_num_array, X_cat_array])
        elif X_num_array.size > 0:
            X_proc = X_num_array
        elif X_cat_array.size > 0:
            X_proc = X_cat_array
        else:
            X_proc = np.asarray(X_df, dtype=np.float64)
        return np.asarray(X_proc, dtype=np.float64)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'CompleteM5PRegressor':

        if self.verbose:
            print("=" * 80)
            print("COMPLETE M5P REGRESSOR V2.2 - TRAINING (WITH FEATURE NAME FIX)")
            print("=" * 80)

        X_processed = self._preprocess_fit(X, y)

        if isinstance(y, pd.Series):
            y = np.asarray(y) if isinstance(y, pd.Series) else y

        self._log(f"\n📊 Initializing M5P:")
        self._log(f"   - Pruning: {not self.unpruned} ({self.pruning_method.upper()})")
        self._log(f"   - Smoothing: {not self.unsmoothed} (k={self.smoothing_constant})")
        self._log(f"   - Ridge α: {self.ridge_alpha}")

        self.model_ = M5PTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_std_reduction=self.min_std_reduction,
            use_pruning=not self.unpruned,
            use_smoothing=not self.unsmoothed,
            smoothing_constant=self.smoothing_constant,
            pruning_factor=self.pruning_factor,
            pruning_method=self.pruning_method,
            ridge_alpha=self.ridge_alpha,
            n_split_percentiles=self.n_split_percentiles,
            random_state=self.random_state
        )

        self._log(f"\n🚀 Training on {X_processed.shape[0]} samples...")
        y_array = np.asarray(y)
        self.model_.fit(X_processed, y_array)

        if self.verbose:
            self._log(f"\n📊 Tree Statistics:")
            self._log(f"   - Depth: {self.model_.get_tree_depth()}")
            self._log(f"   - Leaves: {self.model_.get_n_leaves()}")
            self._log(f"   - Total nodes: {self.model_.node_count}")

        self._log("\n✅ Training completed!")

        if self.verbose:
            print("=" * 80)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_processed = self._preprocess_transform(X)
        return self.model_.predict(X_processed)

    def score(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series],
              sample_weight: Optional[np.ndarray] = None) -> float:
        y_pred = self.predict(X)
        if isinstance(y, pd.Series):
            y = np.asarray(y) if isinstance(y, pd.Series) else y
        return float(r2_score(y, y_pred, sample_weight=sample_weight))

    def get_feature_importances(self) -> np.ndarray:
        """Get raw feature importances array (indexed by PROCESSED features)"""
        if self.model_ is None:
            raise ValueError("Model not trained yet.")
        importances = self.model_.feature_importances_
        return importances if importances is not None else np.array([])

    def get_feature_importances_with_names(self) -> Dict[str, float]:
        """
        NEW METHOD: Get feature importances WITH CORRECT NAMES

        Returns dict mapping processed_feature_names -> importance
        """
        if self.model_ is None or self.processed_feature_names_ is None:
            return {}
        arr = self.model_.feature_importances_
        if arr is None:
            arr = np.array([])
        names = self.processed_feature_names_
        k = min(len(arr), len(names))
        return {names[i]: float(arr[i]) for i in range(k)}

    def export_rules(self) -> str:
        """
        Export decision rules with CORRECT feature names (after preprocessing)

        FIXED: Now uses processed_feature_names_ (with OneHot names)
        """
        if self.model_ is None:
            raise ValueError("Model not trained yet.")
        if self.processed_feature_names_ is not None:
            return self.model_.export_rules(self.processed_feature_names_)
        else:
            return ""

    def print_feature_mapping(self):
        """Print original vs processed feature names"""
        print("\n" + "="*70)
        print("FEATURE NAME MAPPING")
        print("="*70)
        print("\nOriginal Features:")
        if self.all_feature_names_ is not None:
            for i, name in enumerate(self.all_feature_names_):
                ftype = "(categorical)" if self.cat_feature_names_ is not None and name in self.cat_feature_names_ else "(numeric)"
                print(f"  {i}: {name} {ftype}")

        print("\nProcessed Features (after OneHot + preprocessing):")
        if self.processed_feature_names_ is not None:
            for i, name in enumerate(self.processed_feature_names_):
                print(f"  {i}: {name}")

        print("\n" + "="*70)

    def print_tree(self):
        if self.model_ is None:
            raise ValueError("Model not trained yet.")
        self.model_.print_tree()

    def save(self, filepath: str):
        joblib.dump(self, filepath)
        if self.verbose:
            try:
                print(f"Model saved to {filepath}")
            except UnicodeEncodeError:
                print(f"Model saved to {filepath}".encode("ascii", "ignore").decode("ascii"))

    @staticmethod
    def load(filepath: str) -> 'CompleteM5PRegressor':
        model = joblib.load(filepath)
        try:
            print(f"Model loaded from {filepath}")
        except UnicodeEncodeError:
            print(f"Model loaded from {filepath}".encode("ascii", "ignore").decode("ascii"))
        return model

    def get_params(self, deep=True):
        return {
            'unpruned': self.unpruned,
            'unsmoothed': self.unsmoothed,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'min_std_reduction': self.min_std_reduction,
            'smoothing_constant': self.smoothing_constant,
            'pruning_factor': self.pruning_factor,
            'pruning_method': self.pruning_method,
            'ridge_alpha': self.ridge_alpha,
            'n_split_percentiles': self.n_split_percentiles,
            'handle_categorical': self.handle_categorical,
            'handle_missing': self.handle_missing,
            'imputation_strategy': self.imputation_strategy,
            'scale_features': self.scale_features,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Create a model instance with a simplified interface for the API
class M5PModelAPI:
    """Simplified API wrapper for the M5P model"""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        model_filename: str = "m5p_model.pkl",
        schema_filename: str = "m5p_schema.json",
    ):
        self.model: Optional[CompleteM5PRegressor] = None
        self.is_trained = False

        self.model_filename = model_filename
        self.schema_filename = schema_filename

        self._model_dirs = self._build_model_dirs(model_dir)
        self.model_dir = self._model_dirs[0]
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._model_path_cache: Optional[Path] = None
        self.feature_columns_: List[str] = []
        self.required_features_: List[str] = list(REQUIRED_FEATURES)
        self.target_column_: str = DEFAULT_TARGET_COLUMN

    @staticmethod
    def _dedupe_paths(paths: List[Path]) -> List[Path]:
        seen: set[Path] = set()
        unique: List[Path] = []
        for path in paths:
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(resolved)
        return unique

    def _build_model_dirs(self, explicit_dir: Optional[Union[str, Path]]) -> List[Path]:
        module_dir = Path(__file__).resolve().parent
        candidates: List[Path] = []
        if explicit_dir:
            candidates.append(Path(explicit_dir))
        candidates.extend([
            module_dir / "models",
            module_dir.parent / "models",
            module_dir,
            module_dir.parent,
            Path.cwd() / "models",
            Path.cwd(),
        ])
        return self._dedupe_paths(candidates)

    def _candidate_model_paths(self) -> List[Path]:
        module_dir = Path(__file__).resolve().parent
        candidates = [(directory / self.model_filename) for directory in self._model_dirs]
        candidates.extend([
            module_dir / self.model_filename,
            module_dir.parent / self.model_filename,
            Path.cwd() / self.model_filename,
        ])
        return self._dedupe_paths(candidates)

    def _candidate_schema_paths(self) -> List[Path]:
        module_dir = Path(__file__).resolve().parent
        candidates = [(directory / self.schema_filename) for directory in self._model_dirs]
        candidates.extend([
            module_dir / self.schema_filename,
            module_dir.parent / self.schema_filename,
            Path.cwd() / self.schema_filename,
        ])
        return self._dedupe_paths(candidates)

    def _find_existing_model_file(self) -> Optional[Path]:
        if self._model_path_cache and self._model_path_cache.exists():
            return self._model_path_cache
        for path in self._candidate_model_paths():
            if path.exists():
                self._model_path_cache = path
                return path
        return None

    def _load_schema_metadata(self) -> None:
        schema_path = self.get_schema_path()
        if not schema_path or not schema_path.exists():
            return
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return

        feature_columns = payload.get("feature_columns")
        if isinstance(feature_columns, list):
            self.feature_columns_ = feature_columns

        required_features = payload.get("required_features")
        if isinstance(required_features, list) and required_features:
            self.required_features_ = required_features

        target_column = payload.get("target_column")
        if isinstance(target_column, str) and target_column:
            self.target_column_ = target_column

    def _maybe_engineer_features(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, str]:
        base_required = set(REQUIRED_FEATURES_IN) | set(REQUIRED_FEATURES_OUT)
        required_present = all(col in df.columns for col in base_required)
        target_present = target_column in df.columns
        if required_present and target_present:
            return df, target_column

        base_columns = {"date", "cash_in"}
        if not base_columns.issubset(df.columns):
            return df, target_column

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.sort_values("date").reset_index(drop=True)

        work["cash_in_d0"] = pd.to_numeric(
            work.get("cash_in_d0", work["cash_in"]), errors="coerce"
        )
        cash_out_source = work.get("cash_out_d0", work.get("cash_out", 0.0))
        numeric_result = pd.to_numeric(cash_out_source, errors="coerce")
        # Ensure we're working with a Series
        if not isinstance(numeric_result, pd.Series):
            numeric_result = pd.Series(numeric_result)
        work["cash_out_d0"] = numeric_result.fillna(0.0)
        work["cash_net_d0"] = work["cash_in_d0"] - work["cash_out_d0"]

        # Cash-In lag features
        # Ensure we're working with Series before using pandas methods
        cash_in_d0_series = work["cash_in_d0"]
        if not isinstance(cash_in_d0_series, pd.Series):
            cash_in_d0_series = pd.Series(cash_in_d0_series, index=work.index)
        work["lag1_in"] = cash_in_d0_series.shift(1)
        work["lag7_in"] = cash_in_d0_series.shift(7)
        work["roll_mean_7_in"] = cash_in_d0_series.rolling(window=7, min_periods=1).mean()

        # Cash-Out lag features (for cash_out model)
        # Ensure we're working with Series before using pandas methods
        cash_out_d0_series = work["cash_out_d0"]
        if not isinstance(cash_out_d0_series, pd.Series):
            cash_out_d0_series = pd.Series(cash_out_d0_series, index=work.index)
        work["lag1_out"] = cash_out_d0_series.shift(1)
        work["lag7_out"] = cash_out_d0_series.shift(7)
        work["roll_mean_7_out"] = cash_out_d0_series.rolling(window=7, min_periods=1).mean()

        work["dow"] = work["date"].dt.weekday
        work["is_weekend"] = work["dow"].isin([5, 6]).astype(int)
        work["is_month_end"] = work["date"].dt.is_month_end.astype(int)
        work["is_payday"] = work["date"].dt.day.isin([15]).astype(int)
        work.loc[work["is_month_end"] == 1, "is_payday"] = 1

        if "channel" not in work.columns:
            work["channel"] = "DEFAULT"
        else:
            work["channel"] = work["channel"].astype(str)

        # Backfill cash_in lag features
        for col in ("lag1_in", "lag7_in", "roll_mean_7_in"):
            work[col] = pd.to_numeric(work[col], errors="coerce")
            series = work[col] if isinstance(work[col], pd.Series) else pd.Series(work[col])
            work[col] = series.bfill().fillna(work["cash_in_d0"]).fillna(0.0)

        # Backfill cash_out lag features
        for col in ("lag1_out", "lag7_out", "roll_mean_7_out"):
            work[col] = pd.to_numeric(work[col], errors="coerce")
            series = work[col] if isinstance(work[col], pd.Series) else pd.Series(work[col])
            work[col] = series.bfill().fillna(work["cash_out_d0"]).fillna(0.0)

        # Create target column based on its name
        if target_column not in work.columns:
            # Determine which base column to use for target
            if "out" in target_column.lower() or "cash_out" in target_column.lower():
                base_col = "cash_out_d0"
            else:
                base_col = "cash_in_d0"

            # Ensure we're working with a Series
            base_data = work[base_col]
            if not isinstance(base_data, pd.Series):
                base_series = pd.Series(base_data, index=work.index if hasattr(work, 'index') else None)
            else:
                base_series = base_data
            numeric_series = pd.to_numeric(base_series, errors="coerce")
            # Ensure we're working with a Series before using pandas methods
            if not isinstance(numeric_series, pd.Series):
                numeric_series = pd.Series(numeric_series, index=base_series.index if hasattr(base_series, 'index') else work.index)

            # Determine target type: next_day, h7_sum, or next_month_sum
            if "h7" in target_column.lower():
                # Sum of next 7 days
                rolled_sum = numeric_series.rolling(window=7, min_periods=1).sum()
                # Ensure we're working with a Series before using pandas methods
                if not isinstance(rolled_sum, pd.Series):
                    rolled_sum = pd.Series(rolled_sum, index=numeric_series.index)
                work[target_column] = rolled_sum.shift(-7)
            elif "next_month" in target_column.lower():
                # Sum of next calendar month
                # This is an approximation - sum next 30 days
                rolled_sum = numeric_series.rolling(window=30, min_periods=1).sum()
                # Ensure we're working with a Series before using pandas methods
                if not isinstance(rolled_sum, pd.Series):
                    rolled_sum = pd.Series(rolled_sum, index=numeric_series.index)
                work[target_column] = rolled_sum.shift(-30)
            else:
                # Next day (default)
                work[target_column] = numeric_series.shift(-1)

        work = work.dropna(subset=list(base_required) + [target_column])
        return work, target_column

    def _ensure_model_loaded(self) -> Path:
        path = self._find_existing_model_file()
        if path is None:
            raise FileNotFoundError("m5p_model.pkl not found on disk")
        self.model = CompleteM5PRegressor.load(str(path))
        self.is_trained = True
        self.model_dir = path.parent
        self._model_path_cache = path
        self._load_schema_metadata()
        return path

    def has_persisted_model(self) -> bool:
        """
        Check whether a serialized model artifact is available on disk.
        """
        try:
            return self._find_existing_model_file() is not None
        except Exception:
            return False

    def load_from_disk(self) -> Path:
        """
        Force-load the persisted model, raising FileNotFoundError if unavailable.
        """
        return self._ensure_model_loaded()

    def get_schema_path(self) -> Optional[Path]:
        for path in self._candidate_schema_paths():
            if path.exists():
                return path
        return None

    def train(self, data_file_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the M5P model and persist artifacts to disk.
        Returns training metrics.
        """
        df = pd.read_csv(data_file_path).copy()

        if target_column is None:
            target_column = DEFAULT_TARGET_COLUMN

        df, inferred_target = self._maybe_engineer_features(df, target_column)
        target_column = inferred_target or target_column
        if target_column not in df.columns and DEFAULT_TARGET_COLUMN in df.columns:
            target_column = DEFAULT_TARGET_COLUMN

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' could not be derived or found in dataset.")

        if "cash_net_d0" not in df.columns and {"cash_in_d0", "cash_out_d0"}.issubset(df.columns):
            cash_in_series = pd.to_numeric(df["cash_in_d0"], errors="coerce")
            cash_out_series = pd.to_numeric(df["cash_out_d0"], errors="coerce")
            # Ensure both are Series before subtraction
            if not isinstance(cash_in_series, pd.Series):
                cash_in_series = pd.Series(cash_in_series, index=df.index)
            if not isinstance(cash_out_series, pd.Series):
                cash_out_series = pd.Series(cash_out_series, index=df.index)
            df["cash_net_d0"] = cash_in_series - cash_out_series

        # Select appropriate feature set based on target column
        # Always use the correct feature set for the target, ignore self.required_features_
        if "out" in target_column.lower() or "cash_out" in target_column.lower():
            # Cash out models use lag_out features
            required_features = list(REQUIRED_FEATURES_OUT)
        else:
            # Cash in models use lag_in features
            required_features = list(REQUIRED_FEATURES_IN)

        missing_required = [col for col in required_features if col not in df.columns]
        if missing_required:
            raise ValueError(
                "Dataset is missing required feature column(s): "
                + ", ".join(sorted(missing_required))
            )

        for bool_col in ("is_weekend", "is_month_end", "is_payday"):
            df[bool_col] = df[bool_col].astype(int)

        if "channel" in df.columns:
            df["channel"] = df["channel"].astype(str)
        else:
            df["channel"] = "DEFAULT"

        df = df.dropna(subset=required_features + [target_column])
        if df.empty:
            raise ValueError("Dataset does not contain sufficient rows after preprocessing.")

        ordered_features: List[str] = list(required_features)

        X = df[ordered_features]
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(pd.to_numeric(df[target_column], errors="coerce"))

        if np.isnan(y).any():
            raise ValueError("Target column contains non-numeric values after coercion.")

        self.model = CompleteM5PRegressor(
            min_samples_leaf=4,
            max_depth=10,
            min_std_reduction=0.05,
            smoothing_constant=15,
            pruning_factor=2.0,
            ridge_alpha=0.01,
            random_state=42,
            verbose=True
        )

        self.model.fit(X, y)
        self.is_trained = True

        self.feature_columns_ = ordered_features
        # Store the correct feature set used for this model
        self.required_features_ = required_features
        self.target_column_ = target_column

        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))

        model_path = self.model_dir / self.model_filename
        self.model.save(str(model_path))
        self._model_path_cache = model_path

        schema = {
            "target_column": target_column,
            "feature_columns": ordered_features,
            "required_features": required_features,  # Use the correct feature set
            "all_feature_names": self.model.all_feature_names_,
            "categorical_features": list(self.model.cat_feature_names_ or []),
            "numeric_features": list(self.model.num_feature_names_ or []),
        }
        schema_path = self.model_dir / self.schema_filename
        with open(schema_path, "w", encoding="utf-8") as fp:
            json.dump(schema, fp, ensure_ascii=False, indent=2)
        self._load_schema_metadata()

        imps = self.model.get_feature_importances_with_names()
        top_imps = dict(sorted(imps.items(), key=lambda kv: kv[1], reverse=True)[:10])

        return {
            "status": "success",
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "model_path": str(model_path),
            "schema_path": str(schema_path),
            "top_feature_importances": top_imps,
        }

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Make a prediction using the trained model.
        """
        if not self.is_trained or self.model is None:
            try:
                self._ensure_model_loaded()
            except FileNotFoundError as exc:
                raise ValueError("Model not trained yet. Call train() first.") from exc

        model = self.model
        if model is None:
            raise ValueError("Model not available. Train or load before predicting.")

        currency = str(features.get("currency", DEFAULT_CURRENCY)).upper()
        if currency != DEFAULT_CURRENCY:
            raise ValueError("Prediction currently supports currency VND only.")

        if "currency" in features:
            features = dict(features)
            features.pop("currency", None)

        df = pd.DataFrame([features]).copy()

        if "cash_net_d0" not in df.columns and {"cash_in_d0", "cash_out_d0"}.issubset(df.columns):
            cash_in_series = pd.to_numeric(df["cash_in_d0"], errors="coerce")
            cash_out_series = pd.to_numeric(df["cash_out_d0"], errors="coerce")
            # Ensure both are Series before subtraction
            if not isinstance(cash_in_series, pd.Series):
                cash_in_series = pd.Series(cash_in_series, index=df.index)
            if not isinstance(cash_out_series, pd.Series):
                cash_out_series = pd.Series(cash_out_series, index=df.index)
            df["cash_net_d0"] = cash_in_series - cash_out_series

        required_features = list(self.required_features_ or REQUIRED_FEATURES_IN)
        missing_required = [col for col in required_features if col not in df.columns]
        if missing_required:
            raise ValueError(
                "Missing required feature(s) for prediction: "
                + ", ".join(sorted(missing_required))
            )

        for bool_col in ("is_weekend", "is_month_end", "is_payday"):
            df[bool_col] = df[bool_col].astype(int)

        if "channel" in df.columns:
            df["channel"] = df["channel"].astype(str)
        else:
            df["channel"] = "DEFAULT"

        all_features = self.feature_columns_ or required_features
        extra_columns = set(df.columns) - set(all_features)
        if extra_columns:
            warnings.warn(f"Ignoring unknown columns: {sorted(extra_columns)}")

        for col in all_features:
            if col not in df.columns:
                df[col] = np.nan

        df = df[all_features]
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if model.num_feature_names_:
            # Ensure we're working with a DataFrame before applying
            subset = df[model.num_feature_names_]
            if not isinstance(subset, pd.DataFrame):
                subset = pd.DataFrame(subset)
            df[model.num_feature_names_] = subset.apply(pd.to_numeric, errors='coerce')

        prediction = model.predict(df)
        return float(prediction[0])


# Create the model instance that app.py is trying to import
m5p_model = M5PModelAPI()
