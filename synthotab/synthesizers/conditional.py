"""Conditional and Marginal Categorical Synthesizer for discrete tabular data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from synthotab.core.base import BaseSynthesizer
from synthotab.core.schema import ColumnType, TableSchema


class CategoricalSynthesizer(BaseSynthesizer):
    """Models discrete, categorical, and boolean column distributions with conditional dependencies.

    Supports hierarchical conditional probability modeling (P(C_k | C_{parent})) with
    Laplace smoothing, preserving co-occurrence frequencies between discrete features.
    """

    def __init__(
        self,
        random_state: Optional[int] = 42,
        smoothing: float = 1e-3,
    ) -> None:
        super().__init__(random_state=random_state)
        self.smoothing = smoothing
        self.columns: List[str] = []
        self.marginals: Dict[str, Tuple[List[Any], np.ndarray]] = {}
        self.conditional_tables: Dict[str, Dict[Any, Tuple[List[Any], np.ndarray]]] = {}
        self.parent_map: Dict[str, Optional[str]] = {}
        self._rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame) -> CategoricalSynthesizer:
        """Estimates marginal and conditional probability tables from categorical columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected input df to be a pandas DataFrame")
        if df.empty:
            raise ValueError("Cannot fit synthesizer on an empty DataFrame")

        self.schema = TableSchema.infer(df)
        cat_cols = self.schema.categorical_columns

        if not cat_cols:
            raise ValueError("No categorical or boolean columns found in dataset")

        self.columns = cat_cols

        # 1. Fit marginal probabilities for all columns
        for col in self.columns:
            series = df[col].dropna()
            if len(series) == 0:
                categories = ["unknown"]
                probs = np.array([1.0])
            else:
                counts = series.value_counts(normalize=False)
                categories = list(counts.index)
                raw_counts = counts.to_numpy(dtype=np.float64)
                # Apply Laplace smoothing
                smoothed_counts = raw_counts + self.smoothing
                probs = smoothed_counts / np.sum(smoothed_counts)

            self.marginals[col] = (categories, probs)

        # 2. Establish hierarchical conditional dependencies
        # Primary column is the one with highest entropy / cardinality
        primary_col = self.columns[0]
        self.parent_map[primary_col] = None

        for idx in range(1, len(self.columns)):
            child_col = self.columns[idx]
            parent_col = self.columns[idx - 1]
            self.parent_map[child_col] = parent_col

            # Learn P(child | parent)
            cond_dict: Dict[Any, Tuple[List[Any], np.ndarray]] = {}
            child_cats, _ = self.marginals[child_col]

            grouped = df.groupby(parent_col)[child_col]
            for parent_val, group_series in grouped:
                valid_group = group_series.dropna()
                if len(valid_group) == 0:
                    cond_dict[parent_val] = self.marginals[child_col]
                    continue

                cat_counts = np.zeros(len(child_cats), dtype=np.float64)
                for c_idx, cat in enumerate(child_cats):
                    cat_counts[c_idx] = np.sum(valid_group == cat)

                smoothed = cat_counts + self.smoothing
                c_probs = smoothed / np.sum(smoothed)
                cond_dict[parent_val] = (child_cats, c_probs)

            self.conditional_tables[child_col] = cond_dict

        self._is_fitted = True
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Samples categorical rows adhering to hierarchical conditional dependencies."""
        if not self._is_fitted:
            raise RuntimeError("Synthesizer must be fitted before calling sample()")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be a positive integer, got {n_rows}")

        sampled_data: Dict[str, List[Any]] = {col: [] for col in self.columns}

        # Step 1: Sample primary column from its marginal distribution
        primary_col = self.columns[0]
        p_cats, p_probs = self.marginals[primary_col]
        primary_samples = self._rng.choice(p_cats, size=n_rows, p=p_probs)
        sampled_data[primary_col] = list(primary_samples)

        # Step 2: Sample child columns conditioned on parents
        for col_idx in range(1, len(self.columns)):
            col = self.columns[col_idx]
            parent_col = self.parent_map[col]
            parent_values = sampled_data[parent_col]
            cond_table = self.conditional_tables.get(col, {})
            fallback_cats, fallback_probs = self.marginals[col]

            col_samples: List[Any] = []
            for p_val in parent_values:
                if p_val in cond_table:
                    c_cats, c_probs = cond_table[p_val]
                    sampled_val = self._rng.choice(c_cats, p=c_probs)
                else:
                    sampled_val = self._rng.choice(fallback_cats, p=fallback_probs)
                col_samples.append(sampled_val)

            sampled_data[col] = col_samples

        df_out = pd.DataFrame(sampled_data)
        # Restore boolean types if original column was boolean
        for col in self.columns:
            meta = self.schema.columns.get(col)
            if meta and meta.col_type == ColumnType.BOOLEAN:
                df_out[col] = df_out[col].astype(bool)

        return df_out
