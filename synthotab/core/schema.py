"""Table and Column Schema Inference and Constraint Management for SynthoTab."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class ColumnType(str, enum.Enum):
    """Categorization of tabular column data types."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ID = "id"


@dataclass
class ColumnMetadata:
    """Metadata describing empirical distribution and boundaries of a single column."""
    name: str
    col_type: ColumnType
    dtype: str
    is_nullable: bool = False
    null_rate: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    unique_count: int = 0
    categories: Optional[List[Any]] = None
    category_probs: Optional[Dict[Any, float]] = None
    is_primary_key: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metadata to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "col_type": self.col_type.value,
            "dtype": str(self.dtype),
            "is_nullable": self.is_nullable,
            "null_rate": float(self.null_rate),
            "min_val": float(self.min_val) if self.min_val is not None else None,
            "max_val": float(self.max_val) if self.max_val is not None else None,
            "unique_count": int(self.unique_count),
            "categories": [str(c) for c in self.categories] if self.categories is not None else None,
            "is_primary_key": self.is_primary_key,
        }


@dataclass
class TableSchema:
    """Inferred schema representing columns, relationships, and statistical boundaries."""
    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)
    primary_key: Optional[str] = None
    row_count: int = 0

    @classmethod
    def infer(
        cls,
        df: pd.DataFrame,
        primary_key: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
        id_columns: Optional[List[str]] = None,
    ) -> TableSchema:
        """Infers column types, boundaries, null rates, and category domains from a DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected input to be a pandas DataFrame")

        schema = cls(primary_key=primary_key, row_count=len(df))
        categorical_set = set(categorical_columns or [])
        continuous_set = set(continuous_columns or [])
        id_set = set(id_columns or [])

        for col in df.columns:
            series = df[col]
            n_rows = len(series)
            null_count = int(series.isna().sum())
            null_rate = float(null_count / n_rows) if n_rows > 0 else 0.0
            is_nullable = bool(null_count > 0)
            valid_series = series.dropna()
            n_unique = int(valid_series.nunique())
            is_pk = bool(col == primary_key)

            # Determine column type
            if col in id_set or is_pk:
                col_type = ColumnType.ID
            elif col in categorical_set:
                col_type = ColumnType.CATEGORICAL
            elif col in continuous_set:
                col_type = ColumnType.CONTINUOUS
            elif pd.api.types.is_bool_dtype(series.dtype):
                col_type = ColumnType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(series.dtype):
                col_type = ColumnType.DATETIME
            elif pd.api.types.is_numeric_dtype(series.dtype):
                # Distinguish integer vs continuous
                is_int_compat = False
                if len(valid_series) > 0:
                    try:
                        is_int_compat = np.all(np.equal(np.mod(valid_series.to_numpy(), 1), 0))
                    except Exception:
                        is_int_compat = False

                if is_int_compat and n_unique <= 20 and n_rows >= 100:
                    # Low-cardinality discrete integers often represent categories
                    col_type = ColumnType.CATEGORICAL
                elif is_int_compat:
                    col_type = ColumnType.INTEGER
                else:
                    col_type = ColumnType.CONTINUOUS
            else:
                # String, object or mixed
                if n_unique == n_rows and n_rows > 10:
                    col_type = ColumnType.ID
                else:
                    col_type = ColumnType.CATEGORICAL

            min_val = None
            max_val = None
            categories = None
            category_probs = None

            if col_type in (ColumnType.CONTINUOUS, ColumnType.INTEGER) and len(valid_series) > 0:
                min_val = float(valid_series.min())
                max_val = float(valid_series.max())
            elif col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN) and len(valid_series) > 0:
                freq = valid_series.value_counts(normalize=True)
                categories = list(freq.index)
                category_probs = {k: float(v) for k, v in freq.items()}

            meta = ColumnMetadata(
                name=str(col),
                col_type=col_type,
                dtype=str(series.dtype),
                is_nullable=is_nullable,
                null_rate=null_rate,
                min_val=min_val,
                max_val=max_val,
                unique_count=n_unique,
                categories=categories,
                category_probs=category_probs,
                is_primary_key=is_pk,
            )
            schema.columns[str(col)] = meta

        return schema

    def enforce_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforces type bounds, non-negative limits, integer rounding, and category validity."""
        df_out = df.copy()

        for col, meta in self.columns.items():
            if col not in df_out.columns:
                continue

            if meta.col_type == ColumnType.INTEGER:
                # Round to nearest integer and clip between min and max
                val = pd.to_numeric(df_out[col], errors="coerce")
                if meta.min_val is not None and meta.max_val is not None:
                    val = val.clip(lower=meta.min_val, upper=meta.max_val)
                df_out[col] = val.round().astype("Int64" if meta.is_nullable else "int64")

            elif meta.col_type == ColumnType.CONTINUOUS:
                val = pd.to_numeric(df_out[col], errors="coerce")
                if meta.min_val is not None and meta.max_val is not None:
                    # Clip to natural observed range
                    val = val.clip(lower=meta.min_val, upper=meta.max_val)
                df_out[col] = val

            elif meta.col_type == ColumnType.CATEGORICAL:
                if meta.categories is not None and len(meta.categories) > 0:
                    valid_set = set(meta.categories)
                    # Replace out-of-domain categories with highest-frequency fallback
                    fallback = meta.categories[0]
                    df_out[col] = df_out[col].apply(lambda x: x if x in valid_set or pd.isna(x) else fallback)

            elif meta.col_type == ColumnType.BOOLEAN:
                df_out[col] = df_out[col].astype(bool)

            elif meta.col_type == ColumnType.ID:
                # Generate unique sequential IDs if missing or duplicate
                n = len(df_out)
                df_out[col] = [f"SYNTH_{i+1:06d}" for i in range(n)]

        return df_out

    @property
    def continuous_columns(self) -> List[str]:
        """Returns names of continuous columns."""
        return [c for c, m in self.columns.items() if m.col_type == ColumnType.CONTINUOUS]

    @property
    def integer_columns(self) -> List[str]:
        """Returns names of integer columns."""
        return [c for c, m in self.columns.items() if m.col_type == ColumnType.INTEGER]

    @property
    def numerical_columns(self) -> List[str]:
        """Returns all numerical (continuous and integer) columns."""
        return [c for c, m in self.columns.items() if m.col_type in (ColumnType.CONTINUOUS, ColumnType.INTEGER)]

    @property
    def categorical_columns(self) -> List[str]:
        """Returns names of categorical and boolean columns."""
        return [c for c, m in self.columns.items() if m.col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)]
