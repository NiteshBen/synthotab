"""Unit tests for TableSchema and ColumnMetadata."""

import pytest
import numpy as np
import pandas as pd
from synthotab.core.schema import ColumnType, TableSchema, ColumnMetadata


def test_schema_inference_basic():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50],
        "salary": [50000.5, 60000.0, 75000.2, 80000.0, 95000.8, 120000.0],
        "department": ["Eng", "Sales", "Eng", "HR", "Sales", "Eng"],
        "is_active": [True, True, False, True, False, True],
    })

    schema = TableSchema.infer(df)
    assert schema.row_count == 6
    assert schema.columns["age"].col_type == ColumnType.INTEGER
    assert schema.columns["salary"].col_type == ColumnType.CONTINUOUS
    assert schema.columns["department"].col_type == ColumnType.CATEGORICAL
    assert schema.columns["is_active"].col_type == ColumnType.BOOLEAN

    assert schema.columns["age"].min_val == 25.0
    assert schema.columns["age"].max_val == 50.0
    assert schema.columns["salary"].min_val == 50000.5


def test_schema_enforce_constraints():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "score": [10.5, 20.2, 30.8],
        "category": ["A", "B", "A"],
    })
    schema = TableSchema.infer(df)

    synthetic_df = pd.DataFrame({
        "age": [15.2, 29.8, 45.9],  # Out of range and floating point
        "score": [5.0, 22.0, 35.0],   # Out of range
        "category": ["A", "INVALID", "B"],  # Out of domain
    })

    cleaned = schema.enforce_constraints(synthetic_df)

    # Integer ages should be rounded and clipped
    assert cleaned["age"].tolist() == [20, 30, 40]
    # Continuous scores clipped
    assert cleaned["score"].iloc[0] == 10.5
    assert cleaned["score"].iloc[2] == 30.8
    # Invalid category replaced by fallback
    assert cleaned["category"].iloc[1] in ["A", "B"]


def test_schema_null_detection():
    df = pd.DataFrame({
        "val": [1.0, np.nan, 3.0, np.nan],
    })
    schema = TableSchema.infer(df)
    assert schema.columns["val"].is_nullable is True
    assert schema.columns["val"].null_rate == 0.5


def test_schema_invalid_input():
    with pytest.raises(TypeError):
        TableSchema.infer("not a dataframe")  # type: ignore
