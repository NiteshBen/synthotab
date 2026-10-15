"""Unit tests for FidelityEvaluator."""

import pytest
import numpy as np
import pandas as pd
from synthotab.metrics.fidelity import FidelityEvaluator


def test_fidelity_identical_data():
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["A", "B", "A", "B", "A"],
    })

    evaluator = FidelityEvaluator()
    result = evaluator.evaluate(df, df.copy())

    # Identical datasets should score near 100%
    assert result.overall_score >= 95.0
    assert result.correlation_mae == 0.0
    assert "num" in result.column_fidelities
    assert "cat" in result.column_fidelities


def test_fidelity_disjoint_data():
    real_df = pd.DataFrame({
        "num": [0.0, 1.0, 2.0, 3.0],
        "cat": ["A", "A", "A", "A"],
    })
    disjoint_df = pd.DataFrame({
        "num": [1000.0, 1001.0, 1002.0, 1003.0],
        "cat": ["Z", "Z", "Z", "Z"],
    })

    evaluator = FidelityEvaluator()
    result = evaluator.evaluate(real_df, disjoint_df)

    assert result.overall_score < 40.0


def test_fidelity_empty_raises():
    evaluator = FidelityEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(pd.DataFrame(), pd.DataFrame())
