"""Unit tests for built-in dataset loaders."""

import pytest
import pandas as pd
from synthotab.datasets.loaders import load_healthcare_patients, load_financial_churn


def test_load_healthcare_patients():
    df = load_healthcare_patients(n_samples=100, random_state=42)
    assert len(df) == 100
    expected_cols = [
        "patient_id",
        "age",
        "gender",
        "bmi",
        "systolic_bp",
        "cholesterol",
        "glucose",
        "smoker",
        "diagnosis",
        "treatment_cost",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify medical bounds
    assert df["age"].min() >= 18
    assert df["bmi"].min() >= 18.0
    assert df["systolic_bp"].min() >= 90.0


def test_load_financial_churn():
    df = load_financial_churn(n_samples=100, random_state=42)
    assert len(df) == 100
    expected_cols = [
        "customer_id",
        "credit_score",
        "geography",
        "gender",
        "age",
        "tenure",
        "balance",
        "num_of_products",
        "has_cr_card",
        "is_active_member",
        "estimated_salary",
        "exited",
    ]
    for col in expected_cols:
        assert col in df.columns

    assert df["credit_score"].min() >= 350
    assert set(df["exited"].unique()).issubset({0, 1})
