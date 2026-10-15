"""Benchmark datasets for SynthoTab."""

from synthotab.datasets.loaders import (
    load_healthcare_patients,
    load_financial_churn,
)

__all__ = [
    "load_healthcare_patients",
    "load_financial_churn",
]
