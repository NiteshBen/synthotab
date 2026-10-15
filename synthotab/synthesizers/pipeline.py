"""Unified High-Level Pipeline for SynthoTab Synthetic Data Generation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from synthotab.core.base import BaseSynthesizer
from synthotab.core.schema import ColumnType, TableSchema
from synthotab.synthesizers.copula import GaussianCopulaSynthesizer
from synthotab.synthesizers.conditional import CategoricalSynthesizer
from synthotab.privacy.differential import (
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyAccountant,
)
from synthotab.privacy.audit import PrivacyAuditor, PrivacyAuditResult
from synthotab.metrics.fidelity import FidelityEvaluator, FidelityResult


class SynthoTab(BaseSynthesizer):
    """Production-grade tabular synthetic data generator with differential privacy and auditing.

    Combines Gaussian Copulas for continuous/integer features, hierarchical conditional
    sampling for discrete/categorical features, and mathematically calibrated
    differential privacy noise mechanisms.
    """

    def __init__(
        self,
        epsilon: Optional[float] = None,
        delta: float = 1e-5,
        random_state: Optional[int] = 42,
        enforce_constraints: bool = True,
    ) -> None:
        super().__init__(random_state=random_state)
        self.epsilon = float(epsilon) if epsilon is not None else None
        self.delta = float(delta)
        self.enforce_constraints = enforce_constraints

        self.accountant = PrivacyAccountant(target_epsilon=self.epsilon, target_delta=self.delta)
        self.copula_synthesizer: Optional[GaussianCopulaSynthesizer] = None
        self.categorical_synthesizer: Optional[CategoricalSynthesizer] = None
        self.column_order: List[str] = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame) -> SynthoTab:
        """Fits multi-type synthesizers on real dataset and initializes privacy accounting."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected input df to be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        self.schema = TableSchema.infer(df)
        self.column_order = list(df.columns)

        num_cols = self.schema.numerical_columns
        cat_cols = self.schema.categorical_columns

        # Allocate privacy budgets across modules if differential privacy is active
        copula_eps = None
        cat_eps = None
        if self.epsilon is not None:
            # Split budget: 60% for continuous correlation & covariance, 40% for categorical histograms
            budget_share = 0.6 if cat_cols and num_cols else 1.0
            copula_eps = self.epsilon * budget_share if num_cols else 0.0
            cat_eps = self.epsilon * (1.0 - budget_share) if cat_cols else 0.0

            if copula_eps > 0:
                self.accountant.allocate("copula_covariance", copula_eps, delta=self.delta)
            if cat_eps > 0:
                self.accountant.allocate("categorical_marginals", cat_eps, delta=0.0)

        # 1. Fit Gaussian Copula for continuous and integer features
        if num_cols:
            self.copula_synthesizer = GaussianCopulaSynthesizer(random_state=self.random_state)
            self.copula_synthesizer.fit(df[num_cols])

            # Perturb covariance matrix if DP is enabled
            if copula_eps and copula_eps > 0:
                dp_gauss = GaussianMechanism(
                    epsilon=copula_eps,
                    delta=self.delta,
                    sensitivity=2.0 / len(df),
                    random_state=self.random_state,
                )
                noisy_corr = dp_gauss.perturb(self.copula_synthesizer.correlation_matrix)
                # Re-regularize to valid PSD correlation matrix
                self.copula_synthesizer.correlation_matrix = self.copula_synthesizer._regularize_correlation_matrix(noisy_corr)

        # 2. Fit Categorical Synthesizer for discrete, boolean and string features
        if cat_cols:
            self.categorical_synthesizer = CategoricalSynthesizer(random_state=self.random_state)
            self.categorical_synthesizer.fit(df[cat_cols])

            # Perturb category probabilities if DP is enabled
            if cat_eps and cat_eps > 0:
                dp_laplace = LaplaceMechanism(
                    epsilon=cat_eps / max(1, len(cat_cols)),
                    sensitivity=1.0,
                    random_state=self.random_state,
                )
                for col in self.categorical_synthesizer.columns:
                    cats, probs = self.categorical_synthesizer.marginals[col]
                    noisy_probs = dp_laplace.perturb_histogram(probs * len(df))
                    self.categorical_synthesizer.marginals[col] = (cats, noisy_probs)

        self._is_fitted = True
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generates synthetic rows matching original schema and column ordering."""
        if not self._is_fitted:
            raise RuntimeError("SynthoTab must be fitted before calling sample()")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be a positive integer, got {n_rows}")

        parts: List[pd.DataFrame] = []

        if self.copula_synthesizer is not None:
            num_df = self.copula_synthesizer.sample(n_rows)
            parts.append(num_df)

        if self.categorical_synthesizer is not None:
            cat_df = self.categorical_synthesizer.sample(n_rows)
            parts.append(cat_df)

        # Handle ID / Primary Key columns
        id_cols = [c for c, m in self.schema.columns.items() if m.col_type == ColumnType.ID]
        if id_cols:
            id_data = {c: [f"SYNTH_{i+1:06d}" for i in range(n_rows)] for c in id_cols}
            id_df = pd.DataFrame(id_data)
            parts.append(id_df)

        if not parts:
            return pd.DataFrame()

        # Combine all parts along columns
        combined_df = pd.concat(parts, axis=1)

        # Restore original column ordering
        existing_cols = [c for c in self.column_order if c in combined_df.columns]
        combined_df = combined_df[existing_cols]

        # Enforce domain constraints, min/max bounds and integer rounding
        if self.enforce_constraints and self.schema is not None:
            combined_df = self.schema.enforce_constraints(combined_df)

        return combined_df

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> FidelityResult:
        """Runs statistical fidelity evaluation comparing real vs synthetic data."""
        evaluator = FidelityEvaluator()
        return evaluator.evaluate(real_df, synthetic_df)

    def audit_privacy(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> PrivacyAuditResult:
        """Audits synthetic data against real training data for information leakage and memorization."""
        auditor = PrivacyAuditor(random_state=self.random_state)
        return auditor.audit(real_df, synthetic_df)

    def generate_report(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_path: str = "synthotab_report.html",
        title: str = "SynthoTab Synthetic Data Quality & Privacy Report",
    ) -> str:
        """Generates an interactive, standalone HTML quality & privacy report."""
        from synthotab.reporting.html_report import generate_html_report

        fidelity = self.evaluate(real_df, synthetic_df)
        privacy = self.audit_privacy(real_df, synthetic_df)
        accountant_summary = self.accountant.summary() if self.epsilon is not None else None

        return generate_html_report(
            real_df=real_df,
            synthetic_df=synthetic_df,
            fidelity=fidelity,
            privacy=privacy,
            privacy_summary=accountant_summary,
            output_path=output_path,
            title=title,
        )
