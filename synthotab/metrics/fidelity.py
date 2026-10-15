"""Statistical Fidelity and Dependency Evaluator for synthetic tabular data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


@dataclass
class ColumnFidelity:
    """Fidelity metric for a single column."""
    name: str
    column_type: str
    fidelity_score: float  # 0.0 to 1.0
    metric_name: str
    metric_value: float
    p_value: Optional[float] = None
    real_mean: Optional[float] = None
    synth_mean: Optional[float] = None
    real_std: Optional[float] = None
    synth_std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts column metric to dictionary."""
        return {
            "name": self.name,
            "column_type": self.column_type,
            "fidelity_score": round(self.fidelity_score, 4),
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "real_mean": round(self.real_mean, 4) if self.real_mean is not None else None,
            "synth_mean": round(self.synth_mean, 4) if self.synth_mean is not None else None,
        }


@dataclass
class FidelityResult:
    """Aggregated fidelity evaluation result across all columns and dependencies."""
    overall_score: float  # 0 to 100
    marginal_fidelity_score: float  # 0 to 100
    correlation_fidelity_score: float  # 0 to 100
    column_fidelities: Dict[str, ColumnFidelity] = field(default_factory=dict)
    correlation_mae: float = 0.0
    correlation_matrix_real: Optional[pd.DataFrame] = None
    correlation_matrix_synth: Optional[pd.DataFrame] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts full fidelity metrics to dictionary."""
        return {
            "overall_score": round(self.overall_score, 2),
            "marginal_fidelity_score": round(self.marginal_fidelity_score, 2),
            "correlation_fidelity_score": round(self.correlation_fidelity_score, 2),
            "correlation_mae": round(self.correlation_mae, 4),
            "column_fidelities": {k: v.to_dict() for k, v in self.column_fidelities.items()},
            "summary": self.summary,
        }


class FidelityEvaluator:
    """Evaluates marginal distributions and multi-column correlation fidelity."""

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> FidelityResult:
        """Runs comprehensive statistical fidelity benchmarks between real and synthetic tables."""
        if real_df.empty or synthetic_df.empty:
            raise ValueError("Input DataFrames cannot be empty")

        common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        if not common_cols:
            raise ValueError("No common columns found between real and synthetic DataFrames")

        column_fidelities: Dict[str, ColumnFidelity] = {}
        marginal_scores: List[float] = []

        num_cols: List[str] = []

        for col in common_cols:
            r_col = real_df[col].dropna()
            s_col = synthetic_df[col].dropna()

            if len(r_col) == 0 or len(s_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(r_col):
                num_cols.append(col)
                # Two-sample Kolmogorov-Smirnov test
                r_arr = r_col.to_numpy(dtype=np.float64)
                s_arr = s_col.to_numpy(dtype=np.float64)

                ks_stat, ks_pval = stats.ks_2samp(r_arr, s_arr)
                col_score = max(0.0, 1.0 - float(ks_stat))
                marginal_scores.append(col_score)

                column_fidelities[col] = ColumnFidelity(
                    name=col,
                    column_type="numerical",
                    fidelity_score=col_score,
                    metric_name="Kolmogorov-Smirnov (1 - stat)",
                    metric_value=float(ks_stat),
                    p_value=float(ks_pval),
                    real_mean=float(np.mean(r_arr)),
                    synth_mean=float(np.mean(s_arr)),
                    real_std=float(np.std(r_arr)),
                    synth_std=float(np.std(s_arr)),
                )
            else:
                # Categorical / discrete column: Total Variation Distance (TVD)
                r_counts = r_col.value_counts(normalize=True)
                s_counts = s_col.value_counts(normalize=True)
                all_cats = list(set(r_counts.index).union(set(s_counts.index)))

                p_vec = np.array([r_counts.get(c, 0.0) for c in all_cats])
                q_vec = np.array([s_counts.get(c, 0.0) for c in all_cats])

                # Total Variation Distance = 0.5 * sum(|p - q|) in [0, 1]
                tvd = 0.5 * np.sum(np.abs(p_vec - q_vec))
                col_score = max(0.0, 1.0 - float(tvd))
                marginal_scores.append(col_score)

                column_fidelities[col] = ColumnFidelity(
                    name=col,
                    column_type="categorical",
                    fidelity_score=col_score,
                    metric_name="Total Variation Distance (1 - TVD)",
                    metric_value=float(tvd),
                    p_value=None,
                    real_mean=None,
                    synth_mean=None,
                    real_std=None,
                    synth_std=None,
                )

        avg_marginal = float(np.mean(marginal_scores)) if marginal_scores else 0.5

        # Correlation Fidelity
        corr_mae = 0.0
        corr_fidelity = 1.0
        r_corr_df = None
        s_corr_df = None

        if len(num_cols) >= 2:
            r_corr = real_df[num_cols].corr().fillna(0.0)
            s_corr = synthetic_df[num_cols].corr().fillna(0.0)
            r_corr_df = r_corr
            s_corr_df = s_corr

            # Compute upper triangular differences
            triu_idx = np.triu_indices(len(num_cols), k=1)
            diffs = np.abs(r_corr.to_numpy()[triu_idx] - s_corr.to_numpy()[triu_idx])
            corr_mae = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
            corr_fidelity = max(0.0, 1.0 - corr_mae)
        else:
            corr_fidelity = avg_marginal

        overall_score = (0.55 * avg_marginal + 0.45 * corr_fidelity) * 100.0
        overall_score = float(np.clip(overall_score, 0.0, 100.0))

        summary = (
            f"Overall Fidelity: {overall_score:.1f}/100 "
            f"(Marginal: {avg_marginal*100:.1f}%, Correlation: {corr_fidelity*100:.1f}%, Corr MAE: {corr_mae:.3f})"
        )

        return FidelityResult(
            overall_score=overall_score,
            marginal_fidelity_score=avg_marginal * 100.0,
            correlation_fidelity_score=corr_fidelity * 100.0,
            column_fidelities=column_fidelities,
            correlation_mae=corr_mae,
            correlation_matrix_real=r_corr_df,
            correlation_matrix_synth=s_corr_df,
            summary=summary,
        )
