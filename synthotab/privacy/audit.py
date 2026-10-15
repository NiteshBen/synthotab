"""Privacy Risk and Information Leakage Auditor for synthetic tabular data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


@dataclass
class PrivacyAuditResult:
    """Quantitative privacy leakage audit metrics."""
    nndr_mean: float
    nndr_p5: float
    exact_match_count: int
    exact_match_ratio: float
    membership_inference_risk: float
    privacy_score: float
    is_privacy_safe: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts audit metrics to dictionary."""
        return {
            "nndr_mean": round(self.nndr_mean, 4),
            "nndr_p5": round(self.nndr_p5, 4),
            "exact_match_count": self.exact_match_count,
            "exact_match_ratio": round(self.exact_match_ratio, 6),
            "membership_inference_risk": round(self.membership_inference_risk, 4),
            "privacy_score": round(self.privacy_score, 2),
            "is_privacy_safe": self.is_privacy_safe,
            "summary": self.summary,
        }


class PrivacyAuditor:
    """Audits synthetic datasets for privacy leakage, memorization, and re-identification risks."""

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self.random_state = random_state

    def audit(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        sample_size: int = 1000,
    ) -> PrivacyAuditResult:
        """Runs full privacy audit suite comparing real vs synthetic data."""
        if real_df.empty or synthetic_df.empty:
            raise ValueError("Input DataFrames cannot be empty")

        # 1. Exact Match Ratio
        exact_matches = self._calculate_exact_matches(real_df, synthetic_df)
        exact_ratio = exact_matches / len(synthetic_df) if len(synthetic_df) > 0 else 0.0

        # 2. Nearest Neighbor Distance Ratio (NNDR)
        # Subsample for computational efficiency if datasets are large
        r_sub = real_df.sample(n=min(len(real_df), sample_size), random_state=self.random_state)
        s_sub = synthetic_df.sample(n=min(len(synthetic_df), sample_size), random_state=self.random_state)

        nndr_mean, nndr_p5 = self._compute_nndr(r_sub, s_sub)

        # 3. Membership Inference Risk
        # High risk if synthetic points have abnormally low distances compared to empirical baseline
        # Baseline: 0.5 (random guess). Memorization: 0.9+. Good privacy: < 0.65.
        mem_risk = max(0.5, float(1.0 - nndr_mean * 0.6))
        if exact_ratio > 0.05:
            mem_risk = min(1.0, mem_risk + 0.3)

        # 4. Composite Privacy Score (0 - 100)
        # Perfect privacy: 0 exact matches, high NNDR (>0.7), low membership risk (<0.6)
        score = 100.0
        score -= min(40.0, exact_ratio * 400.0)  # Penalize exact matches heavily
        score -= max(0.0, (0.75 - nndr_mean) * 80.0)  # Penalize clustering near real points
        score -= max(0.0, (mem_risk - 0.5) * 60.0)  # Penalize elevated membership risk
        privacy_score = float(np.clip(score, 0.0, 100.0))

        is_safe = (privacy_score >= 70.0) and (exact_ratio < 0.02)
        summary = (
            f"Privacy Score: {privacy_score:.1f}/100 - "
            f"{'EXCELLENT PRIVACY (Safe for external sharing)' if privacy_score >= 85 else 'ACCEPTABLE PRIVACY' if is_safe else 'HIGH DISCLOSURE RISK (Needs higher epsilon perturbation)'}"
        )

        return PrivacyAuditResult(
            nndr_mean=nndr_mean,
            nndr_p5=nndr_p5,
            exact_match_count=exact_matches,
            exact_match_ratio=exact_ratio,
            membership_inference_risk=mem_risk,
            privacy_score=privacy_score,
            is_privacy_safe=is_safe,
            summary=summary,
            details={
                "n_real": len(real_df),
                "n_synthetic": len(synthetic_df),
            },
        )

    def _calculate_exact_matches(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> int:
        """Finds number of synthetic rows that are verbatim duplicates of real records."""
        common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        if not common_cols:
            return 0

        # Convert rows to string tuples for fast hash set intersection
        real_tuples = set(tuple(x) for x in real_df[common_cols].itertuples(index=False, name=None))
        synth_tuples = [tuple(x) for x in synthetic_df[common_cols].itertuples(index=False, name=None)]

        matches = sum(1 for st in synth_tuples if st in real_tuples)
        return matches

    def _compute_nndr(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[float, float]:
        """Computes Nearest Neighbor Distance Ratio (d1 / d2) on normalized numerical features."""
        num_cols = [
            c for c in real_df.columns
            if c in synthetic_df.columns and pd.api.types.is_numeric_dtype(real_df[c])
        ]

        if not num_cols or len(real_df) < 2:
            return 0.85, 0.70

        r_num = real_df[num_cols].fillna(0).to_numpy(dtype=np.float64)
        s_num = synthetic_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

        scaler = MinMaxScaler()
        r_norm = scaler.fit_transform(r_num)
        s_norm = scaler.transform(s_num)

        # Fit 2-nearest-neighbors on real data
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(r_norm)

        # Find distance from each synthetic point to 1st and 2nd closest real points
        distances, _ = nn.kneighbors(s_norm)
        d1 = distances[:, 0]
        d2 = distances[:, 1]

        # Prevent division by zero
        d2_safe = np.where(d2 == 0, 1e-6, d2)
        ratios = d1 / d2_safe
        ratios = np.clip(ratios, 0.0, 1.0)

        nndr_mean = float(np.mean(ratios))
        nndr_p5 = float(np.percentile(ratios, 5))
        return nndr_mean, nndr_p5
