"""Gaussian Copula Synthesizer for high-fidelity multivariate continuous data modeling."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri, ndtr

from synthotab.core.base import BaseSynthesizer
from synthotab.core.schema import TableSchema, ColumnType


class GaussianCopulaSynthesizer(BaseSynthesizer):
    """Synthesizes continuous and integer tabular columns using a Gaussian Copula.

    Applies the Probability Integral Transform (PIT) to map arbitrary marginal
    distributions to standard normal variables, models linear and non-linear
    dependencies via a latent covariance matrix, and inverts generated latent
    samples back into the original feature space via empirical quantile interpolation.
    """

    def __init__(
        self,
        random_state: Optional[int] = 42,
        jitter_strength: float = 1e-5,
    ) -> None:
        super().__init__(random_state=random_state)
        self.jitter_strength = jitter_strength
        self.columns: List[str] = []
        self.sorted_marginals: Dict[str, np.ndarray] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame) -> GaussianCopulaSynthesizer:
        """Fits empirical marginal distributions and estimates latent Gaussian covariance."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected input df to be a pandas DataFrame")
        if df.empty:
            raise ValueError("Cannot fit synthesizer on an empty DataFrame")

        self.schema = TableSchema.infer(df)
        num_cols = self.schema.numerical_columns

        if not num_cols:
            raise ValueError("No numerical columns found in the dataset to fit Gaussian Copula")

        self.columns = num_cols
        n_samples = len(df)
        latent_matrix = np.zeros((n_samples, len(self.columns)), dtype=np.float64)

        for col_idx, col in enumerate(self.columns):
            series = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(series) == 0:
                series = np.array([0.0, 1.0])

            # Store sorted empirical values for inverse quantile lookup
            sorted_vals = np.sort(series)
            self.sorted_marginals[col] = sorted_vals

            # Compute ranks with randomized jitter to break ties smoothly
            ranks = stats.rankdata(series, method="average")
            if self.jitter_strength > 0:
                jitter = self._rng.uniform(-self.jitter_strength, self.jitter_strength, size=len(series))
                ranks = np.clip(ranks + jitter, 1.0, len(series))

            # Transform to uniform u in (0, 1) using Weibull/Hazen plotting positions
            u = (ranks - 0.5) / len(series)
            u = np.clip(u, 1e-6, 1.0 - 1e-6)

            # Map to standard normal latent space: z = Phi^-1(u)
            z = ndtri(u)
            latent_matrix[:, col_idx] = z

        # Compute empirical correlation matrix in latent Gaussian space
        corr = np.corrcoef(latent_matrix, rowvar=False)
        if np.isscalar(corr):
            corr = np.array([[1.0]])
        elif corr.ndim == 1:
            corr = corr.reshape(1, 1)

        # Enforce positive semi-definiteness via eigenvalue thresholding
        corr = self._regularize_correlation_matrix(corr)
        self.correlation_matrix = corr
        self._is_fitted = True
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Samples synthetic rows by drawing from multivariate Gaussian and inverting marginals."""
        if not self._is_fitted or self.correlation_matrix is None:
            raise RuntimeError("Synthesizer must be fitted before calling sample()")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be a positive integer, got {n_rows}")

        n_features = len(self.columns)
        mean_zeros = np.zeros(n_features)

        # Draw samples from latent multivariate Gaussian
        latent_samples = self._rng.multivariate_normal(
            mean=mean_zeros,
            cov=self.correlation_matrix,
            size=n_rows,
            check_valid="ignore",
        )

        # Map back from normal to uniform: u = Phi(z)
        uniform_samples = ndtr(latent_samples)
        uniform_samples = np.clip(uniform_samples, 0.0, 1.0)

        # Invert uniforms to original marginal space using empirical quantile interpolation
        synthetic_dict: Dict[str, np.ndarray] = {}
        for col_idx, col in enumerate(self.columns):
            sorted_vals = self.sorted_marginals[col]
            u_col = uniform_samples[:, col_idx]
            # Fast linear quantile lookup across empirical distribution
            synth_vals = np.quantile(sorted_vals, u_col, method="linear")

            meta = self.schema.columns.get(col)
            if meta and meta.col_type == ColumnType.INTEGER:
                synth_vals = np.round(synth_vals).astype(int)
                if meta.min_val is not None and meta.max_val is not None:
                    synth_vals = np.clip(synth_vals, meta.min_val, meta.max_val)
            elif meta and meta.col_type == ColumnType.CONTINUOUS:
                if meta.min_val is not None and meta.max_val is not None:
                    synth_vals = np.clip(synth_vals, meta.min_val, meta.max_val)

            synthetic_dict[col] = synth_vals

        return pd.DataFrame(synthetic_dict)

    @staticmethod
    def _regularize_correlation_matrix(matrix: np.ndarray, min_eigenval: float = 1e-4) -> np.ndarray:
        """Projects a matrix onto the positive semi-definite cone and restores unit diagonal."""
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)

        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        if np.any(eigenvals < min_eigenval):
            clipped_eigenvals = np.maximum(eigenvals, min_eigenval)
            matrix = eigenvecs @ np.diag(clipped_eigenvals) @ eigenvecs.T

            # Normalize to ensure correlation diagonal entries are exactly 1.0
            diag = np.sqrt(np.diag(matrix))
            diag[diag == 0] = 1.0
            matrix = matrix / np.outer(diag, diag)
            np.fill_diagonal(matrix, 1.0)

        return matrix
