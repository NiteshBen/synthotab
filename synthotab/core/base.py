"""Base synthesizer interface for SynthoTab."""

from __future__ import annotations

import abc
from typing import Optional
import pandas as pd
from synthotab.core.schema import TableSchema


class BaseSynthesizer(abc.ABC):
    """Abstract base class for all tabular data synthesizers."""

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self.random_state = random_state
        self.schema: Optional[TableSchema] = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Indicates whether the synthesizer has been fitted to data."""
        return self._is_fitted

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseSynthesizer:
        """Learns data distributions and dependencies from input DataFrame."""
        pass

    @abc.abstractmethod
    def sample(self, n_rows: int) -> pd.DataFrame:
        """Samples synthetic rows adhering to the learned data distributions."""
        pass

    def fit_sample(self, df: pd.DataFrame, n_rows: Optional[int] = None) -> pd.DataFrame:
        """Convenience method to fit on real data and immediately sample synthetic data."""
        self.fit(df)
        sample_count = n_rows if n_rows is not None else len(df)
        return self.sample(sample_count)
