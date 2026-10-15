"""Unit tests for GaussianCopulaSynthesizer."""

import pytest
import numpy as np
import pandas as pd
from synthotab.synthesizers.copula import GaussianCopulaSynthesizer


def test_copula_fit_and_sample_shape():
    rng = np.random.default_rng(42)
    x1 = rng.normal(10, 2, size=200)
    # Correlated x2
    x2 = x1 * 1.5 + rng.normal(0, 1, size=200)
    x3 = rng.integers(1, 100, size=200)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    copula = GaussianCopulaSynthesizer(random_state=42)
    copula.fit(df)

    assert copula.is_fitted is True
    assert copula.correlation_matrix is not None
    assert copula.correlation_matrix.shape == (3, 3)
    # Diagonal elements must be 1.0
    np.testing.assert_allclose(np.diag(copula.correlation_matrix), 1.0, atol=1e-5)

    synth_df = copula.sample(500)
    assert synth_df.shape == (500, 3)
    assert set(synth_df.columns) == {"x1", "x2", "x3"}


def test_copula_preserves_positive_correlation():
    rng = np.random.default_rng(42)
    x = rng.normal(50, 10, size=500)
    y = 2.0 * x + rng.normal(0, 5, size=500)  # Strong positive correlation

    df = pd.DataFrame({"x": x, "y": y})
    real_corr = df.corr().iloc[0, 1]

    copula = GaussianCopulaSynthesizer(random_state=42)
    copula.fit(df)
    synth_df = copula.sample(1000)
    synth_corr = synth_df.corr().iloc[0, 1]

    # Synthetic correlation should closely track real correlation within tolerance
    assert abs(real_corr - synth_corr) < 0.15


def test_copula_bounds_respected():
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    copula = GaussianCopulaSynthesizer(random_state=42)
    copula.fit(df)
    synth_df = copula.sample(100)

    assert synth_df["num"].min() >= 10.0
    assert synth_df["num"].max() <= 50.0


def test_copula_unfitted_sample_error():
    copula = GaussianCopulaSynthesizer()
    with pytest.raises(RuntimeError):
        copula.sample(10)


def test_copula_empty_dataframe_error():
    copula = GaussianCopulaSynthesizer()
    with pytest.raises(ValueError):
        copula.fit(pd.DataFrame())


def test_copula_no_numeric_columns():
    copula = GaussianCopulaSynthesizer()
    with pytest.raises(ValueError):
        copula.fit(pd.DataFrame({"cat": ["A", "B", "C"]}))
