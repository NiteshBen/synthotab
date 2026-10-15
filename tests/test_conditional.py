"""Unit tests for CategoricalSynthesizer."""

import pytest
import pandas as pd
from synthotab.synthesizers.conditional import CategoricalSynthesizer


def test_categorical_fit_and_sample():
    df = pd.DataFrame({
        "city": ["New York", "London", "Tokyo", "London", "New York", "London"],
        "tier": ["Gold", "Silver", "Platinum", "Silver", "Gold", "Bronze"],
        "subscribed": [True, False, True, False, True, False],
    })

    cat_synth = CategoricalSynthesizer(random_state=42)
    cat_synth.fit(df)
    assert cat_synth.is_fitted is True

    synth_df = cat_synth.sample(100)
    assert synth_df.shape == (100, 3)
    assert set(synth_df["city"].unique()).issubset({"New York", "London", "Tokyo"})
    assert set(synth_df["tier"].unique()).issubset({"Gold", "Silver", "Platinum", "Bronze"})
    assert synth_df["subscribed"].dtype == bool


def test_categorical_unfitted_sample_error():
    synth = CategoricalSynthesizer()
    with pytest.raises(RuntimeError):
        synth.sample(5)


def test_categorical_no_categorical_columns():
    synth = CategoricalSynthesizer()
    with pytest.raises(ValueError):
        synth.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
