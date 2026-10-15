"""Unit tests for the top-level SynthoTab pipeline."""

import pytest
import numpy as np
import pandas as pd
from synthotab.synthesizers.pipeline import SynthoTab


def test_synthotab_end_to_end():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(150)],
        "age": rng.integers(20, 60, size=150),
        "income": rng.normal(50000, 15000, size=150),
        "tier": rng.choice(["Bronze", "Silver", "Gold"], size=150),
        "churn": rng.choice([True, False], size=150),
    })

    model = SynthoTab(random_state=42)
    model.fit(df)

    assert model.is_fitted is True

    synth_df = model.sample(200)
    assert len(synth_df) == 200
    assert list(synth_df.columns) == list(df.columns)

    # Check bounds
    assert synth_df["age"].min() >= df["age"].min()
    assert synth_df["age"].max() <= df["age"].max()

    # Evaluate
    fidelity = model.evaluate(df, synth_df)
    assert fidelity.overall_score > 60.0

    # Audit privacy
    privacy = model.audit_privacy(df, synth_df)
    assert privacy.privacy_score > 70.0


def test_synthotab_with_differential_privacy():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": rng.normal(10, 2, 100),
        "y": rng.normal(20, 5, 100),
        "group": rng.choice(["G1", "G2"], 100),
    })

    model = SynthoTab(epsilon=1.0, random_state=42)
    model.fit(df)

    assert model.accountant.spent_epsilon > 0.0
    synth_df = model.sample(50)
    assert len(synth_df) == 50
    assert "x" in synth_df.columns and "group" in synth_df.columns


def test_synthotab_unfitted_sample_raises():
    model = SynthoTab()
    with pytest.raises(RuntimeError):
        model.sample(10)
