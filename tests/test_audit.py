"""Unit tests for PrivacyAuditor."""

import pytest
import numpy as np
import pandas as pd
from synthotab.privacy.audit import PrivacyAuditor


def test_audit_exact_matches():
    real_df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "feature": [10.0, 20.0, 30.0, 40.0],
    })

    # Exactly replicated row #1 and #2
    synth_df = pd.DataFrame({
        "id": [1, 2, 99, 100],
        "feature": [10.0, 20.0, 99.0, 100.0],
    })

    auditor = PrivacyAuditor(random_state=42)
    result = auditor.audit(real_df, synth_df)

    assert result.exact_match_count == 2
    assert result.exact_match_ratio == 0.5


def test_audit_safe_synthetic_data():
    rng = np.random.default_rng(42)
    # Generate non-overlapping continuous distributions
    real_df = pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(10, 2, 200),
    })

    synth_df = pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(10, 2, 200),
    })

    auditor = PrivacyAuditor(random_state=42)
    result = auditor.audit(real_df, synth_df)

    assert result.exact_match_count == 0
    assert result.nndr_mean > 0.5
    assert result.privacy_score >= 70.0
    assert result.is_privacy_safe is True
    assert "summary" in result.to_dict()


def test_audit_empty_df_raises():
    auditor = PrivacyAuditor()
    with pytest.raises(ValueError):
        auditor.audit(pd.DataFrame(), pd.DataFrame())
