"""Unit tests for Differential Privacy mechanisms and PrivacyAccountant."""

import pytest
import numpy as np
from synthotab.privacy.differential import (
    LaplaceMechanism,
    GaussianMechanism,
    PrivacyAccountant,
)


def test_laplace_mechanism_noise():
    laplace = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, random_state=42)
    val = 100.0
    noisy = laplace.perturb(val)
    assert isinstance(noisy, (float, np.floating))
    assert noisy != val

    # Large sample mean should converge around 0 noise
    samples = laplace.perturb(np.zeros(10000))
    assert abs(np.mean(samples)) < 0.1


def test_laplace_histogram_perturbation():
    laplace = LaplaceMechanism(epsilon=0.5, random_state=42)
    counts = np.array([100, 250, 50, 600])
    probs = laplace.perturb_histogram(counts)

    # Probabilities must be non-negative and sum to 1.0
    assert np.all(probs >= 0.0)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)


def test_gaussian_mechanism():
    gauss = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0, random_state=42)
    arr = np.array([10.0, 20.0, 30.0])
    noisy_arr = gauss.perturb(arr)
    assert noisy_arr.shape == (3,)
    assert not np.array_equal(arr, noisy_arr)


def test_privacy_accountant_budget():
    accountant = PrivacyAccountant(target_epsilon=2.0, target_delta=1e-5)
    accountant.allocate("op1", 0.8)
    assert accountant.spent_epsilon == 0.8
    assert np.isclose(accountant.remaining_epsilon, 1.2)

    accountant.allocate("op2", 1.0)
    assert np.isclose(accountant.spent_epsilon, 1.8)

    # Exceeding budget raises ValueError
    with pytest.raises(ValueError):
        accountant.allocate("op3", 0.5)


def test_privacy_accountant_summary():
    accountant = PrivacyAccountant(target_epsilon=3.0, target_delta=1e-5)
    accountant.allocate("task_1", 1.5, delta=1e-5)
    summary = accountant.summary()
    assert summary["target_epsilon"] == 3.0
    assert summary["spent_epsilon"] == 1.5
    assert summary["total_operations"] == 1
    assert len(summary["audit_log"]) == 1


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        LaplaceMechanism(epsilon=-1.0)

    with pytest.raises(ValueError):
        GaussianMechanism(epsilon=0.0)

    with pytest.raises(ValueError):
        GaussianMechanism(epsilon=1.0, delta=1.5)
