"""Differential Privacy mechanisms and privacy budget accountant for tabular data synthesis."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class LaplaceMechanism:
    """Implements the standard Laplace mechanism for epsilon-differential privacy.

    Noise added is drawn from Laplace(0, scale), where scale = sensitivity / epsilon.
    """

    def __init__(self, epsilon: float, sensitivity: float = 1.0, random_state: Optional[int] = None) -> None:
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be strictly positive, got {sensitivity}")

        self.epsilon = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.scale = self.sensitivity / self.epsilon
        self._rng = np.random.default_rng(random_state)

    def perturb(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Injects calibrated Laplace noise into the target value or array."""
        noise = self._rng.laplace(loc=0.0, scale=self.scale, size=np.shape(value))
        return value + noise

    def perturb_histogram(self, counts: np.ndarray, min_count: float = 0.0) -> np.ndarray:
        """Perturbs a discrete count histogram and enforces non-negativity and normalization."""
        # For histograms where each individual affects at most one bin, L1 sensitivity is 1.0
        noisy_counts = self.perturb(counts.astype(np.float64))
        clipped = np.maximum(noisy_counts, min_count)
        total = np.sum(clipped)
        if total == 0:
            return np.ones_like(counts, dtype=np.float64) / len(counts)
        return clipped / total


class GaussianMechanism:
    """Implements the Gaussian mechanism for (epsilon, delta)-differential privacy.

    Noise variance sigma^2 = 2 * ln(1.25 / delta) * (sensitivity / epsilon)^2.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"Delta must be in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be strictly positive, got {sensitivity}")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.sigma = (self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta))) / self.epsilon
        self._rng = np.random.default_rng(random_state)

    def perturb(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Injects calibrated Gaussian noise into the target value or matrix."""
        noise = self._rng.normal(loc=0.0, scale=self.sigma, size=np.shape(value))
        return value + noise


class PrivacyAccountant:
    """Tracks cumulative epsilon and delta expenditures across synthesizer modules."""

    def __init__(self, target_epsilon: Optional[float] = None, target_delta: float = 1e-5) -> None:
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.audit_log: List[Dict[str, Union[str, float]]] = []

    def allocate(
        self,
        operation_name: str,
        epsilon: float,
        delta: float = 0.0,
    ) -> Tuple[float, float]:
        """Allocates a portion of the privacy budget to a specific operation."""
        if epsilon <= 0:
            raise ValueError("Allocated epsilon must be positive")

        if self.target_epsilon is not None and (self.spent_epsilon + epsilon > self.target_epsilon * 1.0001):
            raise ValueError(
                f"Privacy budget exceeded: requested {epsilon:.4f}, but remaining is "
                f"{max(0.0, self.target_epsilon - self.spent_epsilon):.4f}"
            )

        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.audit_log.append({
            "operation": operation_name,
            "epsilon": float(epsilon),
            "delta": float(delta),
            "cumulative_epsilon": float(self.spent_epsilon),
        })
        return epsilon, delta

    @property
    def remaining_epsilon(self) -> Optional[float]:
        """Remaining unused epsilon budget."""
        if self.target_epsilon is None:
            return None
        return max(0.0, self.target_epsilon - self.spent_epsilon)

    def summary(self) -> Dict[str, Union[float, int, List[Dict[str, Union[str, float]]]]]:
        """Returns comprehensive privacy ledger summary."""
        return {
            "target_epsilon": self.target_epsilon if self.target_epsilon is not None else -1.0,
            "target_delta": self.target_delta,
            "spent_epsilon": round(self.spent_epsilon, 4),
            "spent_delta": self.spent_delta,
            "remaining_epsilon": round(self.remaining_epsilon, 4) if self.remaining_epsilon is not None else -1.0,
            "total_operations": len(self.audit_log),
            "audit_log": self.audit_log,
        }
