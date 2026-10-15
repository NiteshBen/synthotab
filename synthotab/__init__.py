"""SynthoTab: Production-grade synthetic tabular data generation & differential privacy engine."""

from synthotab.core.schema import ColumnMetadata, ColumnType, TableSchema
from synthotab.core.base import BaseSynthesizer
from synthotab.synthesizers.copula import GaussianCopulaSynthesizer
from synthotab.synthesizers.conditional import CategoricalSynthesizer
from synthotab.synthesizers.pipeline import SynthoTab
from synthotab.privacy.differential import (
    LaplaceMechanism,
    GaussianMechanism,
    PrivacyAccountant,
)
from synthotab.privacy.audit import PrivacyAuditor, PrivacyAuditResult
from synthotab.metrics.fidelity import (
    ColumnFidelity,
    FidelityEvaluator,
    FidelityResult,
)
from synthotab.reporting.html_report import generate_html_report
from synthotab.datasets.loaders import (
    load_healthcare_patients,
    load_financial_churn,
)

__version__ = "0.1.0"
__author__ = "Nitesh Beniwal"

__all__ = [
    "__version__",
    "__author__",
    "SynthoTab",
    "TableSchema",
    "ColumnMetadata",
    "ColumnType",
    "BaseSynthesizer",
    "GaussianCopulaSynthesizer",
    "CategoricalSynthesizer",
    "LaplaceMechanism",
    "GaussianMechanism",
    "PrivacyAccountant",
    "PrivacyAuditor",
    "PrivacyAuditResult",
    "FidelityEvaluator",
    "FidelityResult",
    "ColumnFidelity",
    "generate_html_report",
    "load_healthcare_patients",
    "load_financial_churn",
]
