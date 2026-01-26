"""
Predicate tests for measuring desire-like behavioral signatures.

These tests implement protocols for measuring specific behavioral
predicates that distinguish desire-like behavior from reward-seeking.
"""

from src.tests.anticipation import AnticipatoryActivationTest
from src.tests.base_test import BasePredicateTest
from src.tests.extinction import ExtinctionResistanceTest
from src.tests.frustration import FrustrationIntensificationTest
from src.tests.means_end import MeansEndDecouplingTest
from src.tests.opportunity_cost import OpportunityCostSensitivityTest
from src.tests.state_dependent import StateDependentValuationTest

__all__ = [
    "BasePredicateTest",
    "FrustrationIntensificationTest",
    "MeansEndDecouplingTest",
    "ExtinctionResistanceTest",
    "AnticipatoryActivationTest",
    "OpportunityCostSensitivityTest",
    "StateDependentValuationTest",
]
