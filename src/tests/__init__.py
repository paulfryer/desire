"""
Predicate tests for measuring desire-like behavioral signatures.

These tests implement protocols for measuring specific behavioral
predicates that distinguish desire-like behavior from reward-seeking.
"""

from src.tests.base_test import BasePredicateTest
from src.tests.frustration import FrustrationIntensificationTest
from src.tests.means_end import MeansEndDecouplingTest

__all__ = [
    "BasePredicateTest",
    "FrustrationIntensificationTest",
    "MeansEndDecouplingTest",
]
