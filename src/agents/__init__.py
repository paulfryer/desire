"""
Agent implementations for the Desire research project.

Provides different agent architectures (heuristic, RL, active inference)
with a common interface for behavioral comparison.
"""

from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.rl_agent import RLAgent
from src.agents.active_inference import ActiveInferenceAgent

__all__ = [
    "BaseAgent",
    "HeuristicAgent",
    "RLAgent",
    "ActiveInferenceAgent",
]
