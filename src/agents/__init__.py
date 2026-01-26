"""
Agent implementations for the Desire research project.

Provides different agent architectures (heuristic, RL, active inference)
with a common interface for behavioral comparison.
"""

from src.agents.active_inference import ActiveInferenceAgent
from src.agents.active_inference_fast import ActiveInferenceFast
from src.agents.base_agent import BaseAgent
from src.agents.decision_transformer import DecisionTransformerAgent
from src.agents.heuristic import HeuristicAgent

# Extended RL agents
from src.agents.model_based_rl import ModelBasedRLAgent
from src.agents.rl_a2c import A2CAgent
from src.agents.rl_agent import RLAgent
from src.agents.rl_dqn import DQNAgent
from src.agents.rl_sac import SACAgent

__all__ = [
    # Base
    "BaseAgent",
    # Original agents
    "HeuristicAgent",
    "RLAgent",
    "ActiveInferenceAgent",
    "ActiveInferenceFast",
    # Extended RL agents
    "ModelBasedRLAgent",
    "DQNAgent",
    "A2CAgent",
    "SACAgent",
    "DecisionTransformerAgent",
]
