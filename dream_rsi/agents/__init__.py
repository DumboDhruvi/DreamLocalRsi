"""Agent adapter interfaces and implementations."""

from dream_rsi.agents.base import AgentAdapter
from dream_rsi.agents.mock import MockAgent
from dream_rsi.agents.generic_cli import GenericCLIAgent
from dream_rsi.agents.antigravity import AntigravityAgent

__all__ = [
    "AgentAdapter",
    "MockAgent",
    "GenericCLIAgent",
    "AntigravityAgent",
]
