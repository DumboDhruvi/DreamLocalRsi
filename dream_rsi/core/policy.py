"""Exploration policy base interfaces and configurations."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.models import TreeNode
from dream_rsi.core.tree import DiscoveryTree


@dataclass
class PolicyConfig:
    """Configurable parameters for exploration policies."""
    branching_factor: int = 2
    max_depth: int = 6
    parallel_workers: int = 2
    prefer_high_score: float = 0.50
    prefer_diversity: float = 0.30
    prefer_unexplored: float = 0.20
    stop_when_score: float = 0.98
    exploration_constant: float = 1.414  # UCB constant
    diversity_weight: float = 0.15
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branching_factor": self.branching_factor,
            "max_depth": self.max_depth,
            "parallel_workers": self.parallel_workers,
            "prefer_high_score": self.prefer_high_score,
            "prefer_diversity": self.prefer_diversity,
            "prefer_unexplored": self.prefer_unexplored,
            "stop_when_score": self.stop_when_score,
            "exploration_constant": self.exploration_constant,
            "diversity_weight": self.diversity_weight,
            **self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyConfig:
        data = dict(data)
        standard_keys = {
            "branching_factor",
            "max_depth",
            "parallel_workers",
            "prefer_high_score",
            "prefer_diversity",
            "prefer_unexplored",
            "stop_when_score",
            "exploration_constant",
            "diversity_weight",
        }
        known = {k: v for k, v in data.items() if k in standard_keys}
        extra = {k: v for k, v in data.items() if k not in standard_keys}
        return cls(**known, extra_params=extra)


class ExplorationPolicy(abc.ABC):
    """Abstract base class for exploration policies in both online and replay environments."""

    def __init__(self, policy_id: str, config: Optional[PolicyConfig] = None):
        self.policy_id = policy_id
        self.config = config or PolicyConfig()

    @abc.abstractmethod
    def choose_batch(
        self,
        tree: DiscoveryTree,
        budget_mgr: BudgetManager,
        max_workers: int,
    ) -> List[str]:
        """Selects a batch of eligible node IDs (root or current leaves) to expand or refine."""
        pass

    def should_stop(self, tree: DiscoveryTree, budget_mgr: BudgetManager) -> bool:
        """Determines whether exploration should terminate early."""
        best = tree.get_best_node()
        if best and best.score >= self.config.stop_when_score:
            return True
        return budget_mgr.is_exhausted()

    def observe(self, node: TreeNode) -> None:
        """Hook called after an action produces a child node."""
        pass
