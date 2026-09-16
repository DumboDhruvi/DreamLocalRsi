"""Policy base utilities and scoring heuristics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from dream_rsi.core.models import TreeNode
from dream_rsi.core.policy import ExplorationPolicy, PolicyConfig
from dream_rsi.core.tree import DiscoveryTree


class BasePolicy(ExplorationPolicy):
    """Common functionality for tree tracking, visit counts, and scoring."""

    def __init__(self, policy_id: str, config: Optional[PolicyConfig] = None):
        super().__init__(policy_id, config)
        self.node_visit_counts: Dict[str, int] = {}
        self.total_expansions: int = 0

    def observe(self, node: TreeNode) -> None:
        if node.parent_id:
            self.node_visit_counts[node.parent_id] = (
                self.node_visit_counts.get(node.parent_id, 0) + 1
            )
        self.total_expansions += 1

    def get_visits(self, node_id: str) -> int:
        return self.node_visit_counts.get(node_id, 0)
