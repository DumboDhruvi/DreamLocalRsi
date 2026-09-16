"""Adaptive UCB exploration policy with diversity penalties."""

from __future__ import annotations

import math
from typing import List

from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.policies.base import BasePolicy


class AdaptivePolicy(BasePolicy):
    """UCB-style exploration policy balancing exploitation, exploration, and branch diversity."""

    def choose_batch(
        self,
        tree: DiscoveryTree,
        budget_mgr: BudgetManager,
        max_workers: int,
    ) -> List[str]:
        eligible = [
            n for n in tree.get_eligible_nodes()
            if n.depth < self.config.max_depth
        ]
        if not eligible:
            root = tree.get_root()
            return [root.node_id] if root else []

        total_v = max(1, self.total_expansions)
        c = self.config.exploration_constant
        div_w = self.config.diversity_weight

        # Recent completed nodes for diversity penalty
        recent_nodes = list(tree.nodes.values())[-5:]

        scored_candidates = []
        for node in eligible:
            visits = self.get_visits(node.node_id)
            # UCB exploration bonus
            exploration_bonus = c * math.sqrt(math.log(total_v + 1.0) / (visits + 1.0))
            exploitation_score = node.score

            # Diversity penalty
            div_penalty = tree.calculate_diversity_penalty(node, recent_nodes)

            # Composite priority
            priority = (
                self.config.prefer_high_score * exploitation_score
                + self.config.prefer_unexplored * exploration_bonus
                - div_w * div_penalty
            )
            scored_candidates.append((priority, node.node_id))

        # Sort descending by priority
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored_candidates[:max_workers]]
