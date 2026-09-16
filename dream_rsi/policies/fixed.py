"""Fixed baseline exploration policy."""

from __future__ import annotations

from typing import List

from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.policies.base import BasePolicy


class FixedPolicy(BasePolicy):
    """Simple baseline policy: explores root breadth first, then greedily refines best leaves."""

    def choose_batch(
        self,
        tree: DiscoveryTree,
        budget_mgr: BudgetManager,
        max_workers: int,
    ) -> List[str]:
        root = tree.get_root()
        if not root:
            return []

        # If root has fewer than branching_factor children, expand root
        root_children_count = len(root.children_ids)
        branching_factor = self.config.branching_factor

        batch: List[str] = []
        if root_children_count < branching_factor:
            # Need more root branches
            needed = min(branching_factor - root_children_count, max_workers)
            batch.extend([root.node_id] * needed)

        remaining_slots = max_workers - len(batch)
        if remaining_slots > 0:
            leaves = [n for n in tree.get_leaves() if n.node_id != root.node_id and n.depth < self.config.max_depth]
            if leaves:
                # Sort leaves by score descending
                sorted_leaves = sorted(leaves, key=lambda n: n.score, reverse=True)
                for leaf in sorted_leaves[:remaining_slots]:
                    batch.append(leaf.node_id)
            elif not batch:
                # Fallback to root if leaves at max depth
                batch.append(root.node_id)

        return batch[:max_workers]
