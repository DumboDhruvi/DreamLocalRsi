"""Discovery Tree data structure and diversity analytics."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from dream_rsi.core.models import NodeStatus, TreeNode


class DiscoveryTree:
    """Manages the graph of coding attempts, branch ancestry, and diversity metrics."""

    def __init__(self, tree_id: str, task_id: str):
        self.tree_id = tree_id
        self.task_id = task_id
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None

    def add_root(self, node: TreeNode) -> None:
        if self.root_id is not None:
            raise ValueError(f"Tree {self.tree_id} already has a root node ({self.root_id}).")
        node.parent_id = None
        node.depth = 0
        self.nodes[node.node_id] = node
        self.root_id = node.node_id

    def add_child(self, parent_id: str, child: TreeNode) -> None:
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node {parent_id} does not exist in tree {self.tree_id}.")
        parent = self.nodes[parent_id]
        child.parent_id = parent_id
        child.depth = parent.depth + 1
        if child.node_id not in parent.children_ids:
            parent.children_ids.append(child.node_id)
        self.nodes[child.node_id] = child

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.nodes.get(node_id)

    def get_root(self) -> Optional[TreeNode]:
        return self.nodes.get(self.root_id) if self.root_id else None

    def get_leaves(self) -> List[TreeNode]:
        return [node for node in self.nodes.values() if node.is_leaf]

    def get_eligible_nodes(self) -> List[TreeNode]:
        """Eligible nodes for expansion per Dream-RSI: root and current leaves."""
        eligible = []
        root = self.get_root()
        if root is not None:
            eligible.append(root)
        leaves = [node for node in self.get_leaves() if node.node_id != self.root_id]
        eligible.extend(leaves)
        return eligible

    def get_ancestors(self, node_id: str) -> List[TreeNode]:
        ancestors = []
        curr = self.nodes.get(node_id)
        while curr and curr.parent_id:
            parent = self.nodes.get(curr.parent_id)
            if parent:
                ancestors.append(parent)
                curr = parent
            else:
                break
        return ancestors

    def get_branch_path(self, node_id: str) -> List[TreeNode]:
        """Path from root down to the given node."""
        path = self.get_ancestors(node_id)
        path.reverse()
        node = self.nodes.get(node_id)
        if node:
            path.append(node)
        return path

    def get_best_node(self) -> Optional[TreeNode]:
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: n.score)

    def compute_file_overlap(self, node_a: TreeNode, node_b: TreeNode) -> float:
        """Jaccard similarity between files changed in node_a and node_b."""
        set_a = set(node_a.changed_files)
        set_b = set(node_b.changed_files)
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def calculate_diversity_penalty(
        self, candidate_node: TreeNode, recent_nodes: List[TreeNode]
    ) -> float:
        """Calculates penalty based on file and strategy overlap with recent branches."""
        if not recent_nodes:
            return 0.0
        overlaps = [self.compute_file_overlap(candidate_node, r) for r in recent_nodes]
        same_strategy = [1.0 if candidate_node.strategy_id == r.strategy_id else 0.0 for r in recent_nodes]
        avg_file_overlap = sum(overlaps) / len(overlaps)
        avg_strat_overlap = sum(same_strategy) / len(same_strategy)
        return 0.6 * avg_file_overlap + 0.4 * avg_strat_overlap

    def get_stats(self) -> Dict[str, Any]:
        if not self.nodes:
            return {
                "tree_id": self.tree_id,
                "task_id": self.task_id,
                "total_nodes": 0,
                "max_score": 0.0,
                "max_depth": 0,
            }
        scores = [n.score for n in self.nodes.values()]
        depths = [n.depth for n in self.nodes.values()]
        costs = [n.cost_usd for n in self.nodes.values()]
        durations = [n.duration_seconds for n in self.nodes.values()]

        return {
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "total_nodes": len(self.nodes),
            "root_id": self.root_id,
            "num_leaves": len(self.get_leaves()),
            "max_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "max_depth": max(depths),
            "total_cost_usd": sum(costs),
            "total_duration_seconds": sum(durations),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscoveryTree:
        tree = cls(tree_id=data["tree_id"], task_id=data["task_id"])
        tree.root_id = data.get("root_id")
        for nid, ndict in data.get("nodes", {}).items():
            tree.nodes[nid] = TreeNode.from_dict(ndict)
        return tree
