"""Deterministic Replay Simulator and Dream-RSI objective evaluation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.models import Budget, ReplayResult, TreeNode
from dream_rsi.core.policy import ExplorationPolicy
from dream_rsi.core.tree import DiscoveryTree


@dataclass
class ReplayObjectiveConfig:
    """Configurable weights for the Dream-RSI replay objective function."""
    beta1: float = 0.01  # Penalty per revealed attempt
    beta2: float = 0.01  # Bonus for effective parallelism
    quality_weight: float = 1.0
    cost_weight: float = 0.0
    parallelism_weight: float = 1.0


class ReplaySimulator:
    """Simulates exploration policies on recorded discovery trees without calling LLM agents."""

    def __init__(
        self,
        full_tree: DiscoveryTree,
        objective_config: Optional[ReplayObjectiveConfig] = None,
    ):
        self.full_tree = full_tree
        self.objective_config = objective_config or ReplayObjectiveConfig()

    def simulate(
        self,
        policy: ExplorationPolicy,
        budget: Optional[Budget] = None,
    ) -> ReplayResult:
        start_time = time.time()
        sim_budget = budget or Budget()
        budget_mgr = BudgetManager(sim_budget)

        if not self.full_tree.root_id or not self.full_tree.nodes:
            return ReplayResult(
                policy_id=policy.policy_id,
                revealed_node_ids=[],
                best_node_id=None,
                best_score=0.0,
                total_attempts=0,
                decision_rounds=0,
                parallelism=0.0,
                replay_score=0.0,
                stopped_by_policy=True,
                duration_seconds=time.time() - start_time,
            )

        # Initialize replay tree containing only the root
        replay_tree = DiscoveryTree(
            tree_id=f"replay_{self.full_tree.tree_id}",
            task_id=self.full_tree.task_id,
        )
        full_root = self.full_tree.get_root()
        assert full_root is not None

        # Deep copy the root node with empty children initially
        root_copy = TreeNode.from_dict(full_root.to_dict())
        root_copy.children_ids = []
        replay_tree.add_root(root_copy)

        revealed_node_ids: List[str] = [root_copy.node_id]
        unrevealed_children: Dict[str, List[str]] = {}
        for nid, node in self.full_tree.nodes.items():
            unrevealed_children[nid] = list(node.children_ids)

        decision_rounds = 0
        total_revealed_attempts = 0
        stopped_by_policy = False

        while budget_mgr.can_start_round():
            if policy.should_stop(replay_tree, budget_mgr):
                stopped_by_policy = True
                break

            decision_rounds += 1
            budget_mgr.record_round()

            max_workers = budget_mgr.max_allowed_workers(policy.config.parallel_workers)
            if max_workers <= 0:
                break

            chosen_batch = policy.choose_batch(replay_tree, budget_mgr, max_workers)
            if not chosen_batch:
                # Policy chose no actions
                break

            batch_revealed_any = False
            for parent_id in chosen_batch:
                if not budget_mgr.can_schedule_calls(1):
                    break

                # Check if recorded parent has unrevealed children
                available = unrevealed_children.get(parent_id, [])
                if available:
                    child_id = available.pop(0)
                    full_child = self.full_tree.get_node(child_id)
                    if full_child:
                        child_copy = TreeNode.from_dict(full_child.to_dict())
                        child_copy.children_ids = []
                        replay_tree.add_child(parent_id, child_copy)
                        revealed_node_ids.append(child_copy.node_id)
                        total_revealed_attempts += 1
                        batch_revealed_any = True

                        budget_mgr.record_agent_call(
                            tokens=child_copy.tokens_used,
                            cost_usd=child_copy.cost_usd,
                            duration=child_copy.duration_seconds,
                        )
                        budget_mgr.update_best_score(child_copy.score)
                        policy.observe(child_copy)

            if not batch_revealed_any:
                # No new children could be revealed from the selected batch in this recorded history
                break

        duration = time.time() - start_time
        best_node = replay_tree.get_best_node()
        best_score = best_node.score if best_node else 0.0

        # Dream-RSI Objective V:
        # V = max(score) - beta1 * revealed_attempts + beta2 * parallelism
        parallelism = float(total_revealed_attempts) / float(max(1, decision_rounds))
        cfg = self.objective_config
        v_score = (
            cfg.quality_weight * best_score
            - cfg.beta1 * float(total_revealed_attempts)
            + cfg.beta2 * cfg.parallelism_weight * parallelism
            - cfg.cost_weight * budget_mgr.cost_usd
        )

        return ReplayResult(
            policy_id=policy.policy_id,
            revealed_node_ids=revealed_node_ids,
            best_node_id=best_node.node_id if best_node else None,
            best_score=best_score,
            total_attempts=total_revealed_attempts,
            decision_rounds=decision_rounds,
            parallelism=round(parallelism, 4),
            replay_score=round(v_score, 4),
            stopped_by_policy=stopped_by_policy,
            duration_seconds=duration,
        )
