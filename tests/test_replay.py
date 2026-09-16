"""Unit tests for deterministic ReplaySimulator and Dream-RSI objective."""

import pytest
from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.models import Budget, NodeStatus, TreeNode
from dream_rsi.core.policy import ExplorationPolicy, PolicyConfig
from dream_rsi.core.replay import ReplayObjectiveConfig, ReplaySimulator
from dream_rsi.core.tree import DiscoveryTree


class DummyPolicy(ExplorationPolicy):
    """Simple policy choosing root first, then the highest scoring leaf."""

    def choose_batch(self, tree: DiscoveryTree, budget_mgr: BudgetManager, max_workers: int):
        eligible = tree.get_eligible_nodes()
        if not eligible:
            return []
        # Sort by score descending
        sorted_nodes = sorted(eligible, key=lambda n: n.score, reverse=True)
        return [n.node_id for n in sorted_nodes[:max_workers]]


def build_recorded_tree() -> DiscoveryTree:
    tree = DiscoveryTree(tree_id="rec_tree", task_id="task_x")
    root = TreeNode(
        node_id="root",
        parent_id=None,
        task_id="task_x",
        strategy_id="init",
        instruction="Initial prompt",
        workspace_snapshot="snap_root",
        score=0.2,
        status=NodeStatus.COMPLETED,
    )
    tree.add_root(root)

    c1 = TreeNode(
        node_id="c1",
        parent_id="root",
        task_id="task_x",
        strategy_id="attempt_1",
        instruction="Branch 1",
        workspace_snapshot="snap_c1",
        score=0.6,
        status=NodeStatus.COMPLETED,
    )
    c2 = TreeNode(
        node_id="c2",
        parent_id="root",
        task_id="task_x",
        strategy_id="attempt_2",
        instruction="Branch 2",
        workspace_snapshot="snap_c2",
        score=0.85,
        status=NodeStatus.COMPLETED,
    )
    tree.add_child("root", c1)
    tree.add_child("root", c2)

    # Add child to c2
    c2_1 = TreeNode(
        node_id="c2_1",
        parent_id="c2",
        task_id="task_x",
        strategy_id="refine_c2",
        instruction="Refine Branch 2",
        workspace_snapshot="snap_c2_1",
        score=0.95,
        status=NodeStatus.COMPLETED,
    )
    tree.add_child("c2", c2_1)
    return tree


def test_replay_simulator_determinism_and_objective():
    tree = build_recorded_tree()
    policy = DummyPolicy(
        policy_id="test_pol",
        config=PolicyConfig(parallel_workers=2, stop_when_score=0.99),
    )

    obj_cfg = ReplayObjectiveConfig(beta1=0.01, beta2=0.02)
    sim = ReplaySimulator(tree, objective_config=obj_cfg)

    budget = Budget(max_rounds=5, max_agent_calls=10, max_parallel_workers=2)
    res1 = sim.simulate(policy, budget)
    res2 = sim.simulate(policy, budget)

    # Verify determinism
    assert res1.revealed_node_ids == res2.revealed_node_ids
    assert res1.best_score == res2.best_score == 0.95
    assert res1.best_node_id == "c2_1"
    assert res1.total_attempts == 3
    assert res1.replay_score == res2.replay_score

    # Verify Dream-RSI objective V calculation:
    # V = max(score) - beta1 * revealed_attempts + beta2 * parallelism
    expected_parallelism = 3.0 / float(res1.decision_rounds)
    expected_v = 0.95 - (0.01 * 3) + (0.02 * expected_parallelism)
    assert pytest.approx(res1.replay_score, 0.001) == expected_v


def test_replay_simulator_early_stop():
    tree = build_recorded_tree()
    # Stop when score >= 0.8
    policy = DummyPolicy(
        policy_id="early_stop_pol",
        config=PolicyConfig(stop_when_score=0.80, parallel_workers=2),
    )
    sim = ReplaySimulator(tree)
    res = sim.simulate(policy, Budget(max_rounds=10, max_agent_calls=10, max_parallel_workers=2))

    # Should stop once c2 (score 0.85) is revealed without revealing c2_1
    assert res.best_score >= 0.80
    assert res.stopped_by_policy
    assert "c2" in res.revealed_node_ids
    assert "c2_1" not in res.revealed_node_ids
