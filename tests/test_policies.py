"""Unit tests for FixedPolicy, AdaptivePolicy, and EvolvedPolicy."""

import pytest
from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.models import Budget, NodeStatus, TreeNode
from dream_rsi.core.policy import PolicyConfig
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.policies.fixed import FixedPolicy
from dream_rsi.policies.adaptive import AdaptivePolicy
from dream_rsi.policies.evolved import EvolvedPolicy


def build_test_tree() -> DiscoveryTree:
    tree = DiscoveryTree("test_tree", "task_1")
    root = TreeNode(
        node_id="root",
        parent_id=None,
        task_id="task_1",
        strategy_id="init",
        instruction="init",
        workspace_snapshot="s0",
        score=0.2,
        status=NodeStatus.COMPLETED,
    )
    tree.add_root(root)
    return tree


def test_fixed_policy_branching():
    tree = build_test_tree()
    budget_mgr = BudgetManager(Budget(max_rounds=5, max_agent_calls=10))
    policy = FixedPolicy("fixed_test", PolicyConfig(branching_factor=2, parallel_workers=2))

    batch = policy.choose_batch(tree, budget_mgr, max_workers=2)
    # Root has 0 children, branching factor is 2 -> chooses root twice
    assert batch == ["root", "root"]

    # Now simulate children being added
    c1 = TreeNode("c1", "root", "task_1", "strat1", "inst", "s1", score=0.6, status=NodeStatus.COMPLETED)
    c2 = TreeNode("c2", "root", "task_1", "strat2", "inst", "s2", score=0.8, status=NodeStatus.COMPLETED)
    tree.add_child("root", c1)
    tree.add_child("root", c2)

    # Now root has 2 children >= branching_factor 2. It should choose leaves sorted by score descending: [c2, c1]
    batch2 = policy.choose_batch(tree, budget_mgr, max_workers=2)
    assert batch2 == ["c2", "c1"]


def test_adaptive_policy_ucb_and_diversity():
    tree = build_test_tree()
    budget_mgr = BudgetManager(Budget(max_rounds=5, max_agent_calls=10))
    policy = AdaptivePolicy("ucb_test", PolicyConfig(parallel_workers=2, exploration_constant=2.0))

    # With only root, chooses root
    batch = policy.choose_batch(tree, budget_mgr, max_workers=1)
    assert batch == ["root"]

    # Add child
    c1 = TreeNode("c1", "root", "task_1", "strat1", "inst", "s1", score=0.7, status=NodeStatus.COMPLETED)
    tree.add_child("root", c1)
    policy.observe(c1)

    batch2 = policy.choose_batch(tree, budget_mgr, max_workers=2)
    assert len(batch2) == 2
    assert "c1" in batch2 or "root" in batch2
