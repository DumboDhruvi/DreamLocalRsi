"""Unit tests for core models and budget manager."""

import pytest
from dream_rsi.core.models import (
    NodeStatus,
    TreeNode,
    Task,
    Budget,
    AgentResult,
    EvaluationResult,
    ReplayResult,
)
from dream_rsi.core.budget import BudgetManager


def test_task_serialization():
    task = Task(
        task_id="task_1",
        instruction="Fix the bug",
        workspace_root="/tmp/test",
        evaluator_config={"type": "pytest"},
    )
    d = task.to_dict()
    restored = Task.from_dict(d)
    assert restored.task_id == "task_1"
    assert restored.instruction == "Fix the bug"


def test_tree_node_serialization():
    node = TreeNode(
        node_id="n_01",
        parent_id=None,
        task_id="task_1",
        strategy_id="initial_baseline",
        instruction="Analyze issue",
        workspace_snapshot="snap_01",
        changed_files=["app.py"],
        score=0.75,
        status=NodeStatus.COMPLETED,
    )
    assert node.is_root
    assert node.is_leaf

    d = node.to_dict()
    assert d["status"] == "completed"
    restored = TreeNode.from_dict(d)
    assert restored.node_id == "n_01"
    assert restored.status == NodeStatus.COMPLETED
    assert restored.score == 0.75


def test_budget_manager_enforcement():
    budget = Budget(
        max_rounds=2,
        max_agent_calls=3,
        max_parallel_workers=2,
        max_duration_seconds=100.0,
        max_cost_usd=5.0,
        stop_score_threshold=0.95,
    )
    mgr = BudgetManager(budget)

    assert mgr.can_start_round()
    assert mgr.can_schedule_calls(2)

    # Record 1 call
    mgr.record_agent_call(tokens=100, cost_usd=1.0)
    assert mgr.agent_calls == 1
    assert mgr.max_allowed_workers() == 2

    # Record 2 more calls -> reaches 3 calls limit
    mgr.record_agent_call(tokens=200, cost_usd=2.0)
    mgr.record_agent_call(tokens=150, cost_usd=1.5)
    assert not mgr.can_schedule_calls(1)
    assert mgr.is_exhausted()


def test_budget_stop_score_threshold():
    budget = Budget(stop_score_threshold=0.90)
    mgr = BudgetManager(budget)
    assert mgr.can_start_round()
    mgr.update_best_score(0.95)
    assert not mgr.can_start_round()
