"""Unit tests for BenchmarkRunner comparing exploration baselines."""

import os
import tempfile
import pytest
from dream_rsi.agents.mock import MockAgent
from dream_rsi.benchmark import BenchmarkRunner
from dream_rsi.core.budget import Budget
from dream_rsi.core.models import Task
from dream_rsi.evaluators.command import CommandEvaluator
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager


def test_benchmark_runner_baselines():
    with tempfile.TemporaryDirectory() as base_repo, tempfile.TemporaryDirectory() as storage_dir:
        # Create a file
        with open(os.path.join(base_repo, "app.py"), "w") as f:
            f.write("val = 1\n")

        ws_mgr = WorkspaceManager(base_repo, storage_dir=os.path.join(storage_dir, "ws"))
        evaluator = CommandEvaluator({"command": "true"})
        agent = MockAgent()
        db_path = os.path.join(storage_dir, "bench.sqlite")
        store = SQLiteStore(db_path)

        runner = BenchmarkRunner(
            workspace_manager=ws_mgr,
            evaluator=evaluator,
            agent=agent,
            store=store,
        )

        task = Task("t_bench", "Evaluate performance baselines", base_repo)
        budget = Budget(max_rounds=2, max_agent_calls=3)
        metrics = runner.run_all_baselines(task, budget=budget)

        assert len(metrics) == 5
        names = [m.baseline_name for m in metrics]
        assert any("Single-Agent" in n for n in names)
        assert any("Best-of-N" in n for n in names)
        assert any("Fixed" in n for n in names)
        assert any("Adaptive UCB" in n for n in names)
        assert any("Dream-RSI" in n for n in names)
