"""Unit tests for online exploration Orchestrator."""

import os
import tempfile
import pytest
from dream_rsi.agents.mock import MockAgent
from dream_rsi.core.budget import Budget
from dream_rsi.core.models import Task
from dream_rsi.core.orchestrator import Orchestrator
from dream_rsi.evaluators.pytest import PytestEvaluator
from dream_rsi.policies.fixed import FixedPolicy
from dream_rsi.storage.artifacts import ArtifactManager
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager


def test_orchestrator_online_rollout():
    with tempfile.TemporaryDirectory() as base_repo, tempfile.TemporaryDirectory() as storage_dir:
        # Create a repo with a failing test
        code_path = os.path.join(base_repo, "calc.py")
        with open(code_path, "w") as f:
            f.write("def add(a, b): return a - b\n")  # deliberate bug

        test_path = os.path.join(base_repo, "test_calc.py")
        with open(test_path, "w") as f:
            f.write("from calc import add\ndef test_add(): assert add(2, 3) == 5\n")

        ws_mgr = WorkspaceManager(base_repo, storage_dir=os.path.join(storage_dir, "ws"))
        evaluator = PytestEvaluator({"test_target": "test_calc.py", "extra_args": "-q"})
        agent = MockAgent()

        # Mock agent strategy that fixes the bug
        def fix_strategy(ws_path):
            p = os.path.join(ws_path, "calc.py")
            with open(p, "w") as f:
                f.write("def add(a, b): return a + b\n")
            return ["calc.py"]

        agent.register_strategy("explore_branch_1", fix_strategy)

        db_path = os.path.join(storage_dir, "test.sqlite")
        store = SQLiteStore(db_path)
        art_mgr = ArtifactManager(os.path.join(storage_dir, "artifacts"))

        orchestrator = Orchestrator(
            workspace_manager=ws_mgr,
            evaluator=evaluator,
            agent=agent,
            store=store,
            artifact_mgr=art_mgr,
        )

        task = Task(
            task_id="task_calc",
            instruction="Fix add function in calc.py",
            workspace_root=base_repo,
        )

        policy = FixedPolicy("fixed_test")
        budget = Budget(max_rounds=2, max_agent_calls=2, stop_score_threshold=1.0)

        tree = orchestrator.run_exploration(task, policy, budget=budget)

        assert tree.root_id is not None
        root = tree.get_root()
        assert root.score == 0.0  # initial failing test

        best = tree.get_best_node()
        assert best is not None
        assert best.score == 1.0  # successfully solved!
        assert len(tree.nodes) >= 2

        # Verify saved in SQLite
        loaded_tree = store.load_tree(tree.tree_id)
        assert loaded_tree is not None
        assert len(loaded_tree.nodes) == len(tree.nodes)
