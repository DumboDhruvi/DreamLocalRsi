"""End-to-End integration test validating the complete Dream-RSI self-improvement loop:

Online Discovery -> Discovery Tree -> Replay Simulator -> Offline Dreaming -> Improved Policy -> Next Rollout.
"""

import os
import tempfile
import pytest
from dream_rsi.agents.mock import MockAgent
from dream_rsi.core.budget import Budget
from dream_rsi.core.models import Task
from dream_rsi.core.orchestrator import Orchestrator
from dream_rsi.core.policy import PolicyConfig
from dream_rsi.dreaming.simulator import DreamingEngine
from dream_rsi.evaluators.pytest import PytestEvaluator
from dream_rsi.policies.adaptive import AdaptivePolicy
from dream_rsi.policies.fixed import FixedPolicy
from dream_rsi.storage.artifacts import ArtifactManager
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager


def test_dream_rsi_end_to_end_loop():
    with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as env_dir:
        # 1. Setup a tiny Python project with 2 deliberate bugs
        math_py = os.path.join(repo_dir, "math_ops.py")
        with open(math_py, "w") as f:
            f.write(
                "def multiply(a, b):\n"
                "    return a + b  # Bug 1\n\n"
                "def divide(a, b):\n"
                "    return 0  # Bug 2\n"
            )

        test_math_py = os.path.join(repo_dir, "test_math_ops.py")
        with open(test_math_py, "w") as f:
            f.write(
                "from math_ops import multiply, divide\n"
                "def test_multiply():\n"
                "    assert multiply(3, 4) == 12\n\n"
                "def test_divide():\n"
                "    assert divide(10, 2) == 5\n"
            )

        ws_mgr = WorkspaceManager(repo_dir, storage_dir=os.path.join(env_dir, "workspaces"))
        evaluator = PytestEvaluator({"test_target": "test_math_ops.py", "extra_args": "-q"})
        agent = MockAgent()

        # Define strategy 1: fixes multiply only -> score 0.5
        def fix_multiply(ws_path):
            p = os.path.join(ws_path, "math_ops.py")
            with open(p, "w") as f:
                f.write(
                    "def multiply(a, b):\n"
                    "    return a * b  # Fixed\n\n"
                    "def divide(a, b):\n"
                    "    return 0  # Still bug\n"
                )
            return ["math_ops.py"]

        # Define strategy 2 (refinement): fixes both multiply and divide -> score 1.0
        def fix_both(ws_path):
            p = os.path.join(ws_path, "math_ops.py")
            with open(p, "w") as f:
                f.write(
                    "def multiply(a, b):\n"
                    "    return a * b  # Fixed\n\n"
                    "def divide(a, b):\n"
                    "    return a / b  # Fixed\n"
                )
            return ["math_ops.py"]

        agent.register_strategy("explore_branch_1", fix_multiply)
        agent.register_strategy("refine_round_2", fix_both)

        db_path = os.path.join(env_dir, "dream.sqlite")
        store = SQLiteStore(db_path)
        art_mgr = ArtifactManager(os.path.join(env_dir, "artifacts"))

        orch = Orchestrator(
            workspace_manager=ws_mgr,
            evaluator=evaluator,
            agent=agent,
            store=store,
            artifact_mgr=art_mgr,
        )

        task = Task("math_task", "Fix all bugs in math_ops.py", repo_dir)

        # -------------------------------------------------------------
        # Step 1: Initial Online Discovery with baseline FixedPolicy
        # -------------------------------------------------------------
        initial_policy = FixedPolicy(
            "initial_fixed",
            PolicyConfig(branching_factor=1, parallel_workers=1, stop_when_score=0.95),
        )
        first_budget = Budget(max_rounds=2, max_agent_calls=2)

        first_tree = orch.run_exploration(task, initial_policy, budget=first_budget)
        assert first_tree.root_id is not None
        assert first_tree.get_root().score == 0.0  # Baseline: 0/2 passed
        assert len(first_tree.nodes) >= 2  # Root + attempts

        # Best attempt in round 1
        best_round1 = first_tree.get_best_node()
        assert best_round1 is not None

        # -------------------------------------------------------------
        # Step 2: Offline Dreaming across recorded Discovery Tree
        # -------------------------------------------------------------
        # Simulate alternative exploration policies without invoking any coding agent
        dreaming_engine = DreamingEngine(store=store)
        dream_report = dreaming_engine.run_dreaming(
            current_policy=initial_policy,
            num_candidates=12,
            budget=Budget(max_rounds=5, max_agent_calls=10, max_parallel_workers=2),
        )

        assert dream_report.candidates_evaluated == 12
        # Monotonicity check: Selected policy replay objective V >= initial policy replay score
        assert dream_report.selected_replay_score >= dream_report.initial_replay_score

        evolved_policy = dream_report.winning_summary.policy
        assert evolved_policy is not None

        # -------------------------------------------------------------
        # Step 3: Second Online Rollout with Selected / Evolved Policy
        # -------------------------------------------------------------
        second_budget = Budget(max_rounds=3, max_agent_calls=4, stop_score_threshold=1.0)
        second_tree = orch.run_exploration(task, evolved_policy, budget=second_budget)

        best_round2 = second_tree.get_best_node()
        assert best_round2 is not None
        assert best_round2.score == 1.0  # Fully solved all tests!
        assert best_round2.tests_passed == 2
        assert best_round2.tests_failed == 0
