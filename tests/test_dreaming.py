"""Unit tests for offline policy dreaming, replay evaluation, and monotonic selection."""

import os
import tempfile
import pytest
from dream_rsi.core.budget import Budget
from dream_rsi.core.models import NodeStatus, TreeNode
from dream_rsi.core.policy import PolicyConfig
from dream_rsi.core.replay import ReplayObjectiveConfig
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.dreaming.policy_evaluator import PolicyEvaluator
from dream_rsi.dreaming.policy_generator import PolicyGenerator
from dream_rsi.dreaming.selector import PolicySelector
from dream_rsi.dreaming.simulator import DreamingEngine
from dream_rsi.policies.adaptive import AdaptivePolicy
from dream_rsi.policies.fixed import FixedPolicy
from dream_rsi.storage.sqlite import SQLiteStore


def create_mock_historical_tree() -> DiscoveryTree:
    tree = DiscoveryTree("hist_tree_1", "task_x")
    root = TreeNode("root", None, "task_x", "init", "prompt", "s0", score=0.1, status=NodeStatus.COMPLETED)
    tree.add_root(root)

    # Sub-branch 1 (poor score)
    c1 = TreeNode("c1", "root", "task_x", "s1", "p1", "s1", score=0.3, status=NodeStatus.COMPLETED)
    tree.add_child("root", c1)

    # Sub-branch 2 (high score)
    c2 = TreeNode("c2", "root", "task_x", "s2", "p2", "s2", score=0.9, status=NodeStatus.COMPLETED)
    tree.add_child("root", c2)

    # Refinement on c2 (perfect score)
    c2_1 = TreeNode("c2_1", "c2", "task_x", "s3", "p3", "s3", score=1.0, status=NodeStatus.COMPLETED)
    tree.add_child("c2", c2_1)
    return tree


def test_monotonicity_guarantee():
    current_policy = FixedPolicy("curr_base", PolicyConfig(parallel_workers=1))
    trees = [create_mock_historical_tree()]

    generator = PolicyGenerator(seed=123)
    candidates = generator.generate_candidates(current_policy, num_candidates=5)

    # Candidate 0 must be current_policy
    assert candidates[0].policy_id == current_policy.policy_id

    evaluator = PolicyEvaluator()
    summaries = [evaluator.evaluate_candidate(c, trees) for c in candidates]

    selector = PolicySelector(min_improvement_margin=0.0)
    selected, improved, win_sum = selector.select_best_policy(summaries[0], summaries)

    # Replay score of selected policy must be >= baseline
    assert win_sum.mean_replay_score >= summaries[0].mean_replay_score


def test_dreaming_engine_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "dream.sqlite")
        store = SQLiteStore(db_path)

        tree = create_mock_historical_tree()
        store.save_tree(tree)

        # Baseline policy: very poor settings (only 1 worker, low budget)
        base_policy = AdaptivePolicy("poor_baseline", PolicyConfig(parallel_workers=1, stop_when_score=0.5))

        engine = DreamingEngine(store=store)
        report = engine.run_dreaming(
            current_policy=base_policy,
            num_candidates=10,
            budget=Budget(max_rounds=5, max_agent_calls=10),
        )

        assert report.candidates_evaluated == 10
        # The selected policy replay score must be >= initial score
        assert report.selected_replay_score >= report.initial_replay_score
        assert report.selected_policy_id is not None
