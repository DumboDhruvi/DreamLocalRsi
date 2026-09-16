"""Benchmarking harness comparing exploration baselines against Dream-RSI policy improvement."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from dream_rsi.agents.base import AgentAdapter
from dream_rsi.core.budget import Budget
from dream_rsi.core.models import Task
from dream_rsi.core.orchestrator import Orchestrator
from dream_rsi.core.policy import ExplorationPolicy, PolicyConfig
from dream_rsi.dreaming.simulator import DreamingEngine
from dream_rsi.evaluators.base import BaseEvaluator
from dream_rsi.policies.adaptive import AdaptivePolicy
from dream_rsi.policies.evolved import EvolvedPolicy
from dream_rsi.policies.fixed import FixedPolicy
from dream_rsi.storage.artifacts import ArtifactManager
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager


@dataclass
class BenchmarkMetric:
    baseline_name: str
    best_score: float
    solved: bool
    agent_calls: int
    rounds_spent: int
    duration_seconds: float
    total_cost_usd: float
    failed_attempts: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Executes comparative evaluation across exploration baselines on a target coding task."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        evaluator: BaseEvaluator,
        agent: AgentAdapter,
        store: Optional[SQLiteStore] = None,
        artifact_mgr: Optional[ArtifactManager] = None,
    ):
        self.workspace_mgr = workspace_manager
        self.evaluator = evaluator
        self.agent = agent
        self.store = store
        self.artifact_mgr = artifact_mgr

    def run_all_baselines(
        self,
        task: Task,
        budget: Optional[Budget] = None,
    ) -> List[BenchmarkMetric]:
        common_budget = budget or Budget(max_rounds=6, max_agent_calls=12)
        results: List[BenchmarkMetric] = []

        # 1. Single-Agent Direct Solve (Single attempt)
        single_policy = FixedPolicy(
            "single_agent",
            PolicyConfig(branching_factor=1, max_depth=1, parallel_workers=1),
        )
        single_budget = Budget(max_rounds=1, max_agent_calls=1)
        res_single = self._run_single_baseline("Single-Agent Direct", task, single_policy, single_budget)
        results.append(res_single)

        # 2. Best-of-N (Broad root exploration, depth 1)
        best_of_n_policy = FixedPolicy(
            "best_of_n",
            PolicyConfig(branching_factor=4, max_depth=1, parallel_workers=2),
        )
        best_of_n_budget = Budget(max_rounds=2, max_agent_calls=4, max_parallel_workers=2)
        res_bon = self._run_single_baseline("Best-of-N (N=4)", task, best_of_n_policy, best_of_n_budget)
        results.append(res_bon)

        # 3. Fixed Exploration
        fixed_policy = FixedPolicy(
            "fixed_baseline",
            PolicyConfig(branching_factor=2, max_depth=4, parallel_workers=2),
        )
        res_fixed = self._run_single_baseline("Fixed Exploration", task, fixed_policy, common_budget)
        results.append(res_fixed)

        # 4. Adaptive UCB Policy
        ucb_policy = AdaptivePolicy(
            "adaptive_ucb",
            PolicyConfig(parallel_workers=2, exploration_constant=1.414),
        )
        res_ucb = self._run_single_baseline("Adaptive UCB", task, ucb_policy, common_budget)
        results.append(res_ucb)

        # 5. Dream-RSI Policy Improvement
        # Run offline dreaming on previous discovery trees to evolve an improved policy, then run online rollout
        dreaming_engine = DreamingEngine(store=self.store)
        dream_report = dreaming_engine.run_dreaming(
            current_policy=ucb_policy,
            num_candidates=8,
            budget=common_budget,
        )
        evolved_policy = dream_report.winning_summary.policy
        res_dream = self._run_single_baseline(
            f"Dream-RSI (Evolved: {evolved_policy.policy_id[:12]})",
            task,
            evolved_policy,
            common_budget,
        )
        results.append(res_dream)

        return results

    def _run_single_baseline(
        self,
        name: str,
        task: Task,
        policy: ExplorationPolicy,
        budget: Budget,
    ) -> BenchmarkMetric:
        orch = Orchestrator(
            workspace_manager=self.workspace_mgr,
            evaluator=self.evaluator,
            agent=self.agent,
            store=self.store,
            artifact_mgr=self.artifact_mgr,
        )

        start = time.time()
        tree = orch.run_exploration(task, policy, budget=budget)
        elapsed = time.time() - start

        best = tree.get_best_node()
        best_score = best.score if best else 0.0
        solved = best_score >= budget.stop_score_threshold

        attempts = len(tree.nodes) - 1  # excluding baseline root
        failed_count = sum(1 for n in tree.nodes.values() if not n.is_root and n.score < 0.5)
        total_cost = sum(n.cost_usd for n in tree.nodes.values())

        return BenchmarkMetric(
            baseline_name=name,
            best_score=round(best_score, 3),
            solved=solved,
            agent_calls=attempts,
            rounds_spent=max(1, tree.get_stats().get("max_depth", 1)),
            duration_seconds=round(elapsed, 2),
            total_cost_usd=round(total_cost, 4),
            failed_attempts=failed_count,
        )
