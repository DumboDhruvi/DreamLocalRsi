"""Offline replay evaluation across historical discovery trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dream_rsi.core.models import Budget, ReplayResult
from dream_rsi.core.policy import ExplorationPolicy
from dream_rsi.core.replay import ReplayObjectiveConfig, ReplaySimulator
from dream_rsi.core.tree import DiscoveryTree


@dataclass
class CandidateEvaluationSummary:
    policy: ExplorationPolicy
    mean_replay_score: float  # Mean objective V
    mean_best_score: float
    mean_attempts: float
    mean_parallelism: float
    simulations_count: int
    results: List[ReplayResult] = field(default_factory=list)


class PolicyEvaluator:
    """Simulates a candidate policy across multiple historical trees and computes aggregate Dream-RSI objective."""

    def __init__(self, objective_config: Optional[ReplayObjectiveConfig] = None):
        self.objective_config = objective_config or ReplayObjectiveConfig()

    def evaluate_candidate(
        self,
        candidate: ExplorationPolicy,
        trees: List[DiscoveryTree],
        budget: Optional[Budget] = None,
    ) -> CandidateEvaluationSummary:
        if not trees:
            return CandidateEvaluationSummary(
                policy=candidate,
                mean_replay_score=0.0,
                mean_best_score=0.0,
                mean_attempts=0.0,
                mean_parallelism=0.0,
                simulations_count=0,
            )

        results: List[ReplayResult] = []
        for tree in trees:
            simulator = ReplaySimulator(tree, objective_config=self.objective_config)
            res = simulator.simulate(candidate, budget=budget)
            results.append(res)

        n = len(results)
        mean_v = sum(r.replay_score for r in results) / n
        mean_best = sum(r.best_score for r in results) / n
        mean_att = sum(r.total_attempts for r in results) / n
        mean_par = sum(r.parallelism for r in results) / n

        return CandidateEvaluationSummary(
            policy=candidate,
            mean_replay_score=round(mean_v, 4),
            mean_best_score=round(mean_best, 4),
            mean_attempts=round(mean_att, 2),
            mean_parallelism=round(mean_par, 3),
            simulations_count=n,
            results=results,
        )
