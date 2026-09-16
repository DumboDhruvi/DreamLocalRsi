"""Dreaming engine orchestrating offline candidate generation, replay evaluation, and selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dream_rsi.core.models import Budget
from dream_rsi.core.policy import ExplorationPolicy
from dream_rsi.core.replay import ReplayObjectiveConfig
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.dreaming.policy_evaluator import CandidateEvaluationSummary, PolicyEvaluator
from dream_rsi.dreaming.policy_generator import PolicyGenerator
from dream_rsi.dreaming.selector import PolicySelector
from dream_rsi.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class DreamingReport:
    initial_policy_id: str
    selected_policy_id: str
    improved: bool
    initial_replay_score: float
    selected_replay_score: float
    candidates_evaluated: int
    winning_summary: CandidateEvaluationSummary
    all_summaries: List[CandidateEvaluationSummary] = field(default_factory=list)


class DreamingEngine:
    """Coordinates the offline Dream-RSI loop across accumulated discovery trees."""

    def __init__(
        self,
        store: Optional[SQLiteStore] = None,
        generator: Optional[PolicyGenerator] = None,
        evaluator: Optional[PolicyEvaluator] = None,
        selector: Optional[PolicySelector] = None,
    ):
        self.store = store
        self.generator = generator or PolicyGenerator()
        self.evaluator = evaluator or PolicyEvaluator()
        self.selector = selector or PolicySelector()

    def run_dreaming(
        self,
        current_policy: ExplorationPolicy,
        trees: Optional[List[DiscoveryTree]] = None,
        num_candidates: int = 16,
        budget: Optional[Budget] = None,
    ) -> DreamingReport:
        history_trees = trees or []
        if not history_trees and self.store:
            history_trees = self.store.get_all_trees()

        if not history_trees:
            logger.warning("No discovery trees available for offline dreaming replay.")
            empty_summary = CandidateEvaluationSummary(
                policy=current_policy,
                mean_replay_score=0.0,
                mean_best_score=0.0,
                mean_attempts=0.0,
                mean_parallelism=0.0,
                simulations_count=0,
            )
            return DreamingReport(
                initial_policy_id=current_policy.policy_id,
                selected_policy_id=current_policy.policy_id,
                improved=False,
                initial_replay_score=0.0,
                selected_replay_score=0.0,
                candidates_evaluated=0,
                winning_summary=empty_summary,
            )

        # 1. Generate candidate policies (candidate 0 is current_policy)
        candidates = self.generator.generate_candidates(
            current_policy=current_policy,
            num_candidates=num_candidates,
        )

        # 2. Evaluate all candidates over replay trees
        summaries = [
            self.evaluator.evaluate_candidate(cand, history_trees, budget=budget)
            for cand in candidates
        ]

        current_summary = summaries[0]

        # 3. Select best candidate with monotonicity guarantee
        selected_policy, improved, winning_summary = self.selector.select_best_policy(
            current_summary=current_summary,
            candidate_summaries=summaries,
        )

        # 4. Save to store if improved
        if self.store and improved:
            self.store.save_policy(
                policy_id=selected_policy.policy_id,
                name=f"evolved_{selected_policy.policy_id}",
                policy_type=selected_policy.__class__.__name__,
                config=selected_policy.config.to_dict(),
                active=True,
            )

        return DreamingReport(
            initial_policy_id=current_policy.policy_id,
            selected_policy_id=selected_policy.policy_id,
            improved=improved,
            initial_replay_score=current_summary.mean_replay_score,
            selected_replay_score=winning_summary.mean_replay_score,
            candidates_evaluated=len(summaries),
            winning_summary=winning_summary,
            all_summaries=summaries,
        )
