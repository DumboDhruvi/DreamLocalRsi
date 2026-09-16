"""Monotonic policy selector guaranteeing non-regressive exploration improvements."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from dream_rsi.core.policy import ExplorationPolicy
from dream_rsi.dreaming.policy_evaluator import CandidateEvaluationSummary

logger = logging.getLogger(__name__)


class PolicySelector:
    """Selects the best exploration policy from replay evaluations while enforcing monotonicity."""

    def __init__(self, min_improvement_margin: float = 0.0):
        self.min_improvement_margin = min_improvement_margin

    def select_best_policy(
        self,
        current_summary: CandidateEvaluationSummary,
        candidate_summaries: List[CandidateEvaluationSummary],
    ) -> Tuple[ExplorationPolicy, bool, CandidateEvaluationSummary]:
        """Compares all candidate summaries against current policy summary.

        Returns (selected_policy, improved, winning_summary).
        """
        current_score = current_summary.mean_replay_score
        best_summary = current_summary
        improved = False

        # Sort candidates descending by mean_replay_score
        sorted_candidates = sorted(
            candidate_summaries,
            key=lambda s: (s.mean_replay_score, s.mean_best_score),
            reverse=True,
        )

        for cand in sorted_candidates:
            margin = cand.mean_replay_score - current_score
            if margin > self.min_improvement_margin:
                best_summary = cand
                improved = True
                break

        if improved:
            logger.info(
                f"Adopting improved policy '{best_summary.policy.policy_id}' with replay score "
                f"{best_summary.mean_replay_score:.4f} (baseline was {current_score:.4f})."
            )
        else:
            logger.info(
                f"No candidate exceeded baseline replay score {current_score:.4f} + margin {self.min_improvement_margin}. "
                f"Retaining current policy '{current_summary.policy.policy_id}'."
            )

        return best_summary.policy, improved, best_summary
