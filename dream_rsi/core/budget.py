"""Budget management and resource constraint tracking."""

from __future__ import annotations

import time
from typing import Optional

from dream_rsi.core.models import Budget


class BudgetManager:
    """Tracks and enforces limits on agent calls, costs, wall-clock time, and replay rounds."""

    def __init__(self, budget: Optional[Budget] = None):
        self.budget = budget or Budget()
        self.rounds_spent: int = 0
        self.agent_calls: int = 0
        self.replay_simulations: int = 0
        self.tokens_used: int = 0
        self.cost_usd: float = 0.0
        self.start_time: float = time.time()
        self.best_score_observed: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def record_agent_call(
        self,
        tokens: int = 0,
        cost_usd: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        self.agent_calls += 1
        self.tokens_used += tokens
        self.cost_usd += cost_usd

    def record_round(self) -> None:
        self.rounds_spent += 1

    def record_replay_simulation(self) -> None:
        self.replay_simulations += 1

    def update_best_score(self, score: float) -> None:
        if score > self.best_score_observed:
            self.best_score_observed = score

    def can_start_round(self) -> bool:
        if self.rounds_spent >= self.budget.max_rounds:
            return False
        if self.agent_calls >= self.budget.max_agent_calls:
            return False
        if self.elapsed_seconds >= self.budget.max_duration_seconds:
            return False
        if self.cost_usd >= self.budget.max_cost_usd:
            return False
        if self.best_score_observed >= self.budget.stop_score_threshold:
            return False
        return True

    def can_schedule_calls(self, count: int = 1) -> bool:
        if self.agent_calls + count > self.budget.max_agent_calls:
            return False
        if self.elapsed_seconds >= self.budget.max_duration_seconds:
            return False
        if self.cost_usd >= self.budget.max_cost_usd:
            return False
        return True

    def max_allowed_workers(self, requested: Optional[int] = None) -> int:
        remaining_calls = max(0, self.budget.max_agent_calls - self.agent_calls)
        limit = min(self.budget.max_parallel_workers, remaining_calls)
        if requested is not None:
            limit = min(limit, requested)
        return max(0, limit)

    def is_exhausted(self) -> bool:
        return not self.can_start_round()

    def get_status_summary(self) -> str:
        return (
            f"Rounds: {self.rounds_spent}/{self.budget.max_rounds} | "
            f"Agent Calls: {self.agent_calls}/{self.budget.max_agent_calls} | "
            f"Cost: ${self.cost_usd:.2f}/${self.budget.max_cost_usd:.2f} | "
            f"Time: {self.elapsed_seconds:.1f}s/{self.budget.max_duration_seconds:.1f}s | "
            f"Best Score: {self.best_score_observed:.3f}"
        )
