"""Scriptable MockAgent for deterministic testing and reproducible simulation."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

from dream_rsi.agents.base import AgentAdapter
from dream_rsi.core.models import AgentResult, Task


class MockAgent(AgentAdapter):
    """Deterministic agent that executes pre-configured file modifications or responses."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        strategy_handlers: Optional[Dict[str, Callable[[str], List[str]]]] = None,
    ):
        super().__init__(config)
        self.strategy_handlers = strategy_handlers or {}
        self.call_count: int = 0
        self.recorded_calls: List[Dict[str, Any]] = []

    def register_strategy(self, strategy_id: str, handler: Callable[[str], List[str]]) -> None:
        """Register a handler function that takes workspace_path and returns modified file paths."""
        self.strategy_handlers[strategy_id] = handler

    def run(
        self,
        task: Task,
        workspace_path: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ) -> AgentResult:
        start_time = time.time()
        self.call_count += 1
        ctx = context or {}
        strategy_id = ctx.get("strategy_id", "default")
        instruction = ctx.get("instruction", task.instruction)

        self.recorded_calls.append({
            "task_id": task.task_id,
            "strategy_id": strategy_id,
            "instruction": instruction,
            "call_index": self.call_count,
        })

        changed_files = []
        if strategy_id in self.strategy_handlers:
            try:
                changed_files = self.strategy_handlers[strategy_id](workspace_path)
            except Exception as e:
                return AgentResult(
                    success=False,
                    text_summary=f"Strategy handler failed: {e}",
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                )

        duration = time.time() - start_time
        return AgentResult(
            success=True,
            text_summary=f"MockAgent executed strategy '{strategy_id}': {instruction}",
            changed_files=changed_files,
            tool_calls=[{"tool": "edit_file", "strategy": strategy_id}],
            token_usage={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
            cost_usd=0.005,
            duration_seconds=duration,
        )
