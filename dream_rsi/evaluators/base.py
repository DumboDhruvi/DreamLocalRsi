"""Base evaluator interface."""

from __future__ import annotations

import abc
from typing import Any, Dict

from dream_rsi.core.models import EvaluationResult


class BaseEvaluator(abc.ABC):
    """Abstract base class for all task evaluators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def evaluate(self, workspace_path: str) -> EvaluationResult:
        """Runs evaluation inside the target workspace and returns an EvaluationResult."""
        pass
