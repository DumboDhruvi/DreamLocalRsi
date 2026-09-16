"""Base agent adapter interface."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from dream_rsi.core.models import AgentResult, Task


class AgentAdapter(abc.ABC):
    """Abstract adapter defining the coding agent execution interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abc.abstractmethod
    def run(
        self,
        task: Task,
        workspace_path: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ) -> AgentResult:
        """Executes the coding agent in workspace_path and returns AgentResult."""
        pass
