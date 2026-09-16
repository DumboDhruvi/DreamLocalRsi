"""Antigravity agent adapter interface."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, Optional

from dream_rsi.agents.base import AgentAdapter
from dream_rsi.core.models import AgentResult, Task


class AntigravityAgent(AgentAdapter):
    """Adapter for Antigravity CLI / agent calls."""

    def run(
        self,
        task: Task,
        workspace_path: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ) -> AgentResult:
        instruction = (context or {}).get("instruction", task.instruction)
        cmd = self.config.get("command", "agy")

        start_time = time.time()
        try:
            # Invoking Antigravity CLI if available in environment
            res = subprocess.run(
                f"{cmd} run --prompt {subprocess.list2cmdline([instruction])}",
                cwd=workspace_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start_time
            success = res.returncode == 0
            return AgentResult(
                success=success,
                text_summary=res.stdout[:1000] if res.stdout else "",
                duration_seconds=duration,
                error=None if success else res.stderr[:500],
            )
        except Exception as e:
            return AgentResult(
                success=False,
                text_summary=f"Antigravity invocation error: {e}",
                duration_seconds=time.time() - start_time,
                error=str(e),
            )
