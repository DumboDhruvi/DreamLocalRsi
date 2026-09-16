"""Generic CLI agent adapter executing command-line coding agents."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, Optional

from dream_rsi.agents.base import AgentAdapter
from dream_rsi.core.models import AgentResult, Task


class GenericCLIAgent(AgentAdapter):
    """Executes a CLI coding agent (e.g., gemini, claude, codex, or bash script) in an isolated workspace."""

    def run(
        self,
        task: Task,
        workspace_path: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ) -> AgentResult:
        command_template = self.config.get("command", "echo 'No agent command configured'")
        instruction = (context or {}).get("instruction", task.instruction)

        # Environment setup
        env = os.environ.copy()
        env["DREAM_RSI_TASK_ID"] = task.task_id
        env["DREAM_RSI_WORKSPACE"] = workspace_path
        env["DREAM_RSI_INSTRUCTION"] = instruction

        start_time = time.time()
        try:
            res = subprocess.run(
                command_template,
                cwd=workspace_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration = time.time() - start_time
            success = res.returncode == 0

            return AgentResult(
                success=success,
                text_summary=res.stdout[:1000] if res.stdout else "",
                changed_files=[],  # workspace manager will track modified files
                token_usage={"total_tokens": 0},
                cost_usd=0.0,
                duration_seconds=duration,
                raw_output_path=None,
                error=None if success else (res.stderr[:500] or f"Exit code {res.returncode}"),
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return AgentResult(
                success=False,
                text_summary="Agent timed out",
                duration_seconds=duration,
                error=f"Agent execution timed out after {timeout} seconds",
            )
        except Exception as e:
            duration = time.time() - start_time
            return AgentResult(
                success=False,
                text_summary=f"Agent failed with error: {e}",
                duration_seconds=duration,
                error=str(e),
            )
