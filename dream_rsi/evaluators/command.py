"""Command evaluator that executes shell commands in a workspace."""

from __future__ import annotations

import subprocess
import time
from typing import Any, Dict

from dream_rsi.core.models import EvaluationResult
from dream_rsi.evaluators.base import BaseEvaluator


class CommandEvaluator(BaseEvaluator):
    """Executes a shell command and evaluates exit codes, stdout, and duration."""

    def evaluate(self, workspace_path: str) -> EvaluationResult:
        command = self.config.get("command", "true")
        timeout = float(self.config.get("timeout_seconds", 60.0))

        start = time.time()
        try:
            res = subprocess.run(
                command,
                cwd=workspace_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start
            success = res.returncode == 0
            score = 1.0 if success else 0.0

            return EvaluationResult(
                success=success,
                score=score,
                duration_seconds=duration,
                exit_code=res.returncode,
                metrics={"command": command, "stdout": res.stdout, "stderr": res.stderr},
                stdout_artifact=res.stdout,
                stderr_artifact=res.stderr,
            )
        except subprocess.TimeoutExpired as te:
            duration = time.time() - start
            return EvaluationResult(
                success=False,
                score=0.0,
                duration_seconds=duration,
                exit_code=-1,
                error_message=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            duration = time.time() - start
            return EvaluationResult(
                success=False,
                score=0.0,
                duration_seconds=duration,
                exit_code=-1,
                error_message=str(e),
            )
