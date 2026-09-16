"""Benchmark evaluator measuring execution speed and throughput."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any, Dict

from dream_rsi.core.models import EvaluationResult
from dream_rsi.evaluators.base import BaseEvaluator


class BenchmarkEvaluator(BaseEvaluator):
    """Executes a benchmark and derives a score from latency or throughput metrics."""

    def evaluate(self, workspace_path: str) -> EvaluationResult:
        command = self.config.get("command", "python3 -m unittest")
        target_metric = self.config.get("metric", "throughput")  # throughput or latency
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

            # Search output for numeric score or metrics e.g. "SCORE: 0.85" or "OPS/SEC: 1200"
            score = 1.0 if success else 0.0
            match_score = re.search(r"(?:score|throughput|ops_sec):\s*([\d\.]+)", res.stdout, re.IGNORECASE)
            if match_score:
                val = float(match_score.group(1))
                max_ref = float(self.config.get("max_ref_value", 100.0))
                score = min(1.0, val / max_ref)

            return EvaluationResult(
                success=success,
                score=score,
                duration_seconds=duration,
                exit_code=res.returncode,
                metrics={"command": command, "stdout": res.stdout},
                stdout_artifact=res.stdout,
                stderr_artifact=res.stderr,
            )
        except Exception as e:
            return EvaluationResult(
                success=False,
                score=0.0,
                duration_seconds=time.time() - start,
                exit_code=-1,
                error_message=str(e),
            )
