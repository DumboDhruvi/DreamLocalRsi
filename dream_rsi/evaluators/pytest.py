"""Pytest test runner and evaluation parser."""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Dict

from dream_rsi.core.models import EvaluationResult
from dream_rsi.evaluators.base import BaseEvaluator


class PytestEvaluator(BaseEvaluator):
    """Runs pytest inside a workspace and parses test metrics into an EvaluationResult."""

    def evaluate(self, workspace_path: str) -> EvaluationResult:
        test_target = self.config.get("test_target", "")
        extra_args = self.config.get("extra_args", "-q")
        timeout = float(self.config.get("timeout_seconds", 120.0))

        cmd = f"python3 -m pytest {extra_args} {test_target}".strip()
        start = time.time()

        try:
            res = subprocess.run(
                cmd,
                cwd=workspace_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start

            passed, failed, skipped = self._parse_pytest_output(res.stdout + "\n" + res.stderr)
            total = passed + failed + skipped

            # Handle no-test situations explicitly
            if total == 0:
                score = 0.0
                success = False
                error_msg = "No tests were collected or discovered."
            else:
                score = float(passed) / float(total)
                success = (failed == 0 and passed > 0)
                error_msg = None if success else f"{failed} test(s) failed."

            return EvaluationResult(
                success=success,
                score=round(score, 4),
                tests_passed=passed,
                tests_failed=failed,
                tests_skipped=skipped,
                total_tests=total,
                duration_seconds=duration,
                exit_code=res.returncode,
                metrics={
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "total": total,
                },
                stdout_artifact=res.stdout,
                stderr_artifact=res.stderr,
                error_message=error_msg,
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return EvaluationResult(
                success=False,
                score=0.0,
                duration_seconds=duration,
                exit_code=-1,
                error_message=f"Pytest timed out after {timeout} seconds",
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

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, int, int]:
        passed = 0
        failed = 0
        skipped = 0

        # Pattern matches e.g. "5 passed, 1 failed, 2 skipped in 0.12s" or "3 passed in 0.04s"
        match_passed = re.search(r"(\d+)\s+passed", output)
        if match_passed:
            passed = int(match_passed.group(1))

        match_failed = re.search(r"(\d+)\s+failed", output)
        if match_failed:
            failed = int(match_failed.group(1))

        match_skipped = re.search(r"(\d+)\s+skipped", output)
        if match_skipped:
            skipped = int(match_skipped.group(1))

        return passed, failed, skipped
