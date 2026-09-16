"""Composite evaluator combining multiple weighted evaluation signals."""

from __future__ import annotations

from typing import Any, Dict, List

from dream_rsi.core.models import EvaluationResult
from dream_rsi.evaluators.base import BaseEvaluator
from dream_rsi.evaluators.command import CommandEvaluator
from dream_rsi.evaluators.pytest import PytestEvaluator
from dream_rsi.evaluators.benchmark import BenchmarkEvaluator


class CompositeEvaluator(BaseEvaluator):
    """Combines test results, static analysis, benchmarks, and regression checks with custom weights."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sub_evaluators: List[tuple[float, BaseEvaluator, str]] = []
        evaluators_def = self.config.get("evaluators", [])

        for item in evaluators_def:
            weight = float(item.get("weight", 1.0))
            eval_type = item.get("type", "command")
            sub_cfg = item.get("config", {})
            name = item.get("name", eval_type)

            if eval_type == "pytest":
                evaluator = PytestEvaluator(sub_cfg)
            elif eval_type == "benchmark":
                evaluator = BenchmarkEvaluator(sub_cfg)
            else:
                evaluator = CommandEvaluator(sub_cfg)

            self.sub_evaluators.append((weight, evaluator, name))

    def evaluate(self, workspace_path: str) -> EvaluationResult:
        if not self.sub_evaluators:
            return EvaluationResult(
                success=True,
                score=1.0,
                metrics={"warning": "No sub-evaluators configured in composite evaluator"},
            )

        total_weight = sum(w for w, _, _ in self.sub_evaluators)
        if total_weight <= 0:
            total_weight = 1.0

        aggregated_score = 0.0
        all_success = True
        total_duration = 0.0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_tests = 0
        sub_metrics = {}

        for weight, evaluator, name in self.sub_evaluators:
            res = evaluator.evaluate(workspace_path)
            normalized_weight = weight / total_weight
            aggregated_score += res.score * normalized_weight

            if not res.success:
                all_success = False

            total_duration += res.duration_seconds
            total_passed += res.tests_passed
            total_failed += res.tests_failed
            total_skipped += res.tests_skipped
            total_tests += res.total_tests

            sub_metrics[name] = {
                "score": res.score,
                "weight": weight,
                "success": res.success,
                "error": res.error_message,
            }

        return EvaluationResult(
            success=all_success,
            score=round(aggregated_score, 4),
            tests_passed=total_passed,
            tests_failed=total_failed,
            tests_skipped=total_skipped,
            total_tests=total_tests,
            duration_seconds=total_duration,
            metrics={"composite_breakdown": sub_metrics},
        )
