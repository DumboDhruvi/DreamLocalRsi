"""Unit tests for CommandEvaluator, PytestEvaluator, and CompositeEvaluator."""

import os
import tempfile
import pytest
from dream_rsi.evaluators.command import CommandEvaluator
from dream_rsi.evaluators.pytest import PytestEvaluator
from dream_rsi.evaluators.composite import CompositeEvaluator


def test_command_evaluator():
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = CommandEvaluator({"command": "echo 'hello world'"})
        res = evaluator.evaluate(tmpdir)
        assert res.success
        assert res.score == 1.0
        assert "hello world" in res.stdout_artifact

        failing_eval = CommandEvaluator({"command": "exit 1"})
        res_fail = failing_eval.evaluate(tmpdir)
        assert not res_fail.success
        assert res_fail.score == 0.0


def test_pytest_evaluator():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_content = """
def test_ok():
    assert True

def test_fail():
    assert False
"""
        with open(os.path.join(tmpdir, "test_sample.py"), "w") as f:
            f.write(test_content)

        evaluator = PytestEvaluator({"extra_args": "-q"})
        res = evaluator.evaluate(tmpdir)

        assert not res.success  # because 1 failed
        assert res.tests_passed == 1
        assert res.tests_failed == 1
        assert res.total_tests == 2
        assert res.score == 0.5


def test_composite_evaluator():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "evaluators": [
                {"name": "cmd1", "type": "command", "weight": 0.5, "config": {"command": "true"}},
                {"name": "cmd2", "type": "command", "weight": 0.5, "config": {"command": "true"}},
            ]
        }
        composite = CompositeEvaluator(cfg)
        res = composite.evaluate(tmpdir)
        assert res.success
        assert res.score == 1.0

        # One failing sub-evaluator
        cfg["evaluators"][1]["config"]["command"] = "false"
        composite2 = CompositeEvaluator(cfg)
        res2 = composite2.evaluate(tmpdir)
        assert not res2.success
        assert res2.score == 0.5
