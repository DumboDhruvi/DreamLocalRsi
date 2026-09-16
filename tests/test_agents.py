"""Unit tests for MockAgent and GenericCLIAgent."""

import os
import tempfile
import pytest
from dream_rsi.agents.mock import MockAgent
from dream_rsi.agents.generic_cli import GenericCLIAgent
from dream_rsi.core.models import Task


def test_mock_agent_strategy_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = MockAgent()

        def dummy_patch(ws_path):
            p = os.path.join(ws_path, "modified.py")
            with open(p, "w") as f:
                f.write("# modified\n")
            return ["modified.py"]

        agent.register_strategy("strat_patch", dummy_patch)

        task = Task(task_id="t1", instruction="Do work", workspace_root=tmpdir)
        res = agent.run(task, tmpdir, context={"strategy_id": "strat_patch"})

        assert res.success
        assert "modified.py" in res.changed_files
        assert agent.call_count == 1
        assert os.path.exists(os.path.join(tmpdir, "modified.py"))


def test_generic_cli_agent():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = GenericCLIAgent({"command": "echo 'running cli agent'"})
        task = Task(task_id="t2", instruction="Fix bug", workspace_root=tmpdir)
        res = agent.run(task, tmpdir)
        assert res.success
        assert "running cli agent" in res.text_summary
