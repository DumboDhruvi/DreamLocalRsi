"""Unit tests for the Dream-RSI CLI interface."""

import os
import tempfile
from click.testing import CliRunner
from dream_rsi.cli import main


def test_cli_init_and_status():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Test init
        res_init = runner.invoke(main, ["init"])
        assert res_init.exit_code == 0
        assert os.path.exists("config.yaml")
        assert os.path.exists(".dream_rsi/dream_rsi.sqlite")

        # Test status
        res_status = runner.invoke(main, ["status"])
        assert res_status.exit_code == 0
        assert "Total Trees:" in res_status.output
        assert "Active Policy:" in res_status.output


def test_cli_run_and_tree_and_export():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Initialize
        runner.invoke(main, ["init"])

        # Create a dummy test file
        with open("test_sample.py", "w") as f:
            f.write("def test_dummy(): assert True\n")

        # Test run with mock agent
        res_run = runner.invoke(
            main,
            ["run", "--task", "Run dummy task", "--agent", "mock", "--evaluator", "pytest", "--max-rounds", "1", "--max-calls", "1"],
        )
        assert res_run.exit_code == 0
        assert "Exploration Summary" in res_run.output

        # Test tree
        res_tree = runner.invoke(main, ["tree"])
        assert res_tree.exit_code == 0
        assert "Discovery Tree:" in res_tree.output

        # Test best
        res_best = runner.invoke(main, ["best"])
        assert res_best.exit_code == 0
        assert "Best Attempt in Tree" in res_best.output

        # Test export
        res_export = runner.invoke(main, ["export", "-o", "export.json"])
        assert res_export.exit_code == 0
        assert os.path.exists("export.json")
