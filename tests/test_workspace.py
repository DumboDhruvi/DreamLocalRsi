"""Unit tests for WorkspaceManager isolation, diff tracking, and cleanup."""

import os
import subprocess
import tempfile
import pytest
from dream_rsi.workspace.manager import WorkspaceManager


def test_workspace_copy_fallback_isolation():
    with tempfile.TemporaryDirectory() as base_dir, tempfile.TemporaryDirectory() as storage_dir:
        # Create a sample file in base repo
        test_file = os.path.join(base_dir, "sample.py")
        with open(test_file, "w") as f:
            f.write("def foo(): return 1\n")

        ws_mgr = WorkspaceManager(base_dir, storage_dir=storage_dir)
        ws_path, snapshot_id = ws_mgr.create_workspace("attempt_1")

        assert os.path.exists(ws_path)
        assert os.path.exists(os.path.join(ws_path, "sample.py"))

        # Modify file inside isolated workspace
        with open(os.path.join(ws_path, "sample.py"), "w") as f:
            f.write("def foo(): return 2\n")

        # Verify base repo was not affected
        with open(test_file, "r") as f:
            assert f.read() == "def foo(): return 1\n"

        changed = ws_mgr.get_changed_files(ws_path)
        assert "sample.py" in changed

        # Cleanup
        ws_mgr.cleanup_workspace(ws_path)
        assert not os.path.exists(ws_path)


def test_workspace_git_worktree_isolation():
    with tempfile.TemporaryDirectory() as git_dir, tempfile.TemporaryDirectory() as storage_dir:
        # Init git repo with commit
        subprocess.run(["git", "init"], cwd=git_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=git_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=git_dir, check=True)

        sample = os.path.join(git_dir, "code.py")
        with open(sample, "w") as f:
            f.write("val = 10\n")
        subprocess.run(["git", "add", "."], cwd=git_dir, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=git_dir, check=True, capture_output=True)

        ws_mgr = WorkspaceManager(git_dir, storage_dir=storage_dir)
        ws_path, snap_id = ws_mgr.create_workspace("git_attempt_1")

        assert "worktree" in snap_id
        assert os.path.exists(os.path.join(ws_path, "code.py"))

        # Mutate inside worktree
        with open(os.path.join(ws_path, "code.py"), "w") as f:
            f.write("val = 20\n")

        changed = ws_mgr.get_changed_files(ws_path)
        assert "code.py" in changed

        diff = ws_mgr.get_diff(ws_path)
        assert "+val = 20" in diff

        ws_mgr.cleanup_workspace(ws_path)
        assert not os.path.exists(ws_path)
