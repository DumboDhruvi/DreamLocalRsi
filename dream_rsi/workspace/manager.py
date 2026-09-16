"""Isolated workspace manager supporting Git worktrees and copy fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from typing import List, Optional, Tuple


class WorkspaceManager:
    """Provides isolated working directories for agent attempts using Git worktree or fallback copies."""

    def __init__(self, base_repo_dir: str, storage_dir: Optional[str] = None):
        self.base_repo_dir = os.path.abspath(base_repo_dir)
        self.storage_dir = (
            os.path.abspath(storage_dir)
            if storage_dir
            else os.path.join(self.base_repo_dir, ".dream_rsi", "workspaces")
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self._is_git = self._check_git_repo(self.base_repo_dir)

    @staticmethod
    def _check_git_repo(path: str) -> bool:
        git_dir = os.path.join(path, ".git")
        if os.path.isdir(git_dir) or os.path.isfile(git_dir):
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return res.returncode == 0 and "true" in res.stdout.strip()
            except Exception:
                return False
        return False

    def create_workspace(
        self, attempt_id: Optional[str] = None, base_ref: Optional[str] = None
    ) -> Tuple[str, str]:
        """Creates an isolated workspace.

        Returns (workspace_path, snapshot_identifier).
        """
        attempt_id = attempt_id or f"attempt_{uuid.uuid4().hex[:8]}"
        workspace_path = os.path.join(self.storage_dir, attempt_id)

        if self._is_git:
            # Check if repo has at least one commit
            has_commits = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=self.base_repo_dir,
                capture_output=True,
                check=False,
            ).returncode == 0

            if has_commits:
                ref = base_ref or "HEAD"
                cmd = ["git", "worktree", "add", "--detach", workspace_path, ref]
                res = subprocess.run(cmd, cwd=self.base_repo_dir, capture_output=True, text=True)
                if res.returncode == 0:
                    return workspace_path, f"worktree:{attempt_id}"

        # Fallback to isolated directory copy
        self._copy_directory(self.base_repo_dir, workspace_path)
        return workspace_path, f"copy:{attempt_id}"

    def _copy_directory(self, src: str, dst: str) -> None:
        def ignore_patterns(path, names):
            ignored = {".git", ".dream_rsi", "__pycache__", ".pytest_cache", ".venv", "venv"}
            return [n for n in names if n in ignored]

        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True, ignore=ignore_patterns)

    def get_changed_files(self, workspace_path: str) -> List[str]:
        """Identifies files modified, added, or untracked in the workspace."""
        if self._check_git_repo(workspace_path):
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                check=False,
            )
            files = []
            for line in res.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Line format: XY filename or XY filename -> newname
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    files.append(parts[1].split("->")[-1].strip())
            return sorted(list(set(files)))

        # Fallback diffing against base_repo_dir
        changed = []
        for root, _, filenames in os.walk(workspace_path):
            rel_dir = os.path.relpath(root, workspace_path)
            if rel_dir != "." and any(p.startswith(".") for p in rel_dir.split(os.sep)):
                continue
            for f in filenames:
                if f.startswith("."):
                    continue
                rel_path = f if rel_dir == "." else os.path.normpath(os.path.join(rel_dir, f))
                base_path = os.path.join(self.base_repo_dir, rel_path)
                curr_path = os.path.join(workspace_path, rel_path)
                if not os.path.exists(base_path):
                    changed.append(rel_path)
                else:
                    try:
                        with open(base_path, "rb") as fb, open(curr_path, "rb") as fc:
                            if fb.read() != fc.read():
                                changed.append(rel_path)
                    except Exception:
                        changed.append(rel_path)
        return sorted(list(set(changed)))

    def get_diff(self, workspace_path: str) -> str:
        """Returns unified diff of changes made in the workspace."""
        if self._check_git_repo(workspace_path):
            subprocess.run(["git", "add", "-N", "."], cwd=workspace_path, capture_output=True, check=False)
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                check=False,
            )
            return res.stdout
        return ""

    def snapshot_workspace(self, workspace_path: str, destination_dir: str) -> str:
        """Persists the workspace state into destination_dir for future replay/restore."""
        os.makedirs(destination_dir, exist_ok=True)
        self._copy_directory(workspace_path, destination_dir)
        return destination_dir

    def cleanup_workspace(self, workspace_path: str) -> None:
        """Safely removes an isolated workspace and any associated git worktree."""
        if not os.path.exists(workspace_path):
            return

        if self._is_git:
            subprocess.run(
                ["git", "worktree", "remove", "--force", workspace_path],
                cwd=self.base_repo_dir,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=self.base_repo_dir,
                capture_output=True,
                check=False,
            )

        if os.path.exists(workspace_path):
            shutil.rmtree(workspace_path, ignore_errors=True)
