"""Disk-based artifact manager for large logs, diffs, and prompts."""

from __future__ import annotations

import os
from typing import Optional


class ArtifactManager:
    """Stores and retrieves file artifacts on disk to keep SQLite lightweight."""

    def __init__(self, base_artifacts_dir: str):
        self.base_dir = os.path.abspath(base_artifacts_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def save_text_artifact(
        self,
        task_id: str,
        node_id: str,
        artifact_name: str,
        content: str,
    ) -> str:
        target_dir = os.path.join(self.base_dir, task_id, node_id)
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, artifact_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def get_text_artifact(self, file_path: str) -> Optional[str]:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
