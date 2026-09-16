"""Unit tests for SQLiteStore and ArtifactManager."""

import os
import tempfile
import pytest
from dream_rsi.core.models import NodeStatus, Task, TreeNode
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.storage.artifacts import ArtifactManager


def test_sqlite_task_and_tree_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.sqlite")
        store = SQLiteStore(db_path)

        task = Task(
            task_id="t_001",
            instruction="Solve coding problem",
            workspace_root="/home/repo",
            evaluator_config={"type": "pytest"},
        )
        store.save_task(task)

        loaded_task = store.get_task("t_001")
        assert loaded_task is not None
        assert loaded_task.instruction == "Solve coding problem"

        tree = DiscoveryTree(tree_id="tree_persist", task_id="t_001")
        root = TreeNode(
            node_id="n_root",
            parent_id=None,
            task_id="t_001",
            strategy_id="init",
            instruction="Root instruction",
            workspace_snapshot="snap_root",
            score=0.3,
            status=NodeStatus.COMPLETED,
        )
        tree.add_root(root)

        child = TreeNode(
            node_id="n_child",
            parent_id="n_root",
            task_id="t_001",
            strategy_id="refine",
            instruction="Refine instruction",
            workspace_snapshot="snap_child",
            score=0.85,
            status=NodeStatus.COMPLETED,
        )
        tree.add_child("n_root", child)

        store.save_tree(tree)

        loaded_tree = store.load_tree("tree_persist")
        assert loaded_tree is not None
        assert loaded_tree.tree_id == "tree_persist"
        assert len(loaded_tree.nodes) == 2
        assert loaded_tree.get_best_node().node_id == "n_child"
        assert loaded_tree.get_best_node().score == 0.85


def test_artifact_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager(tmpdir)
        art_path = mgr.save_text_artifact("task_x", "node_y", "stdout.log", "some execution log")
        assert os.path.exists(art_path)
        content = mgr.get_text_artifact(art_path)
        assert content == "some execution log"
