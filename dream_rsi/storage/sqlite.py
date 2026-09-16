"""SQLite storage layer for Dream-RSI experience memory and trees."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from dream_rsi.core.models import NodeStatus, Task, TreeNode
from dream_rsi.core.tree import DiscoveryTree


class SQLiteStore:
    """Relational SQLite persistence for discovery trees, nodes, evaluations, policies, and runs."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                instruction TEXT NOT NULL,
                workspace_root TEXT NOT NULL,
                evaluator_config TEXT,
                metadata TEXT,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS discovery_trees (
                tree_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                root_id TEXT,
                created_at REAL,
                FOREIGN KEY (task_id) REFERENCES tasks (task_id)
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                tree_id TEXT NOT NULL,
                parent_id TEXT,
                task_id TEXT NOT NULL,
                strategy_id TEXT,
                instruction TEXT,
                workspace_snapshot TEXT,
                changed_files TEXT,
                artifacts TEXT,
                score REAL,
                tests_passed INTEGER,
                tests_failed INTEGER,
                tests_skipped INTEGER,
                duration_seconds REAL,
                tokens_used INTEGER,
                cost_usd REAL,
                agent_id TEXT,
                timestamp REAL,
                status TEXT,
                error_info TEXT,
                summary TEXT,
                depth INTEGER,
                children_ids TEXT,
                metadata TEXT,
                FOREIGN KEY (tree_id) REFERENCES discovery_trees (tree_id)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                success INTEGER,
                score REAL,
                tests_passed INTEGER,
                tests_failed INTEGER,
                duration_seconds REAL,
                exit_code INTEGER,
                metrics TEXT,
                error_message TEXT,
                timestamp REAL,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            );

            CREATE TABLE IF NOT EXISTS policies (
                policy_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                config TEXT NOT NULL,
                active INTEGER DEFAULT 0,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS policy_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                config TEXT NOT NULL,
                replay_score REAL,
                description TEXT,
                created_at REAL,
                FOREIGN KEY (policy_id) REFERENCES policies (policy_id)
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                tree_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                rounds_spent INTEGER,
                agent_calls INTEGER,
                best_score REAL,
                total_cost_usd REAL,
                duration_seconds REAL,
                created_at REAL,
                FOREIGN KEY (task_id) REFERENCES tasks (task_id),
                FOREIGN KEY (tree_id) REFERENCES discovery_trees (tree_id)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at REAL,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            );
            """)
            conn.commit()

    def save_task(self, task: Task) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (task_id, instruction, workspace_root, evaluator_config, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.instruction,
                    task.workspace_root,
                    json.dumps(task.evaluator_config),
                    json.dumps(task.metadata),
                    task.created_at,
                ),
            )
            conn.commit()

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            return Task(
                task_id=row["task_id"],
                instruction=row["instruction"],
                workspace_root=row["workspace_root"],
                evaluator_config=json.loads(row["evaluator_config"] or "{}"),
                metadata=json.loads(row["metadata"] or "{}"),
                created_at=row["created_at"],
            )

    def save_tree(self, tree: DiscoveryTree) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO discovery_trees (tree_id, task_id, root_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tree.tree_id, tree.task_id, tree.root_id, time.time()),
            )
            for node in tree.nodes.values():
                self._save_node_txn(conn, tree.tree_id, node)
            conn.commit()

    def save_node(self, tree_id: str, node: TreeNode) -> None:
        with self._get_connection() as conn:
            self._save_node_txn(conn, tree_id, node)
            conn.commit()

    def _save_node_txn(self, conn: sqlite3.Connection, tree_id: str, node: TreeNode) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO nodes (
                node_id, tree_id, parent_id, task_id, strategy_id, instruction,
                workspace_snapshot, changed_files, artifacts, score, tests_passed,
                tests_failed, tests_skipped, duration_seconds, tokens_used, cost_usd,
                agent_id, timestamp, status, error_info, summary, depth, children_ids, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.node_id,
                tree_id,
                node.parent_id,
                node.task_id,
                node.strategy_id,
                node.instruction,
                node.workspace_snapshot,
                json.dumps(node.changed_files),
                json.dumps(node.artifacts),
                node.score,
                node.tests_passed,
                node.tests_failed,
                node.tests_skipped,
                node.duration_seconds,
                node.tokens_used,
                node.cost_usd,
                node.agent_id,
                node.timestamp,
                node.status.value,
                node.error_info,
                node.summary,
                node.depth,
                json.dumps(node.children_ids),
                json.dumps(node.metadata),
            ),
        )

    def load_tree(self, tree_id: str) -> Optional[DiscoveryTree]:
        with self._get_connection() as conn:
            tree_row = conn.execute(
                "SELECT * FROM discovery_trees WHERE tree_id = ?", (tree_id,)
            ).fetchone()
            if not tree_row:
                return None

            tree = DiscoveryTree(tree_id=tree_row["tree_id"], task_id=tree_row["task_id"])
            tree.root_id = tree_row["root_id"]

            node_rows = conn.execute(
                "SELECT * FROM nodes WHERE tree_id = ?", (tree_id,)
            ).fetchall()
            for r in node_rows:
                node = TreeNode(
                    node_id=r["node_id"],
                    parent_id=r["parent_id"],
                    task_id=r["task_id"],
                    strategy_id=r["strategy_id"],
                    instruction=r["instruction"],
                    workspace_snapshot=r["workspace_snapshot"],
                    changed_files=json.loads(r["changed_files"] or "[]"),
                    artifacts=json.loads(r["artifacts"] or "{}"),
                    score=r["score"],
                    tests_passed=r["tests_passed"],
                    tests_failed=r["tests_failed"],
                    tests_skipped=r["tests_skipped"],
                    duration_seconds=r["duration_seconds"],
                    tokens_used=r["tokens_used"],
                    cost_usd=r["cost_usd"],
                    agent_id=r["agent_id"],
                    timestamp=r["timestamp"],
                    status=NodeStatus(r["status"]),
                    error_info=r["error_info"],
                    summary=r["summary"],
                    depth=r["depth"],
                    children_ids=json.loads(r["children_ids"] or "[]"),
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                tree.nodes[node.node_id] = node
            return tree

    def get_all_trees(self) -> List[DiscoveryTree]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT tree_id FROM discovery_trees").fetchall()
            trees = []
            for r in rows:
                t = self.load_tree(r["tree_id"])
                if t:
                    trees.append(t)
            return trees

    def save_policy(
        self, policy_id: str, name: str, policy_type: str, config: Dict[str, Any], active: bool = False
    ) -> None:
        with self._get_connection() as conn:
            if active:
                conn.execute("UPDATE policies SET active = 0")
            conn.execute(
                """
                INSERT OR REPLACE INTO policies (policy_id, name, policy_type, config, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (policy_id, name, policy_type, json.dumps(config), 1 if active else 0, time.time()),
            )
            conn.commit()

    def get_active_policy(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM policies WHERE active = 1").fetchone()
            if not row:
                return None
            return {
                "policy_id": row["policy_id"],
                "name": row["name"],
                "policy_type": row["policy_type"],
                "config": json.loads(row["config"]),
                "created_at": row["created_at"],
            }

    def save_run(
        self,
        run_id: str,
        task_id: str,
        tree_id: str,
        policy_id: str,
        rounds_spent: int,
        agent_calls: int,
        best_score: float,
        total_cost_usd: float,
        duration_seconds: float,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, task_id, tree_id, policy_id, rounds_spent,
                    agent_calls, best_score, total_cost_usd, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    tree_id,
                    policy_id,
                    rounds_spent,
                    agent_calls,
                    best_score,
                    total_cost_usd,
                    duration_seconds,
                    time.time(),
                ),
            )
            conn.commit()
