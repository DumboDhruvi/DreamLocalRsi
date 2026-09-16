"""Online exploration orchestrator executing agent attempts and growing the discovery tree."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from dream_rsi.agents.base import AgentAdapter
from dream_rsi.core.budget import BudgetManager
from dream_rsi.core.models import (
    Budget,
    NodeStatus,
    Task,
    TreeNode,
)
from dream_rsi.core.policy import ExplorationPolicy
from dream_rsi.core.tree import DiscoveryTree
from dream_rsi.evaluators.base import BaseEvaluator
from dream_rsi.storage.artifacts import ArtifactManager
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates online rollout: isolates workspaces, invokes coding agents, evaluates attempts, and updates the tree."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        evaluator: BaseEvaluator,
        agent: AgentAdapter,
        store: Optional[SQLiteStore] = None,
        artifact_mgr: Optional[ArtifactManager] = None,
    ):
        self.workspace_mgr = workspace_manager
        self.evaluator = evaluator
        self.agent = agent
        self.store = store
        self.artifact_mgr = artifact_mgr

    def run_exploration(
        self,
        task: Task,
        policy: ExplorationPolicy,
        budget: Optional[Budget] = None,
        tree_id: Optional[str] = None,
    ) -> DiscoveryTree:
        budget_obj = budget or Budget()
        budget_mgr = BudgetManager(budget_obj)
        tree_id = tree_id or f"tree_{uuid.uuid4().hex[:8]}"

        # Initialize tree
        tree = DiscoveryTree(tree_id=tree_id, task_id=task.task_id)

        # 1. Evaluate baseline workspace to create root node
        root_ws, root_snap = self.workspace_mgr.create_workspace("baseline_root")
        try:
            root_eval = self.evaluator.evaluate(root_ws)
            root_node = TreeNode(
                node_id=f"n_root_{uuid.uuid4().hex[:6]}",
                parent_id=None,
                task_id=task.task_id,
                strategy_id="initial_state",
                instruction="Evaluate baseline repository before exploration",
                workspace_snapshot=root_snap,
                score=root_eval.score,
                tests_passed=root_eval.tests_passed,
                tests_failed=root_eval.tests_failed,
                tests_skipped=root_eval.tests_skipped,
                duration_seconds=root_eval.duration_seconds,
                status=NodeStatus.COMPLETED if root_eval.success else NodeStatus.PENDING,
                summary=f"Baseline score: {root_eval.score}",
            )
            tree.add_root(root_node)
            budget_mgr.update_best_score(root_node.score)

            if self.store:
                self.store.save_task(task)
                self.store.save_tree(tree)
        finally:
            self.workspace_mgr.cleanup_workspace(root_ws)

        # 2. Online search loop
        round_idx = 0
        while budget_mgr.can_start_round():
            if policy.should_stop(tree, budget_mgr):
                logger.info("Policy signaled stopping condition.")
                break

            round_idx += 1
            budget_mgr.record_round()

            max_workers = budget_mgr.max_allowed_workers(policy.config.parallel_workers)
            if max_workers <= 0:
                break

            chosen_batch = policy.choose_batch(tree, budget_mgr, max_workers)
            if not chosen_batch:
                logger.info("No eligible nodes selected by policy.")
                break

            for parent_id in chosen_batch:
                if not budget_mgr.can_schedule_calls(1):
                    break

                parent_node = tree.get_node(parent_id)
                if not parent_node:
                    continue

                self._execute_attempt(
                    task=task,
                    tree=tree,
                    parent_node=parent_node,
                    policy=policy,
                    budget_mgr=budget_mgr,
                    round_idx=round_idx,
                )

        # Save run summary
        best_node = tree.get_best_node()
        if self.store:
            self.store.save_run(
                run_id=f"run_{uuid.uuid4().hex[:8]}",
                task_id=task.task_id,
                tree_id=tree.tree_id,
                policy_id=policy.policy_id,
                rounds_spent=budget_mgr.rounds_spent,
                agent_calls=budget_mgr.agent_calls,
                best_score=best_node.score if best_node else 0.0,
                total_cost_usd=budget_mgr.cost_usd,
                duration_seconds=budget_mgr.elapsed_seconds,
            )

        return tree

    def _execute_attempt(
        self,
        task: Task,
        tree: DiscoveryTree,
        parent_node: TreeNode,
        policy: ExplorationPolicy,
        budget_mgr: BudgetManager,
        round_idx: int,
    ) -> TreeNode:
        attempt_id = f"n_{uuid.uuid4().hex[:6]}"
        workspace_path, snap_id = self.workspace_mgr.create_workspace(attempt_id)

        try:
            # Context given to agent
            strategy_id = f"refine_round_{round_idx}" if not parent_node.is_root else f"explore_branch_{round_idx}"
            context = {
                "parent_id": parent_node.node_id,
                "parent_score": parent_node.score,
                "strategy_id": strategy_id,
                "instruction": f"{task.instruction} (focus on improving score from {parent_node.score:.2f})",
            }

            # 1. Agent execution
            agent_res = self.agent.run(task, workspace_path, context=context)

            # 2. Diff and changed files
            changed_files = self.workspace_mgr.get_changed_files(workspace_path)
            diff_text = self.workspace_mgr.get_diff(workspace_path)

            # 3. Objective evaluation
            eval_res = self.evaluator.evaluate(workspace_path)

            # 4. Save artifacts if manager is present
            artifacts: Dict[str, str] = {}
            if self.artifact_mgr and diff_text:
                diff_path = self.artifact_mgr.save_text_artifact(
                    task.task_id, attempt_id, "patch.diff", diff_text
                )
                artifacts["diff"] = diff_path

            if self.artifact_mgr and eval_res.stdout_artifact:
                stdout_path = self.artifact_mgr.save_text_artifact(
                    task.task_id, attempt_id, "eval_stdout.log", eval_res.stdout_artifact
                )
                artifacts["stdout"] = stdout_path

            # 5. Create Child TreeNode
            child_node = TreeNode(
                node_id=attempt_id,
                parent_id=parent_node.node_id,
                task_id=task.task_id,
                strategy_id=strategy_id,
                instruction=context["instruction"],
                workspace_snapshot=snap_id,
                changed_files=changed_files,
                artifacts=artifacts,
                score=eval_res.score,
                tests_passed=eval_res.tests_passed,
                tests_failed=eval_res.tests_failed,
                tests_skipped=eval_res.tests_skipped,
                duration_seconds=agent_res.duration_seconds + eval_res.duration_seconds,
                tokens_used=sum(agent_res.token_usage.values()),
                cost_usd=agent_res.cost_usd,
                agent_id=getattr(self.agent, "name", "agent"),
                status=NodeStatus.COMPLETED if (agent_res.success and eval_res.success) else NodeStatus.PENDING,
                error_info=agent_res.error or eval_res.error_message,
                summary=agent_res.text_summary,
            )

            # 6. Update tree and budget
            tree.add_child(parent_node.node_id, child_node)
            budget_mgr.record_agent_call(
                tokens=child_node.tokens_used,
                cost_usd=child_node.cost_usd,
                duration=child_node.duration_seconds,
            )
            budget_mgr.update_best_score(child_node.score)

            if self.store:
                self.store.save_node(tree.tree_id, child_node)

            policy.observe(child_node)
            return child_node

        finally:
            self.workspace_mgr.cleanup_workspace(workspace_path)
