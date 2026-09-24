"""Model Context Protocol (MCP) STDIO server for Dream-RSI."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any, Dict, Optional

from dream_rsi.agents.generic_cli import GenericCLIAgent
from dream_rsi.agents.mock import MockAgent
from dream_rsi.benchmark import BenchmarkRunner
from dream_rsi.core.budget import Budget
from dream_rsi.core.models import Task
from dream_rsi.core.orchestrator import Orchestrator
from dream_rsi.core.policy import PolicyConfig
from dream_rsi.dreaming.simulator import DreamingEngine
from dream_rsi.evaluators.command import CommandEvaluator
from dream_rsi.evaluators.composite import CompositeEvaluator
from dream_rsi.evaluators.pytest import PytestEvaluator
from dream_rsi.policies.adaptive import AdaptivePolicy
from dream_rsi.storage.artifacts import ArtifactManager
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager


def _get_store(db_path: str = ".dream_rsi/dream_rsi.sqlite") -> SQLiteStore:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return SQLiteStore(db_path)


TOOLS = [
    {
        "name": "dream_rsi_start_task",
        "description": "Launches online exploration on a target coding problem with automated sandboxing, branch rollout, and test evaluation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Instruction describing the coding task to solve.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Root path of target project repository (default: '.').",
                },
                "agent": {
                    "type": "string",
                    "description": "Coding agent type: mock, gemini, claude, codex, or command (default: 'mock').",
                },
                "evaluator": {
                    "type": "string",
                    "description": "Evaluator type: pytest, command, composite (default: 'pytest').",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Maximum decision rounds (default: 10).",
                },
                "max_calls": {
                    "type": "integer",
                    "description": "Maximum coding agent invocations (default: 40).",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "dream_rsi_get_tree",
        "description": "Inspects the active discovery tree, branch hierarchy, node scores, and exploration statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tree_id": {
                    "type": "string",
                    "description": "Discovery tree ID to visualize (defaults to latest tree).",
                }
            },
        },
    },
    {
        "name": "dream_rsi_get_best_attempt",
        "description": "Retrieves the highest-scoring attempt, evaluation metrics, and modified file list from a discovery tree.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tree_id": {
                    "type": "string",
                    "description": "Tree ID to find best attempt from (defaults to latest tree).",
                }
            },
        },
    },
    {
        "name": "dream_rsi_run_dreaming",
        "description": "Triggers offline replay dreaming across historical discovery trees to simulate candidate policies and evolve the exploration policy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "integer",
                    "description": "Number of candidate policies to generate and simulate (default: 16).",
                },
                "margin": {
                    "type": "number",
                    "description": "Minimum improvement margin required to adopt a candidate (default: 0.001).",
                },
            },
        },
    },
    {
        "name": "dream_rsi_benchmark",
        "description": "Runs a comparative benchmark against baseline exploration strategies (Single-Agent, Best-of-N, Fixed, Adaptive UCB, Evolved).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description for the benchmark comparison.",
                },
                "workspace": {
                    "type": "string",
                    "description": "Workspace root path for benchmark runs (default: '.').",
                },
            },
        },
    },
    {
        "name": "dream_rsi_status",
        "description": "Show current Dream-RSI system status, database statistics, total trees/nodes, and active exploration policy.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def handle_tool_call(name: str, args: Dict[str, Any], config_path: str = "config.yaml") -> Dict[str, Any]:
    norm_name = name.replace("-", "_").replace(".", "_")

    if norm_name in ("dream_rsi_start_task", "start_task"):
        task_str = args.get("task")
        if not task_str:
            return {"status": "error", "message": "Missing required 'task' argument"}

        workspace = args.get("workspace", ".")
        agent = args.get("agent", "mock")
        evaluator = args.get("evaluator", "pytest")
        max_rounds = int(args.get("max_rounds", 10))
        max_calls = int(args.get("max_calls", 40))

        db_path = ".dream_rsi/dream_rsi.sqlite"
        art_path = ".dream_rsi/artifacts"
        os.makedirs(art_path, exist_ok=True)

        store = _get_store(db_path)
        art_mgr = ArtifactManager(art_path)
        ws_mgr = WorkspaceManager(workspace, storage_dir=".dream_rsi/workspaces")

        if evaluator == "command":
            eval_engine = CommandEvaluator({})
        elif evaluator == "composite":
            eval_engine = CompositeEvaluator({})
        else:
            eval_engine = PytestEvaluator({})

        if agent == "mock":
            agent_adapter = MockAgent()
        else:
            agent_adapter = GenericCLIAgent({"command": agent})

        active_pol_info = store.get_active_policy()
        if active_pol_info:
            pol_cfg = PolicyConfig.from_dict(active_pol_info["config"])
            policy = AdaptivePolicy(active_pol_info["policy_id"], pol_cfg)
        else:
            policy = AdaptivePolicy("default_policy", PolicyConfig())

        task_obj = Task(task_id=f"task_{abs(hash(task_str)) % 100000}", instruction=task_str, workspace_root=workspace)
        budget_obj = Budget(max_rounds=max_rounds, max_agent_calls=max_calls)

        orchestrator = Orchestrator(
            workspace_manager=ws_mgr,
            evaluator=eval_engine,
            agent=agent_adapter,
            store=store,
            artifact_mgr=art_mgr,
        )

        tree = orchestrator.run_exploration(task_obj, policy, budget=budget_obj)
        stats = tree.get_stats()
        best_node = tree.get_best_node()

        return {
            "status": "success",
            "tree_id": tree.tree_id,
            "task_id": tree.task_id,
            "stats": stats,
            "best_node": {
                "node_id": best_node.node_id if best_node else None,
                "score": best_node.score if best_node else 0.0,
                "tests_passed": best_node.tests_passed if best_node else 0,
                "tests_failed": best_node.tests_failed if best_node else 0,
                "changed_files": best_node.changed_files if best_node else [],
                "strategy": best_node.strategy_id if best_node else None,
            },
        }

    elif norm_name in ("dream_rsi_get_tree", "get_tree"):
        tree_id = args.get("tree_id")
        store = _get_store()
        trees = store.get_all_trees()
        if not trees:
            return {"status": "error", "message": "No discovery trees found in database"}

        target_tree = store.load_tree(tree_id) if tree_id else trees[-1]
        if not target_tree:
            return {"status": "error", "message": f"Tree '{tree_id}' not found"}

        return {
            "status": "success",
            "tree_id": target_tree.tree_id,
            "task_id": target_tree.task_id,
            "stats": target_tree.get_stats(),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "score": n.score,
                    "status": n.status.value,
                    "strategy_id": n.strategy_id,
                    "changed_files": n.changed_files,
                }
                for n in target_tree.nodes.values()
            ],
        }

    elif norm_name in ("dream_rsi_get_best_attempt", "get_best_attempt"):
        tree_id = args.get("tree_id")
        store = _get_store()
        trees = store.get_all_trees()
        if not trees:
            return {"status": "error", "message": "No discovery trees found in database"}

        target_tree = store.load_tree(tree_id) if tree_id else trees[-1]
        if not target_tree:
            return {"status": "error", "message": f"Tree '{tree_id}' not found"}

        best_node = target_tree.get_best_node()
        if not best_node:
            return {"status": "error", "message": f"No nodes found in tree '{target_tree.tree_id}'"}

        return {
            "status": "success",
            "tree_id": target_tree.tree_id,
            "best_node": {
                "node_id": best_node.node_id,
                "score": best_node.score,
                "strategy_id": best_node.strategy_id,
                "tests_passed": best_node.tests_passed,
                "tests_failed": best_node.tests_failed,
                "changed_files": best_node.changed_files,
                "artifacts": best_node.artifacts,
                "summary": best_node.summary,
            },
        }

    elif norm_name in ("dream_rsi_run_dreaming", "run_dreaming"):
        candidates = int(args.get("candidates", 16))
        margin = float(args.get("margin", 0.001))
        store = _get_store()
        trees = store.get_all_trees()
        if not trees:
            return {
                "status": "error",
                "message": "No discovery trees found in database. Run online exploration first to generate histories.",
            }

        active_pol_info = store.get_active_policy()
        if active_pol_info:
            pol_cfg = PolicyConfig.from_dict(active_pol_info["config"])
            curr_policy = AdaptivePolicy(active_pol_info["policy_id"], pol_cfg)
        else:
            curr_policy = AdaptivePolicy("baseline_policy", PolicyConfig())

        engine = DreamingEngine(store=store)
        engine.selector.min_improvement_margin = margin
        report = engine.run_dreaming(current_policy=curr_policy, trees=trees, num_candidates=candidates)

        win = report.winning_summary
        return {
            "status": "success",
            "candidates_evaluated": report.candidates_evaluated,
            "initial_replay_score": report.initial_replay_score,
            "selected_replay_score": report.selected_replay_score,
            "improved": report.improved,
            "selected_policy_id": report.selected_policy_id,
            "winning_summary": {
                "mean_parallelism": win.mean_parallelism,
                "mean_attempts": win.mean_attempts,
                "mean_best_score": win.mean_best_score,
            },
        }

    elif norm_name in ("dream_rsi_benchmark", "benchmark"):
        task_str = args.get("task", "Run baseline comparison")
        workspace = args.get("workspace", ".")
        db_path = ".dream_rsi/dream_rsi.sqlite"
        art_path = ".dream_rsi/artifacts"
        os.makedirs(art_path, exist_ok=True)

        store = _get_store(db_path)
        art_mgr = ArtifactManager(art_path)
        ws_mgr = WorkspaceManager(workspace)
        evaluator = PytestEvaluator({})
        agent = MockAgent()

        runner = BenchmarkRunner(
            workspace_manager=ws_mgr,
            evaluator=evaluator,
            agent=agent,
            store=store,
            artifact_mgr=art_mgr,
        )

        task_obj = Task(task_id="bench_task", instruction=task_str, workspace_root=workspace)
        metrics = runner.run_all_baselines(task_obj)

        return {
            "status": "success",
            "metrics": [
                {
                    "baseline_name": m.baseline_name,
                    "best_score": m.best_score,
                    "solved": m.solved,
                    "agent_calls": m.agent_calls,
                    "duration_seconds": m.duration_seconds,
                }
                for m in metrics
            ],
        }

    elif norm_name in ("dream_rsi_status", "status"):
        store = _get_store()
        trees = store.get_all_trees()
        active_pol = store.get_active_policy()
        total_nodes = sum(len(t.nodes) for t in trees)

        return {
            "status": "success",
            "database": store.db_path,
            "total_trees": len(trees),
            "total_nodes": total_nodes,
            "active_policy": active_pol,
        }

    else:
        return {"status": "error", "message": f"Unknown tool '{name}'"}


def run_stdio_server(config_path: str = "config.yaml"):
    """STDIO JSON-RPC 2.0 loop for Model Context Protocol compliance."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "initialize":
                protocol_version = params.get("protocolVersion", "2024-11-05")
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": protocol_version,
                        "capabilities": {
                            "tools": {},
                        },
                        "serverInfo": {
                            "name": "dream-local-rsi",
                            "version": "0.1.0",
                        },
                    },
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": TOOLS,
                    },
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                # Redirect any stray stdout during tool execution to sys.stderr so stdio JSON-RPC remains clean
                with contextlib.redirect_stdout(sys.stderr):
                    result = handle_tool_call(tool_name, tool_args, config_path=config_path)

                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }
                        ],
                        "isError": result.get("status") == "error",
                        **result,
                    },
                }
            elif method == "ping":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


def main():
    run_stdio_server()


if __name__ == "__main__":
    main()
