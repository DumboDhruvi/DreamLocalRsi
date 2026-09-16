"""Command-line interface for the Dream-RSI coding harness."""

from __future__ import annotations

import json
import os
import sys
import yaml
import click

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
from dream_rsi.policies.fixed import FixedPolicy
from dream_rsi.storage.artifacts import ArtifactManager
from dream_rsi.storage.sqlite import SQLiteStore
from dream_rsi.workspace.manager import WorkspaceManager


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@click.group()
def main():
    """Dream-RSI: Recursive Self-Improvement Coding Harness (arXiv:2609.14858)."""
    pass


@main.command()
@click.option("--output", "-o", default="config.yaml", help="Path to write configuration file.")
def init(output: str):
    """Initialize .dream_rsi workspace, SQLite database, and default config."""
    os.makedirs(".dream_rsi/artifacts", exist_ok=True)
    os.makedirs(".dream_rsi/workspaces", exist_ok=True)
    db_path = ".dream_rsi/dream_rsi.sqlite"
    store = SQLiteStore(db_path)

    # Save default active policy
    default_policy = AdaptivePolicy("policy_v1", PolicyConfig(parallel_workers=2, branching_factor=2))
    store.save_policy(
        policy_id=default_policy.policy_id,
        name="initial_adaptive_ucb",
        policy_type="AdaptivePolicy",
        config=default_policy.config.to_dict(),
        active=True,
    )

    if not os.path.exists(output):
        default_cfg = {
            "agent": {"command": "mock", "timeout_seconds": 600},
            "exploration": {"max_rounds": 10, "max_agent_calls": 40, "parallel_workers": 2, "stop_when_score": 1.0},
            "dreaming": {"enabled": True, "candidates": 16, "min_improvement_margin": 0.001},
            "replay": {"beta1": 0.01, "beta2": 0.01},
            "evaluator": {"type": "pytest", "extra_args": "-q"},
            "storage": {"database": db_path, "artifacts": ".dream_rsi/artifacts", "workspaces": ".dream_rsi/workspaces"},
        }
        with open(output, "w", encoding="utf-8") as f:
            yaml.dump(default_cfg, f, default_flow_style=False)
        click.echo(f"Initialized configuration file at {output}")
    else:
        click.echo(f"Configuration file {output} already exists.")

    click.echo(f"Initialized Dream-RSI environment in .dream_rsi (database: {db_path})")


@main.command()
@click.option("--task", "-t", required=True, help="Instruction describing the coding task.")
@click.option("--workspace", "-w", default=".", help="Root path of target project repository.")
@click.option("--agent", "-a", default="mock", help="Coding agent type: mock, gemini, claude, codex, or command.")
@click.option("--evaluator", "-e", default="pytest", help="Evaluator type: pytest, command, composite.")
@click.option("--max-rounds", default=10, type=int, help="Maximum decision rounds.")
@click.option("--max-calls", default=40, type=int, help="Maximum coding agent invocations.")
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def run(task: str, workspace: str, agent: str, evaluator: str, max_rounds: int, max_calls: int, config: str):
    """Run online exploration on a task, building a discovery tree and evaluating attempts."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    art_path = cfg.get("storage", {}).get("artifacts", ".dream_rsi/artifacts")

    store = SQLiteStore(db_path)
    art_mgr = ArtifactManager(art_path)
    ws_mgr = WorkspaceManager(workspace, storage_dir=cfg.get("storage", {}).get("workspaces", ".dream_rsi/workspaces"))

    # Instantiate evaluator
    if evaluator == "command":
        eval_engine = CommandEvaluator(cfg.get("evaluator", {}))
    elif evaluator == "composite":
        eval_engine = CompositeEvaluator(cfg.get("evaluator", {}))
    else:
        eval_engine = PytestEvaluator(cfg.get("evaluator", {}))

    # Instantiate agent
    if agent == "mock":
        agent_adapter = MockAgent()
    else:
        cmd = cfg.get("agent", {}).get("command", agent)
        agent_adapter = GenericCLIAgent({"command": cmd})

    # Load active policy
    active_pol_info = store.get_active_policy()
    if active_pol_info:
        pol_cfg = PolicyConfig.from_dict(active_pol_info["config"])
        policy = AdaptivePolicy(active_pol_info["policy_id"], pol_cfg)
        click.echo(f"Using active policy: {active_pol_info['policy_id']} ({active_pol_info['name']})")
    else:
        policy = AdaptivePolicy("default_policy", PolicyConfig())
        click.echo("Using default AdaptivePolicy")

    task_obj = Task(task_id=f"task_{abs(hash(task)) % 100000}", instruction=task, workspace_root=workspace)
    budget_obj = Budget(max_rounds=max_rounds, max_agent_calls=max_calls)

    click.echo(f"\n--- Starting Online Exploration: '{task}' ---")
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

    click.echo("\n=== Exploration Summary ===")
    click.echo(f"Tree ID:            {tree.tree_id}")
    click.echo(f"Total Nodes:        {stats['total_nodes']}")
    click.echo(f"Max Tree Depth:     {stats['max_depth']}")
    click.echo(f"Best Score:         {stats['max_score']:.4f}")
    if best_node:
        click.echo(f"Best Node ID:       {best_node.node_id} (Strategy: {best_node.strategy_id})")
        click.echo(f"Tests Passed:       {best_node.tests_passed} / {best_node.tests_passed + best_node.tests_failed}")
    click.echo(f"Total Cost:         ${stats['total_cost_usd']:.4f}")
    click.echo(f"Total Duration:     {stats['total_duration_seconds']:.2f}s")


@main.command()
@click.option("--candidates", default=16, type=int, help="Number of candidate policies to generate and simulate.")
@click.option("--margin", default=0.001, type=float, help="Minimum improvement margin required to adopt a candidate.")
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def dream(candidates: int, margin: float, config: str):
    """Run offline Dreaming: simulate candidate policies on historical discovery trees and select the best."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    store = SQLiteStore(db_path)

    trees = store.get_all_trees()
    if not trees:
        click.echo("No discovery trees found in database. Run online exploration first to generate histories!")
        sys.exit(0)

    active_pol_info = store.get_active_policy()
    if active_pol_info:
        pol_cfg = PolicyConfig.from_dict(active_pol_info["config"])
        curr_policy = AdaptivePolicy(active_pol_info["policy_id"], pol_cfg)
    else:
        curr_policy = AdaptivePolicy("baseline_policy", PolicyConfig())

    click.echo(f"Starting Offline Dreaming across {len(trees)} recorded discovery tree(s)...")
    click.echo(f"Baseline Policy: {curr_policy.policy_id} (Evaluating {candidates} candidates)")

    engine = DreamingEngine(store=store)
    engine.selector.min_improvement_margin = margin
    report = engine.run_dreaming(current_policy=curr_policy, trees=trees, num_candidates=candidates)

    click.echo("\n=== Dreaming Results ===")
    click.echo(f"Candidates Evaluated:    {report.candidates_evaluated}")
    click.echo(f"Baseline Replay Score V: {report.initial_replay_score:.4f}")
    click.echo(f"Selected Replay Score V: {report.selected_replay_score:.4f}")
    click.echo(f"Improved & Adopted:      {report.improved}")
    click.echo(f"Selected Policy ID:      {report.selected_policy_id}")

    win = report.winning_summary
    click.echo(f"Expected Parallelism:    {win.mean_parallelism:.2f}")
    click.echo(f"Expected Mean Attempts:  {win.mean_attempts:.1f}")
    click.echo(f"Expected Mean Score:     {win.mean_best_score:.4f}")


@main.command()
@click.option("--tree-id", default=None, help="Discovery tree ID to visualize (defaults to latest).")
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def tree(tree_id: Optional[str], config: str):
    """Inspect and visualize the hierarchy of a discovery tree."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    store = SQLiteStore(db_path)

    trees = store.get_all_trees()
    if not trees:
        click.echo("No trees found.")
        return

    target_tree = None
    if tree_id:
        target_tree = store.load_tree(tree_id)
    else:
        target_tree = trees[-1]

    if not target_tree:
        click.echo(f"Tree '{tree_id}' not found.")
        return

    click.echo(f"Discovery Tree: {target_tree.tree_id} (Task: {target_tree.task_id})")
    click.echo("=" * 60)

    def print_node(node_id: str, indent: int = 0):
        node = target_tree.get_node(node_id)
        if not node:
            return
        prefix = "  " * indent + ("└── " if indent > 0 else "")
        status_sym = "✓" if node.score >= 0.9 else "•"
        click.echo(
            f"{prefix}[{node.node_id}] {status_sym} Score: {node.score:.2f} | "
            f"Strategy: {node.strategy_id} | Changed: {len(node.changed_files)} files"
        )
        for child_id in node.children_ids:
            print_node(child_id, indent + 1)

    if target_tree.root_id:
        print_node(target_tree.root_id)


@main.command()
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def status(config: str):
    """Show current Dream-RSI status, database summary, and active policy."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    store = SQLiteStore(db_path)

    trees = store.get_all_trees()
    active_pol = store.get_active_policy()

    click.echo("=== Dream-RSI System Status ===")
    click.echo(f"Database:        {db_path}")
    click.echo(f"Total Trees:     {len(trees)}")
    total_nodes = sum(len(t.nodes) for t in trees)
    click.echo(f"Total Nodes:     {total_nodes}")

    if active_pol:
        click.echo(f"Active Policy:   {active_pol['policy_id']} ({active_pol['name']})")
        click.echo(f"Policy Type:     {active_pol['policy_type']}")
        click.echo(f"Policy Config:   {json.dumps(active_pol['config'], indent=2)}")
    else:
        click.echo("Active Policy:   None (run 'dream-rsi init' to initialize)")


@main.command()
@click.option("--tree-id", default=None, help="Tree ID to find best attempt from (defaults to latest).")
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def best(tree_id: Optional[str], config: str):
    """Display the highest-scoring attempt and details."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    store = SQLiteStore(db_path)

    trees = store.get_all_trees()
    if not trees:
        click.echo("No trees found.")
        return

    target_tree = store.load_tree(tree_id) if tree_id else trees[-1]
    if not target_tree:
        click.echo("Tree not found.")
        return

    best_node = target_tree.get_best_node()
    if not best_node:
        click.echo("No nodes in tree.")
        return

    click.echo(f"=== Best Attempt in Tree '{target_tree.tree_id}' ===")
    click.echo(f"Node ID:       {best_node.node_id}")
    click.echo(f"Score:         {best_node.score:.4f}")
    click.echo(f"Strategy:      {best_node.strategy_id}")
    click.echo(f"Tests Passed:  {best_node.tests_passed} (Failed: {best_node.tests_failed})")
    click.echo(f"Changed Files: {', '.join(best_node.changed_files) or 'None'}")
    click.echo(f"Artifacts:     {json.dumps(best_node.artifacts)}")
    if best_node.summary:
        click.echo(f"Summary:       {best_node.summary}")


@main.command()
@click.option("--task", "-t", default="Run baseline comparison", help="Task description.")
@click.option("--workspace", "-w", default=".", help="Workspace path.")
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def benchmark(task: str, workspace: str, config: str):
    """Run benchmark comparing Single-Agent, Best-of-N, Fixed, Adaptive UCB, and Evolved Dream-RSI."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    art_path = cfg.get("storage", {}).get("artifacts", ".dream_rsi/artifacts")

    store = SQLiteStore(db_path)
    art_mgr = ArtifactManager(art_path)
    ws_mgr = WorkspaceManager(workspace)
    evaluator = PytestEvaluator(cfg.get("evaluator", {}))
    agent = MockAgent()

    runner = BenchmarkRunner(
        workspace_manager=ws_mgr,
        evaluator=evaluator,
        agent=agent,
        store=store,
        artifact_mgr=art_mgr,
    )

    task_obj = Task(task_id="bench_task", instruction=task, workspace_root=workspace)
    click.echo("Running benchmark across exploration baselines...")
    metrics = runner.run_all_baselines(task_obj)

    click.echo("\n=== Exploration Benchmark Results ===")
    header = f"{'Baseline':<32} | {'Score':<6} | {'Solved':<6} | {'Calls':<6} | {'Duration':<8}"
    click.echo(header)
    click.echo("-" * len(header))
    for m in metrics:
        click.echo(
            f"{m.baseline_name:<32} | {m.best_score:<6.2f} | {str(m.solved):<6} | "
            f"{m.agent_calls:<6} | {m.duration_seconds:<7.2f}s"
        )


@main.command()
@click.option("--output", "-o", default="dream_rsi_export.json", help="Path to write exported JSON.")
@click.option("--config", "-c", default="config.yaml", help="Configuration file path.")
def export(output: str, config: str):
    """Export discovery trees and run statistics to JSON."""
    cfg = load_config(config)
    db_path = cfg.get("storage", {}).get("database", ".dream_rsi/dream_rsi.sqlite")
    store = SQLiteStore(db_path)

    trees = store.get_all_trees()
    export_data = {
        "total_trees": len(trees),
        "trees": [t.to_dict() for t in trees],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    click.echo(f"Exported {len(trees)} discovery tree(s) to {output}")


if __name__ == "__main__":
    main()
