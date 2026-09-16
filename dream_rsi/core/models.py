"""Core data models for Dream-RSI."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class NodeStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Task:
    """A coding task to be solved by the discovery system."""
    task_id: str
    instruction: str
    workspace_root: str
    evaluator_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        return cls(**data)


@dataclass
class EvaluationResult:
    """Standardized output from an evaluator."""
    success: bool
    score: float  # Normalized 0.0 to 1.0 (or higher if unbounded utility)
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    total_tests: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    stdout_artifact: Optional[str] = None
    stderr_artifact: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationResult:
        return cls(**data)


@dataclass
class AgentResult:
    """Output from an AgentAdapter execution."""
    success: bool
    text_summary: str = ""
    changed_files: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    raw_output_path: Optional[str] = None
    diff_artifact: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentResult:
        return cls(**data)


@dataclass
class TreeNode:
    """A single node in the discovery tree representing an exploration attempt."""
    node_id: str
    parent_id: Optional[str]
    task_id: str
    strategy_id: str
    instruction: str
    workspace_snapshot: str  # Path or Git commit / worktree identifier
    changed_files: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    score: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    duration_seconds: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    agent_id: str = "default"
    timestamp: float = field(default_factory=time.time)
    status: NodeStatus = NodeStatus.PENDING
    error_info: Optional[str] = None
    summary: str = ""
    depth: int = 0
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeNode:
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = NodeStatus(data["status"])
        return cls(**data)


@dataclass
class Budget:
    """Resource constraints for online exploration and offline dreaming."""
    max_rounds: int = 10
    max_agent_calls: int = 40
    max_parallel_workers: int = 4
    max_duration_seconds: float = 3600.0
    max_cost_usd: float = 50.0
    max_replay_simulations: int = 100
    stop_score_threshold: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Budget:
        return cls(**data)


@dataclass
class ReplayResult:
    """Outcome of simulating a policy on a recorded discovery tree."""
    policy_id: str
    revealed_node_ids: List[str]
    best_node_id: Optional[str]
    best_score: float
    total_attempts: int
    decision_rounds: int
    parallelism: float
    replay_score: float  # Objective V
    stopped_by_policy: bool
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplayResult:
        return cls(**data)
