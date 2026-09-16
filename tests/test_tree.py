"""Unit tests for DiscoveryTree operations and diversity analytics."""

import pytest
from dream_rsi.core.models import TreeNode, NodeStatus
from dream_rsi.core.tree import DiscoveryTree


def create_sample_node(node_id: str, parent_id=None, strategy="default", score=0.5, files=None):
    return TreeNode(
        node_id=node_id,
        parent_id=parent_id,
        task_id="task_0",
        strategy_id=strategy,
        instruction=f"Instruction for {node_id}",
        workspace_snapshot=f"snap_{node_id}",
        changed_files=files or ["file.py"],
        score=score,
        status=NodeStatus.COMPLETED,
    )


def test_tree_construction_and_hierarchy():
    tree = DiscoveryTree(tree_id="tree_1", task_id="task_0")
    root = create_sample_node("n_root", parent_id=None, score=0.4)
    tree.add_root(root)

    child1 = create_sample_node("n_c1", parent_id="n_root", score=0.6)
    child2 = create_sample_node("n_c2", parent_id="n_root", score=0.8)
    tree.add_child("n_root", child1)
    tree.add_child("n_root", child2)

    assert tree.root_id == "n_root"
    assert len(tree.nodes) == 3
    assert child1.depth == 1
    assert child2.depth == 1

    eligible = tree.get_eligible_nodes()
    eligible_ids = {n.node_id for n in eligible}
    assert eligible_ids == {"n_root", "n_c1", "n_c2"}

    best = tree.get_best_node()
    assert best is not None
    assert best.node_id == "n_c2"
    assert best.score == 0.8

    path = tree.get_branch_path("n_c1")
    assert [n.node_id for n in path] == ["n_root", "n_c1"]


def test_tree_diversity_metrics():
    tree = DiscoveryTree(tree_id="tree_div", task_id="task_0")
    n1 = create_sample_node("n1", files=["a.py", "b.py"], strategy="strat_a")
    n2 = create_sample_node("n2", files=["b.py", "c.py"], strategy="strat_a")
    n3 = create_sample_node("n3", files=["x.py", "y.py"], strategy="strat_b")

    # File overlap between n1 and n2: 'b.py' common out of {'a.py', 'b.py', 'c.py'} -> 1/3 ~ 0.333
    overlap_1_2 = tree.compute_file_overlap(n1, n2)
    assert pytest.approx(overlap_1_2, 0.01) == 0.333

    # Overlap between n1 and n3: 0.0
    overlap_1_3 = tree.compute_file_overlap(n1, n3)
    assert overlap_1_3 == 0.0

    penalty = tree.calculate_diversity_penalty(n2, [n1])
    assert penalty > 0.0


def test_tree_serialization():
    tree = DiscoveryTree(tree_id="tree_ser", task_id="task_0")
    root = create_sample_node("root", score=0.2)
    tree.add_root(root)
    child = create_sample_node("child", parent_id="root", score=0.9)
    tree.add_child("root", child)

    d = tree.to_dict()
    restored = DiscoveryTree.from_dict(d)
    assert restored.tree_id == "tree_ser"
    assert restored.root_id == "root"
    assert len(restored.nodes) == 2
    assert restored.nodes["child"].depth == 1
