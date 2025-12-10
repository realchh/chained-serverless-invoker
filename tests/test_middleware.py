import pytest

from middleware.serverless_tuner_middleware.critical_path import build_dag
from middleware.serverless_tuner_middleware.config import WorkflowConfig, WorkflowEdge
from middleware.serverless_tuner_middleware.rewrite import rewrite_config_for_critical_path
from middleware.serverless_tuner_middleware.stats import StatSummary


def _stat(p50: float) -> StatSummary:
    return StatSummary(count=1, p50=p50, p90=p50, mean=p50)


def test_critical_path_shifts_when_weights_change():
    # entry -> A -> C and entry -> B -> C
    edges = [
        WorkflowEdge(from_fn="entry", to_fn="A", strategy="dynamic", edge_id="entry-A"),
        WorkflowEdge(from_fn="entry", to_fn="B", strategy="dynamic", edge_id="entry-B"),
        WorkflowEdge(from_fn="A", to_fn="C", strategy="dynamic", edge_id="A-C"),
        WorkflowEdge(from_fn="B", to_fn="C", strategy="dynamic", edge_id="B-C"),
    ]
    dag = build_dag(WorkflowConfig(workflow_id="wf", edges=edges))

    # Path entry->A->C is heavier initially (critical).
    weights = {
        "entry-A": 10.0,
        "A-C": 20.0,
        "entry-B": 5.0,
        "B-C": 5.0,
    }
    total, path = dag.critical_path(edge_weights=weights, node_weights=None)
    assert path == ["entry", "A", "C"]
    assert total == pytest.approx(30.0)

    # Make B->C heavier so entry->B->C becomes critical.
    weights["B-C"] = 40.0
    total, path = dag.critical_path(edge_weights=weights, node_weights=None)
    assert path == ["entry", "B", "C"]
    assert total == pytest.approx(45.0)


def test_rewrite_prefers_fast_mechanism_on_critical_path():
    cfg = WorkflowConfig(
        workflow_id="wf",
        edges=[
            WorkflowEdge(
                from_fn="entry",
                to_fn="A",
                strategy="dynamic",
                endpoint="https://a",
                topic="projects/p/topics/a",
                edge_id="entry-A",
            ),
            WorkflowEdge(
                from_fn="entry",
                to_fn="B",
                strategy="dynamic",
                endpoint="https://b",
                topic="projects/p/topics/b",
                edge_id="entry-B",
            ),
            WorkflowEdge(
                from_fn="A",
                to_fn="C",
                strategy="dynamic",
                endpoint="https://c",
                topic="projects/p/topics/c",
                edge_id="A-C",
            ),
            WorkflowEdge(
                from_fn="B",
                to_fn="C",
                strategy="dynamic",
                endpoint="https://c2",
                topic="projects/p/topics/c2",
                edge_id="B-C",
            ),
        ],
    )

    edge_stats = {
        ("entry-A", "http"): _stat(100.0),
        ("entry-A", "pubsub"): _stat(150.0),
        ("A-C", "http"): _stat(100.0),
        ("A-C", "pubsub"): _stat(160.0),
        ("entry-B", "http"): _stat(10.0),
        ("entry-B", "pubsub"): _stat(20.0),
        ("B-C", "http"): _stat(10.0),
        ("B-C", "pubsub"): _stat(20.0),
    }

    rewritten = rewrite_config_for_critical_path(cfg, edge_stats=edge_stats, node_stats=None)

    # entry->A->C is critical; it should flip to fastest (http).
    edge_map = {(e.from_fn, e.to_fn): e for e in rewritten.edges}
    assert edge_map[("entry", "A")].strategy == "http"
    assert edge_map[("A", "C")].strategy == "http"

    # Non-critical edges default to HTTP after optimization.
    assert edge_map[("entry", "B")].strategy == "http"
    assert edge_map[("B", "C")].strategy == "http"
