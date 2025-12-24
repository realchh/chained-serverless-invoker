from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple

from .config import WorkflowConfig, WorkflowEdge
from .dag.graph import EdgeDef, WorkflowDag
from .stats import StatSummary


def build_dag(config: WorkflowConfig) -> WorkflowDag:
    edges = [EdgeDef(from_fn=e.from_fn, to_fn=e.to_fn, edge_id=e.edge_id) for e in config.edges]
    return WorkflowDag(edges)


def edge_weights_from_stats(edge_stats: Mapping[str, StatSummary]) -> dict[str, float]:
    return {key: summary.p50 for key, summary in edge_stats.items()}


def critical_path_from_stats(
    dag: WorkflowDag,
    *,
    edge_stats: Mapping[str, StatSummary | float] | None = None,
    node_stats: Mapping[str, StatSummary | float] | None = None,
    source_nodes: Iterable[str] | None = None,
    sink_nodes: Iterable[str] | None = None,
) -> Tuple[float, List[str]]:
    edge_weights = {k: (v.p50 if hasattr(v, "p50") else float(v)) for k, v in (edge_stats or {}).items()}
    node_weights = {k: (v.p50 if hasattr(v, "p50") else float(v)) for k, v in (node_stats or {}).items()}

    return dag.critical_path(
        edge_weights=edge_weights,
        node_weights=node_weights,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
    )


def edge_key(edge: WorkflowEdge) -> str:
    return edge.edge_id or f"{edge.from_fn}->{edge.to_fn}"
