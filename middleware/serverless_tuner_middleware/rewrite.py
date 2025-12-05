from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Tuple

from .config import WorkflowConfig, WorkflowEdge, dump_config, load_config
from .critical_path import build_dag, critical_path_from_stats, edge_key
from .stats import StatSummary


def _choose_fastest_mechanism(edge_key_str: str, edge_stats: Mapping[Tuple[str, str], StatSummary]) -> str | None:
    http = edge_stats.get((edge_key_str, "http"))
    pubsub = edge_stats.get((edge_key_str, "pubsub"))

    if http and pubsub:
        return "http" if http.p50 <= pubsub.p50 else "pubsub"
    if http:
        return "http"
    if pubsub:
        return "pubsub"
    return None


def _weight_for_edge(edge: WorkflowEdge, stats: Mapping[Tuple[str, str], StatSummary]) -> float:
    key = edge_key(edge)
    http = stats.get((key, "http"))
    pubsub = stats.get((key, "pubsub"))

    strat = edge.strategy.lower()
    if strat == "http" and http:
        return http.p50
    if strat == "pubsub" and pubsub:
        return pubsub.p50
    if strat == "dynamic":
        candidates = [s.p50 for s in (http, pubsub) if s]
        if candidates:
            return min(candidates)
    return 0.0


def _node_weights(node_stats: Mapping[str, StatSummary]) -> Dict[str, float]:
    return {fn: summary.p50 for fn, summary in node_stats.items()}


def rewrite_config_for_critical_path(
    config: WorkflowConfig,
    *,
    edge_stats: Mapping[Tuple[str, str], StatSummary],
    node_stats: Mapping[str, StatSummary] | None = None,
) -> WorkflowConfig:
    dag = build_dag(config)
    edge_weights = {edge_key(e): _weight_for_edge(e, edge_stats) for e in config.edges}
    node_weights = _node_weights(node_stats or {})

    _, path_nodes = critical_path_from_stats(dag, edge_stats=edge_weights, node_stats=node_weights)

    path_edges = {(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)}

    new_edges: list[WorkflowEdge] = []
    for e in config.edges:
        k = edge_key(e)
        on_path = (e.from_fn, e.to_fn) in path_edges

        if on_path:
            mechanism = _choose_fastest_mechanism(k, edge_stats)
            if mechanism:
                new_edges.append(replace(e, strategy=mechanism))
                continue
        else:
            if e.topic:
                new_edges.append(replace(e, strategy="pubsub"))
                continue

        new_edges.append(e)

    return WorkflowConfig(workflow_id=config.workflow_id, edges=new_edges)


__all__ = [
    "rewrite_config_for_critical_path",
    "load_config",
    "dump_config",
]
