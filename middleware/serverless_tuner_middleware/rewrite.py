from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Tuple

from .config import WorkflowConfig, WorkflowEdge, dump_config, load_config
from . import constants
from .critical_path import build_dag, critical_path_from_stats, edge_key
from .stats import StatSummary


def _mechanism_cost(edge_key_str: str, mechanism: str, stats: Mapping[Tuple[str, str], StatSummary]) -> float | None:
    summary = stats.get((edge_key_str, mechanism))
    return summary.p50 if summary else None


def _edge_cost_for_strategy(edge: WorkflowEdge, strategy: str, stats: Mapping[Tuple[str, str], StatSummary]) -> float | None:
    return _mechanism_cost(edge_key(edge), strategy.lower(), stats)


def _weight_for_edge(edge: WorkflowEdge, stats: Mapping[Tuple[str, str], StatSummary]) -> float:
    key = edge_key(edge)
    http = _mechanism_cost(key, "http", stats)
    pubsub = _mechanism_cost(key, "pubsub", stats)

    strat = edge.strategy.lower()
    if strat == "http" and http is not None:
        return http
    if strat == "pubsub" and pubsub is not None:
        return pubsub
    if strat == "dynamic":
        candidates = [c for c in (http, pubsub) if c is not None]
        if candidates:
            return min(candidates)
    return 0.0  # missing stats → treat as zero so we don't block path computation


def _node_weights(node_stats: Mapping[str, StatSummary]) -> Dict[str, float]:
    return {fn: summary.p50 for fn, summary in node_stats.items()}


def rewrite_config_for_critical_path(
    config: WorkflowConfig,
    *,
    edge_stats: Mapping[Tuple[str, str], StatSummary],
    node_stats: Mapping[str, StatSummary] | None = None,
) -> WorkflowConfig:
    # TODO: incorporate cost-aware scores (latency + \$) once pricing signals are available.
    dag = build_dag(config)
    node_weights = _node_weights(node_stats or {})
    edges = list(config.edges)

    # Greedy: flip one edge on the current critical path at a time if it improves p50 latency.
    max_iters = len(edges) or 1
    for _ in range(max_iters):
        edge_weights = {edge_key(e): _weight_for_edge(e, edge_stats) for e in edges}
        _, path_nodes = critical_path_from_stats(dag, edge_stats=edge_weights, node_stats=node_weights)
        if len(path_nodes) < 2:
            break

        path_pairs = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]

        best_gain = 0.0
        best_idx: int | None = None
        best_mech: str | None = None

        for frm, to in path_pairs:
            try:
                idx = next(i for i, e in enumerate(edges) if e.from_fn == frm and e.to_fn == to)
            except StopIteration:
                continue

            edge = edges[idx]
            current_mech = edge.strategy.lower()
            current_cost = _edge_cost_for_strategy(edge, current_mech, edge_stats)
            if current_cost is None:
                current_cost = 0.0

            for alt in ("http", "pubsub"):
                if alt == current_mech:
                    continue
                alt_cost = _edge_cost_for_strategy(edge, alt, edge_stats)
                if alt_cost is None:
                    continue

                gain = current_cost - alt_cost
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_mech = alt

        # Require a tiny positive gain to avoid churn on noise.
        if best_idx is None or best_mech is None or best_gain <= constants.GAIN_THRESHOLD_MS:
            break

        edges[best_idx] = replace(edges[best_idx], strategy=best_mech)

    return WorkflowConfig(workflow_id=config.workflow_id, edges=edges)


__all__ = [
    "rewrite_config_for_critical_path",
    "load_config",
    "dump_config",
]
