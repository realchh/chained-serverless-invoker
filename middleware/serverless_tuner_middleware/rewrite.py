from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Tuple

from . import constants
from .config import WorkflowConfig, WorkflowEdge, dump_config, load_config
from .critical_path import build_dag, critical_path_from_stats, edge_key
from .stats import StatSummary


def _mechanism_cost(edge_key_str: str, mechanism: str, stats: Mapping[Tuple[str, str], StatSummary]) -> float | None:
    summary = stats.get((edge_key_str, mechanism))
    return summary.p50 if summary else None


def _edge_cost_for_strategy(
    edge: WorkflowEdge, strategy: str, stats: Mapping[Tuple[str, str], StatSummary]
) -> float | None:
    return _mechanism_cost(edge_key(edge), strategy.lower(), stats)


def _weight_for_edge(edge: WorkflowEdge, stats: Mapping[Tuple[str, str], StatSummary]) -> float:
    key = edge_key(edge)
    http = _mechanism_cost(key, "http", stats)
    pubsub = _mechanism_cost(key, "pubsub", stats)

    strat = edge.strategy.lower()
    # Treat dynamic as http by default (cost-efficient baseline).
    if strat == "pubsub" and pubsub is not None:
        return pubsub
    if http is not None:
        return http
    if pubsub is not None:
        return pubsub
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
    # Start with dynamic treated as HTTP (cost baseline). Flips may change the critical path.
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
            # If dynamic or missing, fall back to the best known cost for this edge.
            if current_cost is None:
                current_cost = _weight_for_edge(edge, edge_stats)

            # Identify the fastest available mechanism for this edge.
            http_cost = _mechanism_cost(edge_key(edge), "http", edge_stats)
            pubsub_cost = _mechanism_cost(edge_key(edge), "pubsub", edge_stats)
            pairs: list[tuple[float, str]] = []
            if http_cost is not None:
                pairs.append((http_cost, "http"))
            if pubsub_cost is not None:
                pairs.append((pubsub_cost, "pubsub"))

            if pairs:
                best_cost, fastest_mech = min(pairs, key=lambda x: x[0])
                if fastest_mech != current_mech:
                    gain = (current_cost if current_cost is not None else best_cost) - best_cost
                    # If current is dynamic, allow a zero-gain flip to make it explicit.
                    if current_mech == "dynamic" and gain <= 0:
                        gain = constants.GAIN_THRESHOLD_MS + 0.001
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        best_mech = fastest_mech

        # Require a tiny positive gain to avoid churn on noise.
        if best_idx is None or best_mech is None or best_gain <= constants.GAIN_THRESHOLD_MS:
            break

        edges[best_idx] = replace(edges[best_idx], strategy=best_mech)

    # After optimization, normalize any remaining dynamic to HTTP (baseline).
    updated_edges = [replace(e, strategy="http") if e.strategy.lower() == "dynamic" else e for e in edges]

    return WorkflowConfig(workflow_id=config.workflow_id, edges=updated_edges)


__all__ = [
    "rewrite_config_for_critical_path",
    "load_config",
    "dump_config",
]
