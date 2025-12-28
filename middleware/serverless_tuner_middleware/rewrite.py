from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Tuple

from . import constants
from .config import WorkflowConfig, WorkflowEdge, dump_config, load_config
from .critical_path import build_dag, edge_key
from .dag.graph import WorkflowDag
from .model import REGRESSION_MODEL
from .stats import EdgeSample, NodeSample, StatSummary


def _mechanism_cost(edge_key_str: str, mechanism: str, stats: Mapping[Tuple[str, str], StatSummary]) -> float | None:
    summary = stats.get((edge_key_str, mechanism))
    return summary.p50 if summary else None


def _edge_cost_for_strategy(
    edge: WorkflowEdge, strategy: str, stats: Mapping[Tuple[str, str], StatSummary]
) -> float | None:
    return _mechanism_cost(edge_key(edge), strategy.lower(), stats)


def _mechanism_cost_or_model(
    edge: WorkflowEdge,
    mechanism: str,
    stats: Mapping[Tuple[str, str], StatSummary],
    *,
    region_pair: str | None,
    payload_size_bytes: int | None,
    rate_rps: float | None,
    quantile: float | str = "p50",
) -> float | None:
    cost = _mechanism_cost(edge_key(edge), mechanism, stats)
    if cost is not None:
        return cost
    if region_pair and payload_size_bytes is not None and rate_rps is not None:
        return _model_predict(mechanism, region_pair, payload_size_bytes, rate_rps, quantile=quantile)
    return None


def _model_predict(
    mechanism: str, region_pair: str, payload_size_bytes: int, rate_rps: float, quantile: float | str = "p50"
) -> float | None:
    quant_str = quantile if isinstance(quantile, str) else f"p{int(quantile)}"
    return REGRESSION_MODEL.predict(
        mechanism=mechanism,
        region_pair=region_pair,
        quantile=quant_str,
        payload_bytes=payload_size_bytes,
        rate_rps=rate_rps,
    )


def _weight_for_edge(
    edge: WorkflowEdge,
    stats: Mapping[Tuple[str, str], StatSummary],
    *,
    region_pair: str | None = None,
    payload_size_bytes: int | None = None,
    rate_rps: float | None = None,
    strategy: str | None = None,
    quantile: float | str = "p50",
) -> float:
    """Return a transport weight for the given strategy (obs first, otherwise model)."""

    key = edge_key(edge)
    mech = (strategy or edge.strategy).lower()
    if mech == "dynamic":
        mech = "http"

    obs_cost = _mechanism_cost(key, mech, stats)
    if obs_cost is not None:
        return obs_cost

    # Try to infer region pair from the target endpoint when not provided.
    if region_pair is None:
        region_pair = _infer_region_pair(edge)

    if region_pair and payload_size_bytes is not None and rate_rps is not None:
        model_cost = _model_predict(mech, region_pair, payload_size_bytes, rate_rps, quantile=quantile)
        if model_cost is not None:
            return model_cost

    # Missing stats/model for the chosen mech → treat as zero so path computation still works.
    return 0.0


def rewrite_config_for_critical_path(
    config: WorkflowConfig,
    *,
    edge_stats: Mapping[Tuple[str, str], StatSummary],
    node_stats: Mapping[str, StatSummary] | None = None,
    edge_samples: Iterable[EdgeSample] | None = None,
    node_samples: Iterable[NodeSample] | None = None,
    edge_context: Mapping[str, Tuple[str | None, int | None, float | None]] | None = None,
    verbose: bool = False,
    percentile: float = 50.0,
) -> WorkflowConfig:
    # TODO: incorporate cost-aware scores (latency + \$) once pricing signals are available.
    dag = build_dag(config)
    edges = list(config.edges)

    # Sync bottleneck preference: identify sync edges that frequently win per-run bottlenecks.
    sync_bottleneck_info: Dict[str, Tuple[float, int]] = {}
    flagged_sync_edges: set[str] = set()
    if edge_samples is not None:
        sync_bottleneck_info, flagged_sync_edges = _sync_bottleneck_edges(
            edge_samples,
            run_share_threshold=constants.SYNC_BOTTLENECK_RUN_SHARE_THRESHOLD,
        )

    edge_stats_pct, node_stats_pct = _percentile_stats(edge_samples, node_samples, edge_stats, node_stats, percentile)

    log = print if verbose else (lambda *args, **kwargs: None)
    log("Edges (strategy):")
    for e in edges:
        log(f"  {edge_key(e)} -> {e.strategy}")

    # Greedy: flip one edge on the current critical path at a time if it improves p50 latency.
    # Start with dynamic treated as HTTP (cost baseline). Flips may change the critical path.
    max_iters = min(constants.MAX_EDGE_FLIPS_PER_RUN, len(edges) or 1)
    for iter_idx in range(max_iters):
        edge_weights = {}
        for e in edges:
            ctx = edge_context.get(edge_key(e)) if edge_context else None
            region = ctx[0] if ctx else None
            size_bytes = ctx[1] if ctx else None
            rate_rps = ctx[2] if ctx else None
            edge_weights[edge_key(e)] = _weight_for_edge(
                e,
                edge_stats_pct,
                region_pair=region,
                payload_size_bytes=size_bytes,
                rate_rps=rate_rps,
                strategy=e.strategy,
                quantile=percentile,
            )

        node_weights = {fn: stat.p50 for fn, stat in (node_stats or {}).items()}
        path_nodes, candidates = _select_sync_path(
            dag,
            edges,
            edge_weights,
            node_weights,
            sync_bottleneck_info,
            flagged_sync_edges,
        )
        if len(path_nodes) < 2:
            break

        # Verbose: show candidate paths (sync-run-share primary, end-to-end cost tiebreak)
        log(f"\n[iter {iter_idx+1}] critical-path candidates (sync bottleneck primary):")
        for nodes, share, cost in candidates:
            mark = "*" if nodes == path_nodes else " "
            share_pct = share * 100
            log(f"{mark} sync_run_share={share_pct:.1f}% cost={cost:.3f} nodes={nodes}")

        path_pairs = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
        current_path_cost = sum(
            edge_weights.get(
                edge_key(next(e for e in edges if e.from_fn == frm and e.to_fn == to)),
                0.0,
            )
            for frm, to in path_pairs
        )

        best_gain = 0.0
        best_idx: int | None = None
        best_mech: str | None = None
        best_detail: Dict[str, float | str | None] = {}

        for frm, to in path_pairs:
            try:
                idx = next(i for i, e in enumerate(edges) if e.from_fn == frm and e.to_fn == to)
            except StopIteration:
                continue

            edge = edges[idx]
            current_mech = edge.strategy.lower()
            ctx = edge_context.get(edge_key(edge)) if edge_context else None
            region = ctx[0] if ctx else None
            size_bytes = ctx[1] if ctx else None
            rate_rps = ctx[2] if ctx else None

            current_cost = _mechanism_cost_or_model(
                edge,
                current_mech,
                edge_stats_pct,
                region_pair=region,
                payload_size_bytes=size_bytes,
                rate_rps=rate_rps,
                quantile=percentile,
            )
            if current_cost is None:
                current_cost = _weight_for_edge(
                    edge,
                    edge_stats_pct,
                    region_pair=region,
                    payload_size_bytes=size_bytes,
                    rate_rps=rate_rps,
                    quantile=percentile,
                )

            # Identify the fastest available mechanism for this edge.
            http_cost = _mechanism_cost_or_model(
                edge,
                "http",
                edge_stats_pct,
                region_pair=region,
                payload_size_bytes=size_bytes,
                rate_rps=rate_rps,
                quantile=percentile,
            )
            pubsub_cost = _mechanism_cost_or_model(
                edge,
                "pubsub",
                edge_stats_pct,
                region_pair=region,
                payload_size_bytes=size_bytes,
                rate_rps=rate_rps,
                quantile=percentile,
            )
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
                        best_detail = {
                            "edge_key": edge_key(edge),
                            "current": current_mech,
                            "http_obs": _mechanism_cost(edge_key(edge), "http", edge_stats_pct),
                            "pubsub_obs": _mechanism_cost(edge_key(edge), "pubsub", edge_stats_pct),
                            "http_pred": _model_predict(
                                "http",
                                region,
                                size_bytes,
                                rate_rps,
                                quantile=percentile,
                            )
                            if region and size_bytes is not None and rate_rps is not None
                            else None,
                            "pubsub_pred": _model_predict(
                                "pubsub",
                                region,
                                size_bytes,
                                rate_rps,
                                quantile=percentile,
                            )
                            if region and size_bytes is not None and rate_rps is not None
                            else None,
                            "gain_ms": gain,
                            "best_mech": fastest_mech,
                        }

        # Require a tiny positive gain to avoid churn on noise.
        if best_idx is None or best_mech is None or best_gain <= constants.GAIN_THRESHOLD_MS:
            log("No more edges to flip that clear the gain floor.")
            break

        if best_detail:
            log(
                f"Flipping {best_detail['edge_key']} to {best_detail['best_mech']} (current={best_detail['current']}) "
                f"gain≈{best_detail['gain_ms']:.3f} ms | "
                f"obs http={best_detail['http_obs']} pubsub={best_detail['pubsub_obs']} | "
                f"pred http={best_detail['http_pred']} pubsub={best_detail['pubsub_pred']}"
            )

        edges[best_idx] = replace(edges[best_idx], strategy=best_mech)

        # Recompute path after flip to see if it changed and report improvement.
        edge_weights_after = {}
        for e in edges:
            ctx = edge_context.get(edge_key(e)) if edge_context else None
            region = ctx[0] if ctx else None
            size_bytes = ctx[1] if ctx else None
            rate_rps = ctx[2] if ctx else None
            edge_weights_after[edge_key(e)] = _weight_for_edge(
                e,
                edge_stats_pct,
                region_pair=region,
                payload_size_bytes=size_bytes,
                rate_rps=rate_rps,
                strategy=e.strategy,
                quantile=percentile,
            )

        node_weights_after = node_weights
        new_path_nodes, new_candidates = _select_sync_path(
            dag,
            edges,
            edge_weights_after,
            node_weights_after,
            sync_bottleneck_info,
            flagged_sync_edges,
        )
        new_path_cost = (
            sum(
                edge_weights_after.get(
                    edge_key(next(e for e in edges if e.from_fn == frm and e.to_fn == to)),
                    0.0,
                )
                for frm, to in zip(new_path_nodes, new_path_nodes[1:], strict=False)
            )
            if len(new_path_nodes) >= 2
            else 0.0
        )

        if new_path_nodes == path_nodes:
            log(
                f"Critical path unchanged. Path cost improved from {current_path_cost:.3f} ms to {new_path_cost:.3f} ms"
            )
        else:
            log(
                (
                    f"Critical path changed from {path_nodes} (cost {current_path_cost:.3f} ms) "
                    f"to {new_path_nodes} (cost {new_path_cost:.3f} ms)"
                )
            )
            log("Next critical-path candidates after flip:")
            for nodes, share, cost in new_candidates:
                mark = "*" if nodes == new_path_nodes else " "
                log(f"{mark} sync_run_share={share*100:.1f}% cost={cost:.3f} nodes={nodes}")

        if iter_idx + 1 == max_iters:
            log("Reached maximum edge flips per run; stopping.")

    # After optimization, normalize any remaining dynamic to HTTP (baseline).
    updated_edges = [replace(e, strategy="http") if e.strategy.lower() == "dynamic" else e for e in edges]

    return WorkflowConfig(workflow_id=config.workflow_id, edges=updated_edges)


_REGION_RE = re.compile(r"\b([a-z]+-[a-z0-9]+[0-9])\.run\.app")


def _infer_region_pair(edge: WorkflowEdge) -> str | None:
    """
    Best-effort region-pair inference from the target endpoint hostname.
    If only one region is found, assume src=dst.
    """
    if not edge.endpoint:
        return None
    match = _REGION_RE.search(edge.endpoint)
    if not match:
        return None
    region = match.group(1)
    return f"{region}->{region}"


def _sync_bottleneck_edges(
    edge_samples: Iterable[EdgeSample],
    *,
    run_share_threshold: float,
) -> Tuple[Dict[str, Tuple[float, int]], set[str]]:
    """Compute sync-edge bottleneck run share and flag edges above thresholds."""
    sync_samples = [s for s in edge_samples if ":sync:" in s.edge_key]
    if not sync_samples:
        return {}, set()

    by_run: Dict[str, list[EdgeSample]] = {}
    for sample in sync_samples:
        by_run.setdefault(sample.run_id, []).append(sample)

    winners: Dict[str, int] = {}
    for _run_id, samples in by_run.items():
        if not samples:
            continue
        winner = max(samples, key=lambda s: s.transport_ms)
        winners[winner.edge_key] = winners.get(winner.edge_key, 0) + 1

    sample_counts: Dict[str, int] = {}
    for sample in sync_samples:
        sample_counts[sample.edge_key] = sample_counts.get(sample.edge_key, 0) + 1

    run_total = len(by_run)
    info: Dict[str, Tuple[float, int]] = {}
    flagged: set[str] = set()
    for ek, win_count in winners.items():
        share = win_count / run_total if run_total else 0.0
        cnt = sample_counts.get(ek, 0)
        info[ek] = (share, cnt)
        if share >= run_share_threshold:
            flagged.add(ek)

    return info, flagged


def _enumerate_paths(dag: WorkflowDag) -> list[list[str]]:
    """List all source→sink node paths in the DAG."""
    sources = dag.sources()
    sinks = set(dag.sinks())

    paths: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        if node in sinks or not dag.successors(node):
            paths.append(list(path))
            return
        for succ in dag.successors(node):
            dfs(succ, path + [succ])

    for src in sources:
        dfs(src, [src])

    return paths


def _select_sync_path(
    dag: WorkflowDag,
    edges: list[WorkflowEdge],
    edge_weights: Mapping[str, float],
    node_weights: Mapping[str, float],
    sync_bottleneck_info: Mapping[str, Tuple[float, int]],
    flagged_sync_edges: set[str],
) -> Tuple[list[str], list[Tuple[list[str], float, float]]]:
    """Pick path containing the highest run-share sync edge; break ties by end-to-end cost."""

    scored: list[Tuple[list[str], float, float, bool]] = []

    all_paths = _enumerate_paths(dag)
    if not all_paths:
        return [], []

    edge_by_pair: Dict[Tuple[str, str], WorkflowEdge] = {(e.from_fn, e.to_fn): e for e in edges}

    # Determine target sync edge(s): prefer flagged, else max run-share band if any, else none.
    target_edges: set[str] = set()
    if flagged_sync_edges:
        target_edges = set(flagged_sync_edges)
    elif sync_bottleneck_info:
        max_share = max(share for share, _ in sync_bottleneck_info.values())
        band = max_share * (1 - constants.SYNC_RUN_SHARE_TOLERANCE)
        target_edges = {ek for ek, (share, _) in sync_bottleneck_info.items() if share >= band}

    for nodes in all_paths:
        pairs = list(zip(nodes, nodes[1:], strict=False))
        cost = 0.0
        contains_target = False
        for frm, to in pairs:
            edge = edge_by_pair.get((frm, to))
            if edge is None:
                continue
            ek = edge_key(edge)
            cost += float(edge_weights.get(ek, 0.0))
            if ek in target_edges:
                contains_target = True
        # Include node runtimes along the path
        for n in nodes:
            cost += float(node_weights.get(n, 0.0))

        share = 0.0
        for frm, to in pairs:
            edge = edge_by_pair.get((frm, to))
            if edge is None:
                continue
            ek = edge_key(edge)
            if ek in sync_bottleneck_info:
                share = max(share, sync_bottleneck_info[ek][0])

        scored.append((nodes, share, cost, contains_target))

    # If any paths contain target sync edges, keep only those; else keep all.
    if any(item[3] for item in scored):
        scored = [item for item in scored if item[3]]

    # Pick max share, then max cost
    max_share = max(item[1] for item in scored) if scored else 0.0
    candidates = [item for item in scored if item[1] == max_share]
    if not candidates:
        return [], []

    chosen = max(candidates, key=lambda x: x[2])
    # strip contains_target in output
    candidates_out = [(nodes, share, cost) for nodes, share, cost, _ in scored]
    return chosen[0], candidates_out


def _calc_percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (pct / 100.0) * (len(vals) - 1)
    lower = int(k)
    upper = min(lower + 1, len(vals) - 1)
    weight = k - lower
    return vals[lower] * (1 - weight) + vals[upper] * weight


def _percentile_stats(
    edge_samples: Iterable[EdgeSample] | None,
    node_samples: Iterable[NodeSample] | None,
    fallback_edge_stats: Mapping[Tuple[str, str], StatSummary],
    fallback_node_stats: Mapping[str, StatSummary] | None,
    percentile: float,
) -> Tuple[Dict[Tuple[str, str], StatSummary], Dict[str, StatSummary]]:
    edge_stats_pct: Dict[Tuple[str, str], StatSummary] = {}
    node_stats_pct: Dict[str, StatSummary] = {}

    if edge_samples is not None:
        edge_values: Dict[Tuple[str, str], List[float]] = {}
        for s in edge_samples:
            edge_values.setdefault((s.edge_key, s.mechanism), []).append(s.transport_ms)
        for key, vals in edge_values.items():
            v = _calc_percentile(vals, percentile)
            edge_stats_pct[key] = StatSummary(count=len(vals), p50=v, p90=v, mean=v)

    if node_samples is not None:
        node_values: Dict[str, List[float]] = {}
        for ns in node_samples:
            node_values.setdefault(ns.fn_name, []).append(ns.runtime_ms)
        for fn, vals in node_values.items():
            v = _calc_percentile(vals, percentile)
            node_stats_pct[fn] = StatSummary(count=len(vals), p50=v, p90=v, mean=v)

    # Fill missing with fallback summaries (use p50 field even though it may be a different percentile)
    for key, summary in fallback_edge_stats.items():
        edge_stats_pct.setdefault(key, summary)
    if fallback_node_stats:
        for fn, summary in fallback_node_stats.items():
            node_stats_pct.setdefault(fn, summary)

    return edge_stats_pct, node_stats_pct


__all__ = [
    "rewrite_config_for_critical_path",
    "load_config",
    "dump_config",
]
