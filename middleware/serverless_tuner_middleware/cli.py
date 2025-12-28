from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from .config import WorkflowConfig, WorkflowEdge, dump_config, load_config
from .critical_path import edge_key
from .logs import parse_events_from_lines
from .model import REGRESSION_MODEL
from .rewrite import _infer_region_pair, rewrite_config_for_critical_path
from .stats import (
    aggregate_edge_stats,
    aggregate_node_stats,
    compute_edge_samples,
    compute_node_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite workflow config based on recorded invoker logs.")
    parser.add_argument("--logs", required=True, help="Path to NDJSON log file containing invoker_edge_send/recv.")
    parser.add_argument("--config-in", required=True, help="Path to existing workflow config JSON.")
    parser.add_argument("--config-out", required=True, help="Path to write the rewritten workflow config JSON.")
    parser.add_argument(
        "--mode",
        choices=["critical-path", "fastest-model"],
        default="critical-path",
        help="Rewrite strategy: critical-path greedy (default) or fastest per edge using model predictions.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed selection, path, and flip information.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=50.0,
        help="Percentile to optimize (e.g., 50, 75, 90, 99). Applies to observed stats and model quantile.",
    )

    args = parser.parse_args()

    log_path = Path(args.logs)
    with log_path.open("r", encoding="utf-8") as f:
        sends, recvs = parse_events_from_lines(f)

    edge_samples = compute_edge_samples(sends, recvs)
    node_samples = compute_node_samples(sends, recvs)
    edge_stats = aggregate_edge_stats(edge_samples)
    node_stats = aggregate_node_stats(node_samples)

    # Build edge context: region (from endpoint), avg payload, inferred rate.
    payloads: Dict[str, list[int]] = defaultdict(list)
    timestamps: Dict[str, list[int]] = defaultdict(list)
    for send in sends:
        payloads[send.edge_key].append(send.payload_size)
        timestamps[send.edge_key].append(send.ts_ms)

    edge_context: Dict[str, Tuple[str | None, int | None, float | None]] = {}
    config = load_config(args.config_in)
    for e in config.edges:
        key = edge_key(e)
        sizes = payloads.get(key, [])
        avg_payload = int(sum(sizes) / len(sizes)) if sizes else None
        ts = timestamps.get(key, [])
        duration_s = (max(ts) - min(ts)) / 1000.0 if len(ts) > 1 else 0.0
        rate_rps = (len(ts) / duration_s) if duration_s > 0 else None
        region = _infer_region_pair(e)
        edge_context[key] = (region, avg_payload, rate_rps)

    if args.mode == "fastest-model":
        # Choose the fastest predicted mechanism per edge (ignoring critical path).
        new_edges = []
        for e in config.edges:
            ctx = edge_context.get(edge_key(e))
            region = ctx[0] if ctx else None
            size_bytes = ctx[1] if ctx else None
            rate_rps = ctx[2] if ctx else None

            preds = {}
            quant = f"p{int(args.percentile)}"
            if region and size_bytes is not None and rate_rps is not None:
                for mech in ("http", "pubsub"):
                    preds[mech] = REGRESSION_MODEL.predict(
                        mech, region, quant, payload_bytes=size_bytes, rate_rps=rate_rps
                    )

            if preds and any(v is not None for v in preds.values()):
                best_mech = min((m for m, v in preds.items() if v is not None), key=lambda m: preds[m])  # type: ignore[arg-type]
                new_edges.append(
                    WorkflowEdge(
                        from_fn=e.from_fn,
                        to_fn=e.to_fn,
                        strategy=best_mech,
                        endpoint=e.endpoint,
                        topic=e.topic,
                        edge_id=e.edge_id,
                    )
                )
            else:
                new_edges.append(e)
        rewritten = WorkflowConfig(workflow_id=config.workflow_id, edges=new_edges)
    else:
        rewritten = rewrite_config_for_critical_path(
            config,
            edge_stats=edge_stats,
            node_stats=node_stats,
            edge_samples=edge_samples,
            node_samples=node_samples,
            edge_context=edge_context,
            verbose=args.verbose,
            percentile=args.percentile,
        )
    dump_config(rewritten, args.config_out)

    print(f"Wrote updated config to {args.config_out}")


if __name__ == "__main__":
    main()
