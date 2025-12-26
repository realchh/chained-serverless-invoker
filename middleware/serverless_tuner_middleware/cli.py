from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from .config import dump_config, load_config
from .critical_path import edge_key
from .logs import parse_events_from_lines
from .rewrite import rewrite_config_for_critical_path, _infer_region_pair
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
        payloads[edge_key(send)].append(send.payload_size)
        timestamps[edge_key(send)].append(send.ts_ms)

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

    rewritten = rewrite_config_for_critical_path(
        config,
        edge_stats=edge_stats,
        node_stats=node_stats,
        edge_context=edge_context,
    )
    dump_config(rewritten, args.config_out)

    print(f"Wrote updated config to {args.config_out}")


if __name__ == "__main__":
    main()
