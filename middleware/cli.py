from __future__ import annotations

import argparse
from pathlib import Path

from serverless_tuner_middleware.config import dump_config, load_config
from serverless_tuner_middleware.logs import parse_events_from_lines
from serverless_tuner_middleware.rewrite import rewrite_config_for_critical_path
from serverless_tuner_middleware.stats import (
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

    config = load_config(args.config_in)
    rewritten = rewrite_config_for_critical_path(config, edge_stats=edge_stats, node_stats=node_stats)
    dump_config(rewritten, args.config_out)

    print(f"Wrote updated config to {args.config_out}")


if __name__ == "__main__":
    main()
