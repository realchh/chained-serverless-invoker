from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import sys

# Allow imports from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from middleware.serverless_tuner_middleware.logs import parse_events_from_lines
from middleware.serverless_tuner_middleware.stats import (
    compute_edge_samples,
    compute_node_samples,
    aggregate_edge_stats,
    aggregate_node_stats,
)
from middleware.serverless_tuner_middleware.config import load_config, dump_config, WorkflowConfig, WorkflowEdge
from middleware.serverless_tuner_middleware.critical_path import build_dag, critical_path_from_stats, edge_key
from middleware.serverless_tuner_middleware.model import REGRESSION_MODEL
from middleware.serverless_tuner_middleware.rewrite import rewrite_config_for_critical_path, _infer_region_pair

LOG_PATH = Path("results/tts/logs.ndjson")
CFG_IN = Path("results/tts/invoker_config.ndjson")
CFG_OUT = Path("results/tts/invoker_config_model_tuned.ndjson")


def main() -> None:
    with LOG_PATH.open() as f:
        sends, recvs = parse_events_from_lines(f)

    edge_samples = compute_edge_samples(sends, recvs)
    node_samples = compute_node_samples(sends, recvs)
    edge_stats = aggregate_edge_stats(edge_samples)
    node_stats = aggregate_node_stats(node_samples)

    edge_payloads = defaultdict(list)
    edge_ts = defaultdict(list)
    for s in sends:
        edge_payloads[edge_key(s)].append(s.payload_size)
        edge_ts[edge_key(s)].append(s.ts_ms)

    # Print edge stats
    print("Edge stats (p50/p90, count) and payloads:")
    for (ek, mech), stat in sorted(edge_stats.items()):
        sizes = edge_payloads.get(ek, [])
        avg_p = sum(sizes) / len(sizes) if sizes else 0
        min_p = min(sizes) if sizes else 0
        max_p = max(sizes) if sizes else 0
        ts = edge_ts.get(ek, [])
        dur = (max(ts) - min(ts)) / 1000.0 if len(ts) > 1 else 0.0
        rate = len(ts) / dur if dur > 0 else 0.0
        print(
            f"  {ek} [{mech}] p50={stat.p50:.2f} p90={stat.p90:.2f} count={stat.count} | "
            f"payload_avg={int(avg_p)}B min={min_p} max={max_p} rate={rate:.3f} rps"
        )

    # Print node stats
    print("\nNode stats (p50 runtime):")
    for fn, stat in sorted(node_stats.items()):
        print(f"  {fn}: p50={stat.p50:.2f} count={stat.count}")

    cfg = load_config(CFG_IN)
    dag = build_dag(cfg)
    edge_weights = {edge_key(e): edge_stats.get((edge_key(e), e.strategy.lower()), None) for e in cfg.edges}
    edge_weights = {k: (v.p50 if v else 0.0) for k, v in edge_weights.items()}
    crit_cost, crit_nodes = critical_path_from_stats(dag, edge_stats=edge_weights, node_stats=node_stats)
    print(f"\nCurrent critical path: {' -> '.join(crit_nodes)} (weight={crit_cost:.2f} ms)")

    # Model predictions
    print("\nModel predictions (p50) for each edge:")
    predictions = {}
    for e in cfg.edges:
        reg = _infer_region_pair(e)
        sizes = edge_payloads.get(edge_key(e))
        if not sizes or not reg:
            continue
        avg_payload = int(sum(sizes) / len(sizes))
        ts = edge_ts[edge_key(e)]
        dur = (max(ts) - min(ts)) / 1000.0 if len(ts) > 1 else 0.0
        rate = len(ts) / dur if dur > 0 else 0.0
        preds = {mech: REGRESSION_MODEL.predict(mech, reg, "p50", payload_bytes=avg_payload, rate_rps=rate) for mech in ("http", "pubsub")}
        predictions[edge_key(e)] = (reg, avg_payload, rate, preds)
        print(
            f"  {edge_key(e)} region={reg} payload={avg_payload}B rate={rate:.3f}rps "
            f"http={preds['http']} pubsub={preds['pubsub']}"
        )

    # Choose fastest predicted mechanism per edge where available.
    new_edges = []
    for e in cfg.edges:
        pred = predictions.get(edge_key(e))
        if pred:
            preds = pred[3]
            best_mech = min((m for m, v in preds.items() if v is not None), key=lambda m: preds[m]) if any(
                v is not None for v in preds.values()
            ) else e.strategy
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

    new_cfg = WorkflowConfig(workflow_id=cfg.workflow_id, edges=new_edges)
    dump_config(new_cfg, CFG_OUT)
    print("\nRewritten edges:")
    for e in new_cfg.edges:
        print(f"  {e.from_fn}->{e.to_fn}: strategy={e.strategy}")
    print(f"\nWrote {CFG_OUT}")


if __name__ == "__main__":
    main()
