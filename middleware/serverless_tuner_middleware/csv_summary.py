from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Dict, List, Tuple

PERCENTILES = (10, 25, 50, 95, 99)


def percentile(values: List[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (pct / 100) * (len(sorted_vals) - 1)
    lower = floor(k)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = k - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def summarize_directory(csv_dir: Path) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    metrics: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)

    for path in csv_dir.glob("*.csv"):
        mechanism = "http" if "http" in path.name else "pubsub"
        region_pair = "ea1->we1" if "ea1_to_we1" in path.name else "ea1->ea1"

        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg_size = row.get("message_size") or row.get("payload_size") or "unknown"
                rate = row.get("invocation_rate") or row.get("rate") or "unknown"
                transport = row.get("transport_latency_ms") or row.get("latency_ms") or row.get("end_to_end_latency_ms")
                try:
                    latency = float(transport)
                except (TypeError, ValueError):
                    continue
                metrics[(mechanism, region_pair, msg_size, rate)].append(latency)

    summaries: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for key, vals in metrics.items():
        if not vals:
            continue
        summaries[key] = {f"p{p}": float(percentile(vals, p)) for p in PERCENTILES}
        summaries[key]["count"] = len(vals)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency CSVs into percentiles.")
    default_dir = Path(__file__).resolve().parent / "csv"
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=default_dir,
        help=f"Directory containing latency result CSVs. Defaults to {default_dir}",
    )
    parser.add_argument("--save-json", type=Path, help="Optional path to write the summary as JSON.")
    parser.add_argument("--plot", type=Path, help="Optional path to save a PNG plot (requires matplotlib).")
    args = parser.parse_args()

    summaries = summarize_directory(args.csv_dir)
    if not summaries:
        print(f"No CSV data found under {args.csv_dir}")
        return
    print("mechanism,region_pair,size,rate,count,p10,p25,p50,p95,p99")
    for key in sorted(summaries):
        stats = summaries[key]
        row = [
            *key,
            str(stats["count"]),
            *(f"{stats[f'p{p}']:.2f}" for p in PERCENTILES),
        ]
        print(",".join(row))

    if args.save_json:
        args.save_json.write_text(
            json.dumps(
                {
                    "metadata": {"source_dir": str(args.csv_dir), "percentiles": list(PERCENTILES)},
                    "data": {",".join(key): summaries[key] for key in summaries},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to {args.save_json}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("matplotlib not installed; skipping plot.")
            return

        categories = []
        values = []
        colors = []
        color_map = {"http": "#1f77b4", "pubsub": "#ff7f0e"}
        for key in sorted(summaries):
            mech, region, size, rate = key
            label = f"{mech}-{region}-{size}-{rate}"
            categories.append(label)
            values.append(summaries[key]["p50"])
            colors.append(color_map.get(mech, "#999999"))

        plt.figure(figsize=(max(8, len(categories) * 0.6), 5))
        plt.bar(categories, values, color=colors)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Transport latency p50 (ms)")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        plt.close()
        print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
