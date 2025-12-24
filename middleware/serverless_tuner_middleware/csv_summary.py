from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Dict, List, Tuple

PERCENTILES = (10, 25, 50, 95, 99)
LATENCY_FIELDS = [
    "end_to_end_latency_ms",
    "total_latency_ms",
    "total_http_latency_ms",
    "total_pubsub_latency_ms",
    "latency_ms",
    "transport_latency_ms",
]


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
                val = None
                for field in LATENCY_FIELDS:
                    if row.get(field) not in (None, ""):
                        val = row[field]
                        break
                try:
                    latency = float(val) if val is not None else None
                except (TypeError, ValueError):
                    continue
                if latency is None:
                    continue
                # Clamp negative transport/acks to zero to avoid skew.
                latency = max(latency, 0.0)
                metrics[(mechanism, region_pair, msg_size, rate)].append(latency)

    summaries: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for key, vals in metrics.items():
        if not vals:
            continue
        stats: Dict[str, float] = {}
        for p in PERCENTILES:
            val = percentile(vals, p)
            if val is not None:
                stats[f"p{p}"] = float(val)
        stats["count"] = len(vals)
        summaries[key] = stats
    return summaries


def _trim(values: List[float], low_pct: float = 1.0, high_pct: float = 99.0) -> List[float]:
    """Trim extremes (e.g., cold starts) by percentile range."""
    if not values:
        return values
    sorted_vals = sorted(values)
    k_low = int(len(sorted_vals) * low_pct / 100)
    k_high = int(len(sorted_vals) * high_pct / 100)
    k_high = min(k_high, len(sorted_vals) - 1)
    return sorted_vals[k_low : k_high + 1]


def _load_raw(csv_dir: Path) -> List[Tuple[str, str, str, str, float]]:
    """Return raw rows as (mechanism, region_pair, size, rate, latency_ms)."""
    rows: List[Tuple[str, str, str, str, float]] = []
    for path in csv_dir.glob("*.csv"):
        mechanism = "http" if "http" in path.name else "pubsub"
        region_pair = "ea1->we1" if "ea1_to_we1" in path.name else "ea1->ea1"
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg_size = row.get("message_size") or row.get("payload_size") or "unknown"
                rate = row.get("invocation_rate") or row.get("rate") or "unknown"
                val = None
                for field in LATENCY_FIELDS:
                    if row.get(field) not in (None, ""):
                        val = row[field]
                        break
                try:
                    latency = float(val) if val is not None else None
                except (TypeError, ValueError):
                    continue
                if latency is None:
                    continue
                latency = max(latency, 0.0)
                rows.append((mechanism, region_pair, msg_size, rate, latency))
    return rows


def plot_scenarios(csv_dir: Path, output_prefix: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping scatter/box plots.")
        return

    raw = _load_raw(csv_dir)
    if not raw:
        print(f"No data to plot under {csv_dir}")
        return

    for mech in ("http", "pubsub"):
        filtered = [(r, s, rate, lat) for m, r, s, rate, lat in raw if m == mech]
        if not filtered:
            continue

        # Group by scenario label for a compact boxplot: region|size|rate
        groups: Dict[str, List[float]] = defaultdict(list)
        for region, size, rate, lat in filtered:
            label = f"{region}-{size}-{rate}"
            groups[label].append(lat)

        groups = {label: _trim(vals) for label, vals in groups.items() if vals}

        labels = sorted(groups.keys())
        data = [groups[label] for label in labels]

        plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Transport latency (ms)")
        plt.title(f"{mech.upper()} latency per scenario")
        plt.tight_layout()
        outfile = output_prefix.with_name(f"{output_prefix.stem}_{mech}{output_prefix.suffix or '.png'}")
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"Wrote {mech} plot to {outfile}")


def _parse_size_kb(size_label: str) -> float:
    if size_label.lower().startswith("small"):
        return 10.0
    if size_label.lower().startswith("large"):
        return 1024.0
    try:
        return float(size_label)
    except ValueError:
        return 0.0


def _parse_rate(rate_label: str) -> float:
    if rate_label.endswith("_rps"):
        rate_label = rate_label.replace("_rps", "")
    try:
        return float(rate_label)
    except ValueError:
        return 0.0


def _group_trimmed(raw: List[Tuple[str, str, str, str, float]]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Nested dict: mechanism -> region -> scenario_label (size-rate) -> trimmed samples.
    """
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for mech, region, size, rate, lat in raw:
        label = f"{size}-{rate}"
        grouped[mech][region][label].append(lat)
    # Trim per scenario
    for mech in grouped:
        for region in grouped[mech]:
            for label, vals in list(grouped[mech][region].items()):
                grouped[mech][region][label] = _trim(vals)
    return grouped


def plot_modes(csv_dir: Path, output_prefix: Path, modes: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping requested plots.")
        return

    raw = _load_raw(csv_dir)
    if not raw:
        print(f"No data to plot under {csv_dir}")
        return

    grouped = _group_trimmed(raw)
    mechanisms = sorted(grouped.keys())
    regions = sorted({region for mech in grouped.values() for region in mech})

    def next_path(suffix: str) -> Path:
        return output_prefix.with_name(f"{output_prefix.stem}_{suffix}{output_prefix.suffix or '.png'}")

    if "bars" in modes:
        for mech in mechanisms:
            labels = []
            p50 = []
            p95 = []
            counts = []
            for region in regions:
                scenarios = grouped[mech].get(region, {})
                for scenario in sorted(scenarios):
                    vals = scenarios[scenario]
                    if not vals:
                        continue
                    labels.append(f"{region}-{scenario}")
                    p50.append(percentile(vals, 50) or 0.0)
                    p95.append(percentile(vals, 95) or 0.0)
                    counts.append(len(vals))

            if not labels:
                continue

            x = list(range(len(labels)))
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
            ax.bar(x, p50, color="#1f77b4", alpha=0.7, label="p50")
            ax.scatter(x, p95, color="#d62728", marker="o", label="p95")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=75, ha="right")
            ax.set_ylabel("Transport latency (ms)")
            ax.set_title(f"{mech.upper()} p50/p95 by scenario (trimmed 1–99 pct)")
            ax.legend()
            fig.tight_layout()
            outfile = next_path(f"{mech}_bars")
            fig.savefig(outfile, dpi=150)
            plt.close(fig)
            print(f"Wrote {mech} bars plot to {outfile}")

    if "violins" in modes:
        for mech in mechanisms:
            for region in regions:
                scenarios = grouped[mech].get(region, {})
                labels = sorted(scenarios.keys())
                data = [scenarios[label] for label in labels if scenarios[label]]
                labels = [label for label in labels if scenarios[label]]
                if not data:
                    continue

                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
                parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#1f77b4" if mech == "http" else "#ff7f0e")
                    pc.set_alpha(0.6)
                # Overlay p50 and p95 markers
                positions = list(range(1, len(labels) + 1))
                p50 = [percentile(d, 50) or 0.0 for d in data]
                p95 = [percentile(d, 95) or 0.0 for d in data]
                ax.scatter(positions, p50, color="#000000", marker="o", s=10, label="p50")
                ax.scatter(positions, p95, color="#d62728", marker="x", s=14, label="p95")
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=75, ha="right")
                ax.set_ylabel("Transport latency (ms)")
                ax.set_title(f"{mech.upper()} {region} violins (trimmed 1–99 pct)")
                ax.legend()
                fig.tight_layout()
                outfile = next_path(f"{mech}_{region}_violins")
                fig.savefig(outfile, dpi=150)
                plt.close(fig)
                print(f"Wrote {mech} {region} violins to {outfile}")

    if "strip" in modes:
        for mech in mechanisms:
            for region in regions:
                scenarios = grouped[mech].get(region, {})
                labels = sorted(scenarios.keys())
                if not labels:
                    continue
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
                for idx, label in enumerate(labels):
                    vals = scenarios[label]
                    if not vals:
                        continue
                    jitter = [idx + (0.1 * (i / len(vals) - 0.5)) for i in range(len(vals))]
                    ax.scatter(jitter, vals, alpha=0.5, s=6, color="#1f77b4" if mech == "http" else "#ff7f0e")
                ax.set_xticks(list(range(len(labels))))
                ax.set_xticklabels(labels, rotation=75, ha="right")
                ax.set_ylabel("Transport latency (ms, log scale)")
                ax.set_yscale("log")
                ax.set_title(f"{mech.upper()} {region} strip (trimmed 1–99 pct)")
                fig.tight_layout()
                outfile = next_path(f"{mech}_{region}_strip")
                fig.savefig(outfile, dpi=150)
                plt.close(fig)
                print(f"Wrote {mech} {region} strip plot to {outfile}")

    if "regress" in modes:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            print("numpy not installed; skipping regression plots.")
            return

        for mech in mechanisms:
            for region in regions:
                scenarios = grouped[mech].get(region, {})
                points = []
                for label, vals in scenarios.items():
                    if not vals:
                        continue
                    size_label, rate_label = label.split("-", 1)
                    size_val = _parse_size_kb(size_label)
                    rate = _parse_rate(rate_label)
                    for v in vals:
                        points.append((size_val, rate, v))
                if not points:
                    continue

                # Fit simple linear model: latency = a * size_kb + b * rate + c
                X = np.array([[p[0], p[1], 1.0] for p in points])
                y = np.array([p[2] for p in points])
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                a, b, c = coef

                # Scatter by size, color by rate bucket
                fig, ax = plt.subplots(figsize=(6, 4))
                rates = sorted({p[1] for p in points})
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                for idx, r in enumerate(rates):
                    xs = [p[0] for p in points if p[1] == r]
                    ys = [p[2] for p in points if p[1] == r]
                    ax.scatter(xs, ys, color=colors[idx % len(colors)], alpha=0.6, s=12, label=f"{r} rps")
                    # Regression line per rate
                    xs_line = np.array([min(xs), max(xs)])
                    ys_line = a * xs_line + b * r + c
                    ax.plot(xs_line, ys_line, color=colors[idx % len(colors)], linestyle="--", linewidth=1)

                ax.set_xlabel("Payload size (KB)")
                ax.set_ylabel("End-to-end latency (ms)")
                ax.set_title(f"{mech.upper()} {region} linear fit: latency = {a:.2f}*size + {b:.2f}*rate + {c:.2f}")
                ax.legend()
                fig.tight_layout()
                outfile = output_prefix.with_name(
                    f"{output_prefix.stem}_{mech}_{region}_regress{output_prefix.suffix or '.png'}"
                )
                fig.savefig(outfile, dpi=150)
                plt.close(fig)
                print(f"Wrote {mech} {region} regression plot to {outfile}")

    if "surface" in modes:
        try:
            import numpy as np  # type: ignore
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            print("numpy/matplotlib 3D not installed; skipping surface plots.")
            return

        for mech in mechanisms:
            for region in regions:
                scenarios = grouped[mech].get(region, {})
                points = []
                for label, vals in scenarios.items():
                    if not vals:
                        continue
                    size_label, rate_label = label.split("-", 1)
                    size_val = _parse_size_kb(size_label)
                    rate = _parse_rate(rate_label)
                    for v in vals:
                        points.append((size_val, rate, v))
                if not points:
                    continue

                xs = np.array([p[0] for p in points])
                rs = np.array([p[1] for p in points])
                ys = np.array([p[2] for p in points])

                # Fit simple plane: latency = a*size + b*rate + c
                X = np.column_stack((xs, rs, np.ones_like(xs)))
                coef, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
                a, b, c = coef

                size_grid = np.linspace(xs.min(), xs.max(), 30)
                rate_grid = np.linspace(rs.min(), rs.max(), 30)
                S, R = np.meshgrid(size_grid, rate_grid)
                Z = a * S + b * R + c

                fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(S, R, Z, alpha=0.4, color="#1f77b4" if mech == "http" else "#ff7f0e")
                ax.scatter(xs, rs, ys, color="#000000", s=8, alpha=0.6)
                ax.set_xlabel("Payload size (KB)")
                ax.set_ylabel("Rate (rps)")
                ax.set_zlabel("End-to-end latency (ms)")
                ax.set_title(f"{mech.upper()} {region} plane fit: {a:.2f}*size + {b:.2f}*rate + {c:.2f}")
                fig.tight_layout()
                outfile = output_prefix.with_name(
                    f"{output_prefix.stem}_{mech}_{region}_surface{output_prefix.suffix or '.png'}"
                )
                fig.savefig(outfile, dpi=150)
                plt.close(fig)
                print(f"Wrote {mech} {region} surface plot to {outfile}")


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
    parser.add_argument(
        "--plot-scenarios",
        type=Path,
        help="Optional prefix path to save per-mechanism scenario boxplots (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-modes",
        type=str,
        default="",
        help="Comma-separated plot styles among: bars, violins, strip (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-prefix",
        type=Path,
        default=Path("latency"),
        help="Prefix path (stem) for plot-modes outputs, suffix is added automatically.",
    )
    parser.add_argument(
        "--plot-surface",
        action="store_true",
        help="Generate a simple 3D surface (size vs rate vs latency) per mechanism/region (requires matplotlib).",
    )
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

    if args.plot_scenarios:
        plot_scenarios(args.csv_dir, args.plot_scenarios)

    if args.plot_modes:
        modes = [m.strip() for m in args.plot_modes.split(",") if m.strip()]
        if modes:
            plot_modes(args.csv_dir, args.plot_prefix, modes)


if __name__ == "__main__":
    main()
