from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List

from .csv_summary import LATENCY_FIELDS, _parse_rate, _parse_size_kb


def _infer_region_pair(path: Path) -> str:
    name = path.name
    if "ea1_to_we1" in name:
        return "us-east1->us-west1"
    if "ea1_to_ea1" in name:
        return "us-east1->us-east1"
    return "us-east1->us-east1"


def _vcpu_mem_from_name(name: str) -> tuple[int | None, int | None]:
    m = re.search(r"(\d+)_cpu_(\d+)_gb", name)
    if not m:
        return None, None
    vcpu = int(m.group(1))
    mem_mb = int(m.group(2)) * 1024
    return vcpu, mem_mb


def _message_size_bytes(label: str | None) -> int | None:
    if label is None or label == "":
        return None
    size_kb = _parse_size_kb(str(label))
    if size_kb > 0:
        return int(size_kb * 1024)
    try:
        val = float(label)
        return int(val)
    except ValueError:
        return None


def normalize(csv_dir: Path, out_path: Path) -> None:
    rows: List[dict] = []
    for path in csv_dir.glob("*.csv"):
        if path.name == out_path.name:
            continue
        mechanism = "http" if "http" in path.name else "pubsub"
        region_pair = _infer_region_pair(path)
        vcpu, mem_mb = _vcpu_mem_from_name(path.name)
        if vcpu is None:
            vcpu = 1
        if mem_mb is None:
            mem_mb = 1024
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                size_bytes = _message_size_bytes(row.get("message_size") or row.get("payload_size"))
                rate = _parse_rate(row.get("invocation_rate") or row.get("rate") or "0")
                latency_val = None
                for field in LATENCY_FIELDS:
                    if row.get(field) not in (None, ""):
                        latency_val = row[field]
                        break
                try:
                    latency = float(latency_val) if latency_val is not None else None
                except (TypeError, ValueError):
                    latency = None
                if latency is None or size_bytes is None:
                    continue
                rows.append(
                    {
                        "mechanism": mechanism,
                        "region_pair": region_pair,
                        "message_size_bytes": size_bytes,
                        "rate_rps": rate,
                        "end_to_end_latency_ms": max(latency, 0.0),
                        "vcpu": vcpu if vcpu is not None else "",
                        "memory_mb": mem_mb if mem_mb is not None else "",
                        "source": path.name,
                    }
                )

    if not rows:
        print("No rows to write")
        return

    fieldnames = [
        "mechanism",
        "region_pair",
        "message_size_bytes",
        "rate_rps",
        "end_to_end_latency_ms",
        "vcpu",
        "memory_mb",
        "source",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    base = Path(__file__).parent / "csv"
    out = base / "latency_samples.csv"
    normalize(base, out)
