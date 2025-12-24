from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

TOTAL_FIELDS = [
    "end_to_end_latency_ms",
    "total_http_latency_ms",
    "total_pubsub_latency_ms",
    "total_latency_ms",
    "latency_ms",
    "transport_latency_ms",
]

OVERHEAD_FIELDS = [
    "publisher_processing_latency_ms",
    "request_latency_ms",
]


def find_latency(row: dict) -> Optional[str]:
    for field in TOTAL_FIELDS:
        val = row.get(field)
        if val not in (None, ""):
            return val
    return None


def find_overhead(row: dict) -> Optional[str]:
    for field in OVERHEAD_FIELDS:
        val = row.get(field)
        if val not in (None, ""):
            return val
    return None


def rewrite_csv(path: Path) -> None:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    out_rows = []
    for row in rows:
        latency_val = find_latency(row)
        if latency_val is None:
            continue
        overhead_val = find_overhead(row) or ""
        out_rows.append(
            {
                "scenario": row.get("scenario", ""),
                "message_size": row.get("message_size") or row.get("payload_size") or "",
                "invocation_rate": row.get("invocation_rate") or row.get("rate") or "",
                "invocation_index": row.get("invocation_index") or row.get("index") or "",
                "end_to_end_latency_ms": latency_val,
                "publisher_processing_latency_ms": overhead_val,
            }
        )

    if not out_rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "message_size",
                "invocation_rate",
                "invocation_index",
                "end_to_end_latency_ms",
                "publisher_processing_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def rewrite_directory(csv_dir: Path) -> None:
    for path in csv_dir.glob("*.csv"):
        rewrite_csv(path)


if __name__ == "__main__":
    rewrite_directory(Path(__file__).resolve().parent / "csv")
