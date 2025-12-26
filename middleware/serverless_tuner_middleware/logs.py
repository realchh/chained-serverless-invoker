from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class SendEvent:
    run_id: str
    taint: str
    from_fn: str
    to_fn: str
    mechanism: str
    ts_ms: int
    payload_size: int
    edge_id: str | None = None

    @property
    def edge_key(self) -> str:
        return self.edge_id or f"{self.from_fn}->{self.to_fn}"


@dataclass(frozen=True)
class RecvEvent:
    run_id: str
    taint: str
    fn_name: str
    ts_ms: int
    payload_size: int


def parse_events_from_lines(lines: Iterable[str]) -> Tuple[List[SendEvent], List[RecvEvent]]:
    """
    Parse NDJSON log lines into send/recv events based on the expected schema.

    Lines that cannot be parsed or do not contain the expected fields are skipped.
    """
    sends: List[SendEvent] = []
    recvs: List[RecvEvent] = []

    for line in lines:
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        message = record.get("message") or record.get("msg")
        inv = record.get("invoker") or {}
        # Cloud Run stdout lines often stash the payload under textPayload: "invoker_edge_send {...}"
        if not message and isinstance(record.get("textPayload"), str):
            text = record["textPayload"]
            if text.startswith("invoker_edge_send ") or text.startswith("invoker_edge_recv "):
                parts = text.split(" ", 1)
                if len(parts) == 2:
                    message = parts[0]
                    try:
                        inv = json.loads(parts[1])
                    except json.JSONDecodeError:
                        inv = {}
        if not isinstance(inv, dict):
            continue

        if message == "invoker_edge_send":
            run_id = inv.get("run_id")
            taint = inv.get("taint")
            from_fn = inv.get("from_fn")
            to_fn = inv.get("to_fn")
            mechanism = inv.get("mechanism")
            ts_ms = inv.get("ts_ms")
            payload_size = inv.get("payload_size")

            if None in (run_id, taint, from_fn, to_fn, mechanism, ts_ms, payload_size):
                continue

            sends.append(
                SendEvent(
                    run_id=str(run_id),
                    taint=str(taint),
                    from_fn=str(from_fn),
                    to_fn=str(to_fn),
                    mechanism=str(mechanism),
                    ts_ms=int(ts_ms),  # type: ignore[arg-type]
                    payload_size=int(payload_size),  # type: ignore[arg-type]
                    edge_id=inv.get("edge_id"),
                )
            )
        elif message == "invoker_edge_recv":
            run_id = inv.get("run_id")
            taint = inv.get("taint")
            fn_name = inv.get("fn_name")
            ts_ms = inv.get("ts_ms")
            payload_size = inv.get("payload_size")

            if None in (run_id, taint, fn_name, ts_ms, payload_size):
                continue

            recvs.append(
                RecvEvent(
                    run_id=str(run_id),
                    taint=str(taint),
                    fn_name=str(fn_name),
                    ts_ms=int(ts_ms),  # type: ignore[arg-type]
                    payload_size=int(payload_size),  # type: ignore[arg-type]
                )
            )

    return sends, recvs
