from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from .logs import RecvEvent, SendEvent
from .utils import percentile


@dataclass(frozen=True)
class EdgeSample:
    run_id: str
    taint: str
    from_fn: str
    to_fn: str
    mechanism: str
    transport_ms: float
    edge_id: str | None = None

    @property
    def edge_key(self) -> str:
        return self.edge_id or f"{self.from_fn}->{self.to_fn}"


@dataclass(frozen=True)
class NodeSample:
    run_id: str
    fn_name: str
    runtime_ms: float


@dataclass(frozen=True)
class StatSummary:
    count: int
    p50: float
    p90: float
    mean: float


def _percentile(values: List[float], pct: float) -> float:
    """Backwards-compatible wrapper around utils.percentile."""
    return percentile(values, pct)


def _summarize(values: List[float]) -> StatSummary:
    if not values:
        return StatSummary(count=0, p50=0.0, p90=0.0, mean=0.0)

    return StatSummary(
        count=len(values),
        p50=_percentile(values, 50),
        p90=_percentile(values, 90),
        mean=mean(values),
    )


def compute_edge_samples(sends: Iterable[SendEvent], recvs: Iterable[RecvEvent]) -> List[EdgeSample]:
    recv_map: Dict[Tuple[str, str], RecvEvent] = {(r.run_id, r.taint): r for r in recvs}
    samples: List[EdgeSample] = []

    for send in sends:
        recv = recv_map.get((send.run_id, send.taint))
        if not recv:
            continue

        transport_ms = float(recv.ts_ms - send.ts_ms)
        if transport_ms < 0:
            transport_ms = 0.0

        samples.append(
            EdgeSample(
                run_id=send.run_id,
                taint=send.taint,
                from_fn=send.from_fn,
                to_fn=send.to_fn,
                mechanism=send.mechanism,
                transport_ms=transport_ms,
                edge_id=send.edge_id,
            )
        )

    return samples


def compute_node_samples(sends: Iterable[SendEvent], recvs: Iterable[RecvEvent]) -> List[NodeSample]:
    first_send_ts: Dict[Tuple[str, str], int] = {}
    for send in sends:
        key = (send.run_id, send.from_fn)
        first_send_ts[key] = min(first_send_ts.get(key, send.ts_ms), send.ts_ms)

    samples: List[NodeSample] = []
    for recv in recvs:
        key = (recv.run_id, recv.fn_name)
        send_ts = first_send_ts.get(key)
        if send_ts is None:
            continue

        runtime_ms = float(send_ts - recv.ts_ms)
        if runtime_ms < 0:
            runtime_ms = 0.0

        samples.append(NodeSample(run_id=recv.run_id, fn_name=recv.fn_name, runtime_ms=runtime_ms))

    return samples


def aggregate_edge_stats(samples: Iterable[EdgeSample]) -> Dict[Tuple[str, str], StatSummary]:
    groups: Dict[Tuple[str, str], List[float]] = {}
    for sample in samples:
        key = (sample.edge_key, sample.mechanism)
        groups.setdefault(key, []).append(sample.transport_ms)

    return {key: _summarize(vals) for key, vals in groups.items()}


def aggregate_node_stats(samples: Iterable[NodeSample]) -> Dict[str, StatSummary]:
    groups: Dict[str, List[float]] = {}
    for sample in samples:
        groups.setdefault(sample.fn_name, []).append(sample.runtime_ms)

    return {fn: _summarize(vals) for fn, vals in groups.items()}
