from __future__ import annotations

from typing import Iterable, List


def percentile(values: Iterable[float], pct: float) -> float:
    """Linear interpolation percentile; pct must be in (0, 100)."""
    if pct <= 0 or pct >= 100:
        raise ValueError("percentile must be between 0 and 100 (exclusive)")

    vals: List[float] = sorted(values)
    if not vals:
        return 0.0

    k = (pct / 100.0) * (len(vals) - 1)
    lower = int(k)
    upper = min(lower + 1, len(vals) - 1)
    weight = k - lower
    return vals[lower] * (1 - weight) + vals[upper] * weight
