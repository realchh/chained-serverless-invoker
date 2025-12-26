from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .csv_summary import _load_raw, _parse_rate, _parse_size_kb, _trim, percentile


@dataclass(frozen=True)
class Coeffs:
    a_inv_rate: float
    b_size: float
    c_floor: float
    d_rate: float
    k_shift: float


@dataclass
class RegressionModel:
    """Simple per-mechanism/region/quantile regression model."""

    coeffs: Dict[Tuple[str, str, str], Coeffs]

    def _normalize_region_pair(self, region_pair: str) -> str:
        left, _, right = region_pair.partition("->")
        left = _REGION_ALIAS.get(left, left)
        right = _REGION_ALIAS.get(right, right)
        return f"{left}->{right}"

    def predict(
        self,
        mechanism: str,
        region_pair: str,
        quantile: str,
        *,
        payload_bytes: int,
        rate_rps: float,
    ) -> float | None:
        region_pair = self._normalize_region_pair(region_pair)
        key = (mechanism.lower(), region_pair, quantile.lower())
        coeff = self.coeffs.get(key)
        if not coeff:
            return None

        rate = max(rate_rps, 0.0)
        denom = coeff.k_shift + rate
        if denom <= 0:
            denom = 1e-6

        latency = (
            coeff.c_floor
            + coeff.a_inv_rate / denom
            + coeff.d_rate * rate
            + coeff.b_size * float(payload_bytes)
        )
        return max(latency, 0.0)


def _solve_linear(features: List[List[float]], targets: List[float], ridge: float = 1e-6) -> Optional[List[float]]:
    """
    Solve (X^T X) w = X^T y for w using a tiny Gaussian elimination.
    Returns None if the system is singular or not enough samples.
    """
    if not features or len(features) != len(targets):
        return None
    n = len(features[0])
    if len(features) < n:
        return None

    # Build normal equations with a tiny ridge term for stability.
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    b = [0.0 for _ in range(n)]
    for x, y in zip(features, targets):
        for i in range(n):
            b[i] += x[i] * y
            for j in range(n):
                A[i][j] += x[i] * x[j]
    for i in range(n):
        A[i][i] += ridge

    # Gaussian elimination with partial pivoting.
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[pivot_row][i]) < 1e-12:
            return None
        if pivot_row != i:
            A[i], A[pivot_row] = A[pivot_row], A[i]
            b[i], b[pivot_row] = b[pivot_row], b[i]

        pivot = A[i][i]
        for j in range(i, n):
            A[i][j] /= pivot
        b[i] /= pivot

        for r in range(i + 1, n):
            factor = A[r][i]
            if abs(factor) < 1e-12:
                continue
            for c in range(i, n):
                A[r][c] -= factor * A[i][c]
            b[r] -= factor * b[i]

    w = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * w[j] for j in range(i + 1, n))
        w[i] = b[i] - s
    return w


def _fit_for_quantile(
    samples: Iterable[Tuple[float, float, float]],
    k_grid: Iterable[float],
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Fit coefficients for a single quantile given scenario samples.
    Sample tuple: (payload_bytes, rate_rps, latency_ms_at_quantile)
    Returns (a_inv_rate, b_size, c_floor, d_rate, k_shift)
    """
    best: Optional[Tuple[float, float, float, float, float]] = None
    best_err = float("inf")
    for k in k_grid:
        feats: List[List[float]] = []
        ys: List[float] = []
        for payload_bytes, rate, latency in samples:
            rate = max(rate, 0.0)
            denom = k + rate
            denom = denom if denom > 0 else 1e-6
            feats.append([1.0, 1.0 / denom, rate, float(payload_bytes)])
            ys.append(latency)
        coeffs = _solve_linear(feats, ys)
        if coeffs is None:
            continue
        c_floor, a_inv_rate, d_rate, b_size = coeffs
        # Enforce simple constraints: non-negative floor, size slope, inverse-rate weight.
        c_floor = max(c_floor, 0.0)
        a_inv_rate = max(a_inv_rate, 0.0)
        b_size = max(b_size, 0.0)
        # Compute error.
        err = 0.0
        for payload_bytes, rate, latency in samples:
            rate = max(rate, 0.0)
            denom = k + rate
            denom = denom if denom > 0 else 1e-6
            pred = c_floor + a_inv_rate / denom + d_rate * rate + b_size * float(payload_bytes)
            diff = pred - latency
            err += diff * diff
        if err < best_err:
            best_err = err
            best = (a_inv_rate, b_size, c_floor, d_rate, k)
    return best


def fit_regression_model(csv_dir: Path, quantiles: Iterable[str] = ("p50", "p90", "p99")) -> RegressionModel:
    """
    Fit a regression model per mechanism/region/quantile from benchmark CSVs.
    """
    unified_path = csv_dir / "latency_samples.csv"
    scenarios: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
    if unified_path.exists():
        with unified_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                mech = row.get("mechanism")
                region = row.get("region_pair")
                try:
                    size_bytes = float(row.get("message_size_bytes", 0))
                    rate_rps = float(row.get("rate_rps", 0))
                    latency = float(row.get("end_to_end_latency_ms", 0))
                except (TypeError, ValueError):
                    continue
                if not mech or not region:
                    continue
                scenarios.setdefault((mech, region), []).append((size_bytes, rate_rps, latency))
    else:
        raw = _load_raw(csv_dir)
        for mech, region, size_label, rate_label, latency in raw:
            payload_kb = _parse_size_kb(size_label)
            rate = _parse_rate(rate_label)
            if payload_kb <= 0:
                continue
            scenarios.setdefault((mech, region), []).append((payload_kb * 1024.0, rate, latency))

    coeffs: Dict[Tuple[str, str, str], Coeffs] = {}
    k_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    for quantile in quantiles:
        q_pct = float(quantile.replace("p", ""))
        bucketed: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
        for (mech, region), vals in scenarios.items():
            latencies = [v[2] for v in vals]
            trimmed = _trim(latencies)
            if not trimmed:
                continue
            latency_q = percentile(trimmed, q_pct)
            if latency_q is None:
                continue
            size_median = percentile([v[0] for v in vals], 50) or 0.0
            rate_median = percentile([v[1] for v in vals], 50) or 0.0
            bucketed.setdefault((mech, region), []).append((size_median, rate_median, latency_q))

        for (mech, region), samples in bucketed.items():
            if len(samples) < 2:
                continue
            fit = _fit_for_quantile(samples, k_grid)
            if fit is None:
                continue
            a_inv_rate, b_size, c_floor, d_rate, k_shift = fit
            coeffs[(mech, region, quantile)] = Coeffs(
                a_inv_rate=a_inv_rate,
                b_size=b_size,
                c_floor=c_floor,
                d_rate=d_rate,
                k_shift=k_shift,
            )

    return RegressionModel(coeffs=coeffs)


def load_default_model() -> RegressionModel:
    csv_dir = Path(__file__).parent / "csv"
    return fit_regression_model(csv_dir)


REGRESSION_MODEL = load_default_model()

_REGION_ALIAS = {
    "us-east1": "ea1",
    "us-west1": "we1",
    "us-west2": "we2",
    "northamerica-northeast1": "nne1",
    # extend as more benchmark data is added
}

__all__ = ["RegressionModel", "Coeffs", "fit_regression_model", "REGRESSION_MODEL"]
