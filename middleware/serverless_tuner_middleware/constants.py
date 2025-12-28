"""
Central place for tunable knobs and placeholder latency model coefficients.

These values are intentionally simple; they can be updated without touching logic.
"""

# Minimum p50 latency improvement (ms) required to flip a path edge.
GAIN_THRESHOLD_MS = 1

# Minimum share of runs (0-1) where a sync edge must be the bottleneck to treat it as critical offline.
SYNC_BOTTLENECK_RUN_SHARE_THRESHOLD = 0.20

# Sync-first critical-path tolerance: keep any path whose sync-edge count is at least this
# fraction of the maximum sync count across all source→sink paths. Default: 90%.
SYNC_CRITICAL_PATH_TOLERANCE = 0.90

# Sync bottleneck run-share tolerance band: include any path containing a sync edge whose run-share
# is within this fraction of the top run-share edge.
SYNC_RUN_SHARE_TOLERANCE = 0.10

# Maximum edges the tuner will flip in a single run. Prevents excessive churn.
MAX_EDGE_FLIPS_PER_RUN = 10

# Placeholder latency model coefficients (per mechanism).
# Units are milliseconds; slopes are per kilobyte to keep numbers small.
HTTP_LATENCY_BASE_MS = 50.0
HTTP_LATENCY_SLOPE_PER_KB_MS = 0.4

PUBSUB_LATENCY_BASE_MS = 200.0
PUBSUB_LATENCY_SLOPE_PER_KB_MS = 0.05
