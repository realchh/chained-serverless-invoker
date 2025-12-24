"""
Central place for tunable knobs and placeholder latency model coefficients.

These values are intentionally simple; they can be updated without touching logic.
"""

# Minimum p50 latency improvement (ms) required to flip a path edge.
GAIN_THRESHOLD_MS = 0.001

# Placeholder latency model coefficients (per mechanism).
# Units are milliseconds; slopes are per kilobyte to keep numbers small.
HTTP_LATENCY_BASE_MS = 50.0
HTTP_LATENCY_SLOPE_PER_KB_MS = 0.4

PUBSUB_LATENCY_BASE_MS = 200.0
PUBSUB_LATENCY_SLOPE_PER_KB_MS = 0.05
