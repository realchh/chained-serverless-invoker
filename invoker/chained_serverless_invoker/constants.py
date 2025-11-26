# --- System Constants (Hard Limits & Safety) ---
# Google Cloud Pub/Sub hard limit (10MB): https://docs.cloud.google.com/pubsub/quotas
PUBSUB_MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

# Default thread pool size for asynchronous HTTP workers
DEFAULT_HTTP_MAX_WORKERS = 100

# Timeout for fire-and-forget logic (ensures the task is picked up by the executor) in seconds
DEFAULT_HTTP_FUTURE_TIMEOUT_SEC = 0.005

# Timeout executor's wait time to wait for http response in seconds
DEFAULT_HTTP_REQUEST_TIMEOUT_SEC = 300

# --- Dynamic Thresholds (Cutoff for switching strategy) ---
# Our research cutoff for "small" vs "large" messages (1MB), may change later with modelling
DEFAULT_HTTP_CUTOFF_BYTES = 1 * 1024 * 1024

# --- Latency Coefficients (Currently unused but future-proofed for the smart model) ---
HTTP_LATENCY_SLOPE = 0.0005  # seconds per byte
HTTP_LATENCY_INTERCEPT = 0.05  # seconds (base overhead)

PUBSUB_LATENCY_SLOPE = 0.0001  # seconds per byte
PUBSUB_LATENCY_INTERCEPT = 0.2  # seconds (base overhead)

# --- Pricing (USD) ---
PRICE_PER_GB_EGRESS = 0.12
PRICE_PER_PUBSUB_REQUEST = 0.0000004

# --- Metadata Keys ---
DEFAULT_META_KEY = "__invoker"
