from dataclasses import dataclass
from . import constants


@dataclass
class InvokerConfig:
    """
    Configuration for the DynamicInvoker.
    Allows overriding defaults for experimentation.
    """
    http_cutoff_bytes: int = constants.DEFAULT_HTTP_CUTOFF_BYTES
    pubsub_max_bytes: int = constants.PUBSUB_MAX_PAYLOAD_BYTES

    # Cost/Performance Model Parameters
    http_slope: float = constants.HTTP_LATENCY_SLOPE
    http_intercept: float = constants.HTTP_LATENCY_INTERCEPT

    pubsub_slope: float = constants.PUBSUB_LATENCY_SLOPE
    pubsub_intercept: float = constants.PUBSUB_LATENCY_INTERCEPT