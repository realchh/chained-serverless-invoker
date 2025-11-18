import json
from enum import Enum, auto
from typing import Callable, Any

from .constants import (
    PUBSUB_MAX_PAYLOAD_BYTES,
    DEFAULT_HTTP_CUTOFF_BYTES
)
from .invoker import AbstractInvoker
from .http_invoker import HttpInvoker
from .pubsub_invoker import PubSubInvoker
from .config import InvokerConfig


class InvocationMode(Enum):
    """Defines the reliability mode chosen by the user."""
    DYNAMIC = auto()
    FORCE_HTTP = auto()  # Fast/Cheap, At-Most-Once
    FORCE_PUBSUB = auto()  # Slower/Costly, At-Least-Once (reliable)


class DynamicInvoker:
    """
    Context class for the Strategy Pattern. Decides which invoker to use
    based on payload size, reliability preference, and current configuration.
    """

    def __init__(self,
                 pubsub_client: Any,
                 token_fetcher: Callable[[str], str],
                 config: InvokerConfig = None):

        self.config = config or InvokerConfig()

        # Initialize concrete strategies
        self.http_invoker: AbstractInvoker = HttpInvoker(token_fetcher)
        self.pubsub_invoker: AbstractInvoker = PubSubInvoker(pubsub_client)

    def invoke(self,
               target_identifier: str,
               payload: dict[str, Any],
               mode: InvocationMode = InvocationMode.DYNAMIC) -> None:

        json_payload = json.dumps(payload)
        payload_bytes = len(json_payload.encode('utf-8'))

        # --- Decision Logic (The Switch) ---

        # 1. Handle Hard Limits (Pub/Sub Max Size)
        if payload_bytes > self.config.pubsub_max_bytes:
            if mode == InvocationMode.FORCE_PUBSUB:
                raise ValueError(
                    f"Payload size ({payload_bytes} bytes) exceeds Pub/Sub hard limit "
                    f"({self.config.pubsub_max_bytes} bytes). Cannot force Pub/Sub."
                )
            # Default to HTTP if Pub/Sub limit is breached, regardless of mode
            # print(f"Payload > {self.config.pubsub_max_bytes}, forcing HTTP.")
            return self.http_invoker.invoke(target_identifier, json_payload)

        # 2. Handle Forced Modes
        if mode == InvocationMode.FORCE_PUBSUB:
            # print("Forcing Pub/Sub (Reliable).")
            return self.pubsub_invoker.invoke(target_identifier, json_payload)

        if mode == InvocationMode.FORCE_HTTP:
            # print("Forcing HTTP (Fast/Unreliable).")
            return self.http_invoker.invoke(target_identifier, json_payload)

        # 3. Handle Dynamic Mode (Based on hardcoded/configured cutoff)
        if mode == InvocationMode.DYNAMIC:
            # Use HTTP for payloads larger than the dynamic cutoff (latency optimization)
            if payload_bytes > self.config.http_cutoff_bytes:
                # print(f"Payload > {self.config.http_cutoff_bytes}, dynamically choosing HTTP (Faster).")
                return self.http_invoker.invoke(target_identifier, json_payload)

            # Use Pub/Sub for small payloads (reliability/cost savings on low-RPS Pub/Sub)
            # print("Payload is small, dynamically choosing Pub/Sub (Reliable/Better fit).")
            return self.pubsub_invoker.invoke(target_identifier, json_payload)

        # Should not be reached
        raise ValueError("Invalid Invocation Mode.")