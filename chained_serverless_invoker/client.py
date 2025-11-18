import json
from enum import Enum, auto
from typing import Callable, Any

from .constants import PUBSUB_MAX_PAYLOAD_BYTES
from .invokers import AbstractInvoker, HttpInvoker, PubSubInvoker
from .config import InvokerConfig


class InvocationMode(Enum):
    DYNAMIC = auto()
    FORCE_HTTP = auto()
    FORCE_PUBSUB = auto()


class DynamicInvoker:
    def __init__(self,
                 pubsub_client: Any,
                 token_fetcher: Callable[[str], str],
                 config: InvokerConfig = None):

        self.config = config or InvokerConfig()
        self.http_invoker = HttpInvoker(token_fetcher)
        self.pubsub_invoker = PubSubInvoker(pubsub_client)

    def invoke(self,
               target_identifier: str,
               payload: dict[str, Any],
               mode: InvocationMode = InvocationMode.DYNAMIC,
               **kwargs) -> Any:  # Returns a Future-like object

        json_payload = json.dumps(payload)
        payload_bytes = len(json_payload.encode('utf-8'))

        # Helper to execute and log/print decision if needed
        invoker = None

        # 1. Hard Limits
        if payload_bytes > self.config.pubsub_max_bytes:
            if mode == InvocationMode.FORCE_PUBSUB:
                # Fallback or Error? Let's fallback to HTTP for safety in a library
                # or raise error if strict compliance is needed.
                pass
            invoker = self.http_invoker

        elif mode == InvocationMode.FORCE_PUBSUB:
            invoker = self.pubsub_invoker
        elif mode == InvocationMode.FORCE_HTTP:
            invoker = self.http_invoker

        elif mode == InvocationMode.DYNAMIC:
            if payload_bytes > self.config.http_cutoff_bytes:
                invoker = self.http_invoker
            else:
                invoker = self.pubsub_invoker

        # Default fallback
        if invoker is None:
            invoker = self.pubsub_invoker

        return invoker.invoke(target_identifier, json_payload, **kwargs)