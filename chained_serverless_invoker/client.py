import json
from enum import Enum, auto
from typing import Callable, Any, Optional

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
               payload: str,
               pubsub_topic: Optional[str] = None,
               http_url: Optional[str] = None,
               mode: InvocationMode = InvocationMode.DYNAMIC,
               **kwargs) -> Any:  # Returns a Future-like object
        """
        Invoker wrapper for dynamic invocation for either Pub/Sub or HTTP.
        Args:
            payload: The payload to send to the target (string).
            pubsub_topic: Pub/Sub topic name (if any).
            http_url: HTTP Endpoint URL (if any).
            mode: Invocation mode. If left unspecified, the invoker will dynamically choose between Pub/Sub and HTTP.
            **kwargs: Additional keyword arguments to pass to the invoker, such as auth_token.

        Returns:
            A future-like object (concurrent.futures.Future or google.api_core.future.Future).
        """
        if not http_url and not pubsub_topic:
            raise ValueError("At least one target (http_url or pubsub_topic) must be provided.")

        if mode == InvocationMode.FORCE_HTTP and not http_url:
            raise ValueError("InvocationMode is FORCE_HTTP, but no 'http_url' was provided.")

        if mode == InvocationMode.FORCE_PUBSUB and not pubsub_topic:
            raise ValueError("InvocationMode is FORCE_PUBSUB, but no 'pubsub_topic' was provided.")

        if mode == InvocationMode.DYNAMIC:
            if http_url and not pubsub_topic:
                mode = InvocationMode.FORCE_HTTP

            if pubsub_topic and not http_url:
                mode = InvocationMode.FORCE_PUBSUB

        payload_bytes = len(payload.encode('utf-8'))
        use_http = False
        if mode == InvocationMode.FORCE_PUBSUB:
            if payload_bytes > self.config.pubsub_max_bytes:
                raise ValueError(f"Message size ({payload_bytes/(8*1024*1024)} MB) is greater than the max PubSub message\n"
                                 f"size ({self.config.pubsub_max_bytes/(8*1024*1024)} MB).\n"
                                 f"Please use the HTTP invoker instead or reduce your message size. ")

            print("Using Pub/Sub invoker")
            use_http = False

        elif mode == InvocationMode.FORCE_HTTP:
            print("Using HTTP invoker")
            use_http = True

        elif mode == InvocationMode.DYNAMIC:
            print("Dynamically choosing invoker based on message size...")
            if payload_bytes > self.config.http_cutoff_bytes:
                print("Using HTTP invoker")
                use_http = True
            else:
                print("Using Pub/Sub invoker")
                use_http = False

        # Default fallback to Pub/Sub
        if use_http:
            return self.http_invoker.invoke(http_url, payload, **kwargs)
        else:
            return self.pubsub_invoker.invoke(pubsub_topic, payload, **kwargs)