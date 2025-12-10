import base64
import json
import logging
import time
import uuid
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple

from .config import InvokerConfig
from .constants import DEFAULT_META_KEY
from .invokers import HttpInvoker, PubSubInvoker
from .invokers.types import EdgeConfig, InvokerMetadata

logger = logging.getLogger(__name__)


class InvocationMode(Enum):
    DYNAMIC = auto()
    FORCE_HTTP = auto()
    FORCE_PUBSUB = auto()


class DynamicInvoker:
    def __init__(self, pubsub_client: Any, token_fetcher: Callable[[str], str], config: Optional[InvokerConfig] = None):
        self.config = config or InvokerConfig()
        self.http_invoker = HttpInvoker(token_fetcher)
        self.pubsub_invoker = PubSubInvoker(pubsub_client)

    def invoke(
        self,
        payload: str,
        pubsub_topic: Optional[str] = None,
        http_url: Optional[str] = None,
        mode: InvocationMode = InvocationMode.DYNAMIC,
        log_decision: bool = True,
        **kwargs: Any,
    ) -> Any:  # Returns a Future-like object
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

        payload_bytes = len(payload.encode("utf-8"))
        use_http = False
        if mode == InvocationMode.FORCE_PUBSUB:
            if payload_bytes > self.config.pubsub_max_bytes:
                raise ValueError(
                    f"Message size ({payload_bytes / (8 * 1024 * 1024)} MB) is greater than\n"
                    f"the max PubSub message size ({self.config.pubsub_max_bytes / (8 * 1024 * 1024)} MB).\n"
                    f"Please use the HTTP invoker instead or reduce your message size."
                )

            print("Force use Pub/Sub invoker")
            use_http = False

        elif mode == InvocationMode.FORCE_HTTP:
            print("Force use HTTP invoker")
            use_http = True

        elif mode == InvocationMode.DYNAMIC:
            print("Dynamically choosing invoker based on message size...")
            if payload_bytes > self.config.http_cutoff_bytes:
                print("Using HTTP invoker")
                use_http = True
            else:
                print("Using Pub/Sub invoker")
                use_http = False

        if log_decision:
            mechanism = "http" if use_http else "pubsub"
            logger.info(
                "invoker_invoke",
                extra={
                    "invoker": {
                        "mechanism": mechanism,
                        "mode": mode.name,
                        "ts_ms": int(time.time() * 1000),
                        "payload_size": payload_bytes,
                    }
                },
            )

        # Default fallback to Pub/Sub
        if use_http and http_url:
            return self.http_invoker.invoke(http_url, payload, **kwargs)
        elif not use_http and pubsub_topic:
            return self.pubsub_invoker.invoke(pubsub_topic, payload, **kwargs)
        else:
            raise ValueError("No target provided.")

    def invoke_edge(
        self,
        meta: InvokerMetadata,
        target_fn: str,
        payload: Dict[str, Any],
        *,
        http_url: str | None = None,
        pubsub_topic: str | None = None,
        mode: InvocationMode = InvocationMode.DYNAMIC,
        meta_key: str = DEFAULT_META_KEY,
        **kwargs: Any,
    ) -> Any:
        edge = next((e for e in meta.edges if e.to == target_fn), None)

        # 1) Use static edge strategy if provided
        if edge:
            strat = edge.strategy.lower()
            if strat == "http":
                mode = InvocationMode.FORCE_HTTP
            elif strat == "pubsub":
                mode = InvocationMode.FORCE_PUBSUB

            http_url = http_url or edge.endpoint
            pubsub_topic = pubsub_topic or edge.topic

        # 2) If still DYNAMIC, but only one target is set, normalize like invoke()
        if mode == InvocationMode.DYNAMIC:
            if http_url and not pubsub_topic:
                mode = InvocationMode.FORCE_HTTP
            elif pubsub_topic and not http_url:
                mode = InvocationMode.FORCE_PUBSUB

        # 3) Attach metadata for the receiver
        taint = uuid.uuid4().hex
        edge_id = edge.edge_id if edge else None

        payload[meta_key] = {
            "fn_name": target_fn,
            "run_id": meta.run_id,
            "taint": taint,
            "edges": [e.__dict__ for e in meta.edges],
        }

        payload_str = json.dumps(payload)
        payload_bytes = len(payload_str.encode("utf-8"))
        send_start_ms = int(time.time() * 1000)

        # 4) Decide mechanism based on mode + size (must match actual behavior)
        if mode == InvocationMode.FORCE_HTTP:
            mechanism = "http"
        elif mode == InvocationMode.FORCE_PUBSUB:
            mechanism = "pubsub"
        else:
            # DYNAMIC and both targets available → size-based rule
            if payload_bytes > self.config.http_cutoff_bytes:
                mechanism = "http"
                mode = InvocationMode.FORCE_HTTP
            else:
                mechanism = "pubsub"
                mode = InvocationMode.FORCE_PUBSUB

        # 5) Log SEND side
        logger.info(
            "invoker_edge_send",
            extra={
                "invoker": {
                    "run_id": meta.run_id,
                    "taint": taint,
                    "from_fn": meta.fn_name,
                    "to_fn": target_fn,
                    "edge_id": edge_id,
                    "mechanism": mechanism,
                    "ts_ms": send_start_ms,
                    "payload_size": len(payload_str),
                }
            },
        )

        # 6) Delegate to the existing invoke(); mode is now FORCE_* so it won't re-decide
        return self.invoke(
            payload_str,
            pubsub_topic=pubsub_topic,
            http_url=http_url,
            mode=mode,
            log_decision=False,  # avoid duplicate logging; send-side log is invoker_edge_send
            **kwargs,
        )


def bootstrap_from_request(request: Any, meta_key: str = DEFAULT_META_KEY) -> Tuple[Optional[InvokerMetadata], Dict[str, Any]]:
    """
    Parse incoming request/event payload for DAG metadata.

    Supports:
    - HTTP frameworks exposing .get_data() or .data (bytes/str)
    - Pub/Sub-style event dicts with a base64-encoded "data" field
    """
    recv_start_ms = int(time.time() * 1000)

    raw_body: bytes | str | None = None

    if hasattr(request, "get_data"):
        raw_body = request.get_data()
    elif hasattr(request, "data"):
        raw_body = request.data
    elif isinstance(request, dict) and "data" in request:
        raw_body = request["data"]

    if isinstance(raw_body, str):
        # Pub/Sub delivers base64-encoded strings in some runtimes
        try:
            raw_body = base64.b64decode(raw_body)
        except Exception:
            raw_body = raw_body.encode("utf-8", errors="ignore")

    payload: Dict[str, Any] = {}

    if raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return None, {}

    meta_dict = payload.get(meta_key)
    if not meta_dict:
        return None, payload

    fn_name = meta_dict.get("fn_name", "")
    run_id = meta_dict.get("run_id", "")
    taint = meta_dict.get("taint", "")
    edges = [EdgeConfig(**e) for e in meta_dict.get("edges", [])]

    logger.info(
        "invoker_edge_recv",
        extra={
            "invoker": {
                "run_id": run_id,
                "taint": taint,
                "fn_name": fn_name,
                "ts_ms": recv_start_ms,
                "payload_size": len(raw_body) if raw_body else 0,
            }
        },
    )

    meta = InvokerMetadata(
        fn_name=fn_name,
        run_id=run_id,
        taint=taint,
        edges=edges,
    )

    return meta, payload
