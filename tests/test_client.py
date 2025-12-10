import base64
import json
import logging
from unittest.mock import patch

import pytest

from invoker.chained_serverless_invoker import DynamicInvoker, InvocationMode
from invoker.chained_serverless_invoker.client import bootstrap_from_request
from invoker.chained_serverless_invoker.constants import DEFAULT_META_KEY
from invoker.chained_serverless_invoker.invokers.types import EdgeConfig, InvokerMetadata

# Define some dummy data
SMALL_PAYLOAD = "small"
LARGE_PAYLOAD = "x" * (1024 * 1024 * 2)  # 2MB


# ========== Existing tests for basic DynamicInvoker.invoke ==========


def test_dynamic_mode_selects_pubsub_for_small_payload(mock_pubsub_client, mock_token_fetcher):
    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    # Call with small payload (and both targets available)
    invoker.invoke(
        pubsub_topic="projects/my-project/topics/my-topic",
        payload=SMALL_PAYLOAD,
        http_url="https://my-func.run.app",
        mode=InvocationMode.DYNAMIC,
    )

    # Assert Pub/Sub was called
    mock_pubsub_client.publish.assert_called_once()


def test_dynamic_mode_switches_to_http_for_large_payload(mock_pubsub_client, mock_token_fetcher):
    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    with patch("chained_serverless_invoker.invokers.http_invoker.requests.post") as mock_post:
        mock_post.return_value.status_code = 200

        future = invoker.invoke(
            pubsub_topic="projects/my-project/topics/my-topic",
            payload=LARGE_PAYLOAD,
            http_url="https://my-func.run.app",
            mode=InvocationMode.DYNAMIC,
        )

        future.result()

        mock_post.assert_called_once()
        mock_pubsub_client.publish.assert_not_called()


def test_force_http_works(mock_pubsub_client, mock_token_fetcher):
    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    with patch("chained_serverless_invoker.invokers.http_invoker.requests.post") as mock_post:
        mock_post.return_value.status_code = 200

        future = invoker.invoke(
            http_url="https://my-func.run.app", payload=SMALL_PAYLOAD, mode=InvocationMode.FORCE_HTTP
        )

        future.result()

        mock_post.assert_called_once()


def test_missing_target_raises_error(mock_pubsub_client, mock_token_fetcher):
    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    with pytest.raises(ValueError, match="At least one target"):
        invoker.invoke(http_url=None, payload="test")


def test_invoke_edge_http_forced_by_edge_strategy(mock_pubsub_client, mock_token_fetcher, caplog):
    """
    Edge with strategy='http' should force HTTP, log invoker_edge_send,
    and publish NO Pub/Sub messages.
    """
    caplog.set_level(logging.INFO, logger="chained_serverless_invoker.client")

    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    edges = [
        EdgeConfig(
            to="B",
            strategy="http",
            endpoint="https://example.com/next",
            topic="projects/x/topics/ignored",
            edge_id="A->B",
        )
    ]
    meta = InvokerMetadata(fn_name="A", run_id="run-1", taint="root", edges=edges)
    payload = {"foo": "bar"}

    with patch("chained_serverless_invoker.invokers.http_invoker.requests.post") as mock_post:
        mock_post.return_value.status_code = 200

        result = invoker.invoke_edge(
            meta,
            target_fn="B",
            payload=payload,
        )

        # Returned future from HttpInvoker
        future = result
        future.result()

        mock_post.assert_called_once()
        mock_pubsub_client.publish.assert_not_called()

    # Metadata injected into payload
    assert DEFAULT_META_KEY in payload
    meta_block = payload[DEFAULT_META_KEY]
    assert meta_block["fn_name"] == "B"
    assert meta_block["run_id"] == "run-1"
    assert "taint" in meta_block
    assert isinstance(meta_block["edges"], list)

    # Logging: invoker_edge_send record with correct fields
    send_records = [r for r in caplog.records if r.message == "invoker_edge_send"]
    assert send_records, "no invoker_edge_send log found"

    rec = send_records[0]
    inv = rec.invoker
    assert inv["run_id"] == "run-1"
    assert inv["from_fn"] == "A"
    assert inv["to_fn"] == "B"
    assert inv["edge_id"] == "A->B"
    assert inv["mechanism"] == "http"
    assert inv["payload_size"] > 0


def test_invoke_edge_dynamic_picks_pubsub_for_small_payload(mock_pubsub_client, mock_token_fetcher, caplog):
    """
    With strategy='dynamic' and both endpoint + topic set, small payload should choose Pub/Sub
    and log mechanism='pubsub'.
    """
    caplog.set_level(logging.INFO, logger="chained_serverless_invoker.client")

    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)
    # Keep default cutoff (1MB) – small payload will be < cutoff → Pub/Sub

    edges = [
        EdgeConfig(
            to="B",
            strategy="dynamic",
            endpoint="https://example.com/next",
            topic="projects/x/topics/next",
            edge_id="A->B",
        )
    ]
    meta = InvokerMetadata(fn_name="A", run_id="run-2", taint="root", edges=edges)
    payload = {"x": 1}  # tiny payload

    with patch("chained_serverless_invoker.invokers.http_invoker.requests.post") as mock_post:
        mock_post.return_value.status_code = 200

        # Invoke in DYNAMIC mode; we expect Pub/Sub path
        result = invoker.invoke_edge(
            meta,
            target_fn="B",
            payload=payload,
            mode=InvocationMode.DYNAMIC,
        )

        # Pub/Sub invoker returns a Future-like object (from mock_pubsub_client)
        future = result
        future.result()

        mock_pubsub_client.publish.assert_called_once()
        mock_post.assert_not_called()

    send_records = [r for r in caplog.records if r.message == "invoker_edge_send"]
    assert send_records
    inv = send_records[0].invoker
    assert inv["run_id"] == "run-2"
    assert inv["mechanism"] == "pubsub"


class DummyRequest:
    def __init__(self, body: bytes):
        self._body = body

    def get_data(self) -> bytes:
        return self._body


def test_bootstrap_from_request_reconstructs_metadata_and_logs(caplog):
    caplog.set_level(logging.INFO, logger="chained_serverless_invoker.client")

    body = {
        "x": 42,
        DEFAULT_META_KEY: {
            "fn_name": "B",
            "run_id": "run-3",
            "taint": "t-123",
            "edges": [{"to": "C", "strategy": "dynamic", "endpoint": None, "topic": None, "edge_id": "B->C"}],
        },
    }

    req = DummyRequest(json.dumps(body).encode("utf-8"))

    meta, payload = bootstrap_from_request(req)

    # Metadata reconstructed
    assert meta is not None
    assert meta.fn_name == "B"
    assert meta.run_id == "run-3"
    assert meta.taint == "t-123"
    assert len(meta.edges) == 1
    assert meta.edges[0].to == "C"
    assert meta.edges[0].strategy == "dynamic"

    # Payload preserved
    assert payload["x"] == 42

    # Logging: invoker_edge_recv record
    recv_records = [r for r in caplog.records if r.message == "invoker_edge_recv"]
    assert recv_records
    inv = recv_records[0].invoker
    assert inv["run_id"] == "run-3"
    assert inv["taint"] == "t-123"
    assert inv["fn_name"] == "B"
    assert inv["payload_size"] > 0


def test_bootstrap_from_request_without_metadata_returns_none():
    body = {"x": 99}
    req = DummyRequest(json.dumps(body).encode("utf-8"))

    meta, payload = bootstrap_from_request(req)

    assert meta is None
    assert payload["x"] == 99


def test_bootstrap_from_pubsub_event_base64(caplog):
    caplog.set_level(logging.INFO, logger="chained_serverless_invoker.client")

    body = {
        "x": "ps",
        DEFAULT_META_KEY: {
            "fn_name": "B",
            "run_id": "run-ps",
            "taint": "t-ps",
            "edges": [],
        },
    }
    encoded = base64.b64encode(json.dumps(body).encode("utf-8")).decode("utf-8")
    event = {"data": encoded}

    meta, payload = bootstrap_from_request(event)

    assert meta is not None
    assert meta.fn_name == "B"
    assert payload["x"] == "ps"

    recv_records = [r for r in caplog.records if r.message == "invoker_edge_recv"]
    assert recv_records
