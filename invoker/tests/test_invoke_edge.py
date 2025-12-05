import json
import logging
from unittest.mock import patch

import pytest

from invoker.chained_serverless_invoker import DynamicInvoker, InvocationMode
from invoker.chained_serverless_invoker.client import bootstrap_from_request
from invoker.chained_serverless_invoker.constants import DEFAULT_META_KEY
from invoker.chained_serverless_invoker.invokers.types import EdgeConfig, InvokerMetadata

pytest_plugins = ["tests.conftest"]


def test_edge_strategy_overrides_explicit_mode_pubsub(mock_pubsub_client, mock_token_fetcher, caplog):
    caplog.set_level(logging.INFO, logger="chained_serverless_invoker.client")
    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    edges = [
        EdgeConfig(
            to="worker",
            strategy="pubsub",
            endpoint="https://should.not/be-used",
            topic="projects/demo/topics/worker",
            edge_id="A->worker",
        )
    ]
    meta = InvokerMetadata(fn_name="A", run_id="run-edge", taint="t-root", edges=edges)
    payload = {"hello": "world"}

    with patch("chained_serverless_invoker.invokers.http_invoker.requests.post") as mock_post:
        mock_post.return_value.status_code = 200

        fut = invoker.invoke_edge(
            meta,
            target_fn="worker",
            payload=payload,
            mode=InvocationMode.FORCE_HTTP,  # should be ignored in favor of edge.strategy
        )
        fut.result()

    mock_pubsub_client.publish.assert_called_once()
    mock_post.assert_not_called()

    send_logs = [r for r in caplog.records if r.message == "invoker_edge_send"]
    assert send_logs
    inv = send_logs[0].invoker
    assert inv["edge_id"] == "A->worker"
    assert inv["mechanism"] == "pubsub"
    assert inv["from_fn"] == "A"
    assert inv["to_fn"] == "worker"


class DummyRequest:
    def __init__(self, body: bytes):
        self._body = body

    def get_data(self) -> bytes:
        return self._body


def test_metadata_injected_and_bootstrap_preserves_fields(mock_pubsub_client, mock_token_fetcher, caplog):
    caplog.set_level(logging.INFO, logger="chained_serverless_invoker.client")
    invoker = DynamicInvoker(mock_pubsub_client, mock_token_fetcher)

    edges = [EdgeConfig(to="B", strategy="http", endpoint="https://next", topic=None, edge_id="A->B")]
    meta = InvokerMetadata(fn_name="A", run_id="run-meta", taint="root", edges=edges)
    payload = {"value": 1}

    with patch("chained_serverless_invoker.invokers.http_invoker.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        fut = invoker.invoke_edge(meta, target_fn="B", payload=payload)
        fut.result()

    assert DEFAULT_META_KEY in payload
    meta_block = payload[DEFAULT_META_KEY]
    assert meta_block["run_id"] == "run-meta"
    assert meta_block["fn_name"] == "B"
    assert meta_block["edges"]
    assert meta_block["taint"]

    send_logs = [r for r in caplog.records if r.message == "invoker_edge_send"]
    assert send_logs

    req_body = json.dumps(payload).encode("utf-8")
    req = DummyRequest(req_body)
    recv_meta, recv_payload = bootstrap_from_request(req)

    assert recv_meta is not None
    assert recv_meta.fn_name == "B"
    assert recv_meta.run_id == "run-meta"
    assert recv_payload["value"] == 1

    recv_logs = [r for r in caplog.records if r.message == "invoker_edge_recv"]
    assert recv_logs
    recv_inv = recv_logs[0].invoker
    assert recv_inv["run_id"] == "run-meta"
    assert recv_inv["fn_name"] == "B"
    assert recv_inv["payload_size"] > 0
