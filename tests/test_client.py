from unittest.mock import patch

import pytest

from chained_serverless_invoker import DynamicInvoker, InvocationMode

# Define some dummy data
SMALL_PAYLOAD = "small"
LARGE_PAYLOAD = "x" * (1024 * 1024 * 2)  # 2MB


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
