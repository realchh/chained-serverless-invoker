from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_pubsub_client():
    client = MagicMock()
    # Mock the publish method to return a Future-like object
    future = MagicMock()
    future.result.return_value = "msg_id_123"
    client.publish.return_value = future
    return client


@pytest.fixture
def mock_token_fetcher():
    return MagicMock(return_value="fake_token")
