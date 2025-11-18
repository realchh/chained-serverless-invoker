import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Optional

import requests

from .invoker import AbstractInvoker
from .constants import DEFAULT_HTTP_FUTURE_TIMEOUT_SEC, DEFAULT_HTTP_MAX_WORKERS


# Shared thread pool for all HTTP invocations (Class-level singleton pattern)
class HTTPExecutorManager:
    _executor: Optional[ThreadPoolExecutor] = None
    _lock = threading.Lock()

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        if cls._executor is None:
            with cls._lock:
                if cls._executor is None:
                    cls._executor = ThreadPoolExecutor(
                        max_workers=DEFAULT_HTTP_MAX_WORKERS,
                        thread_name_prefix="Caribou-HTTP-Worker"
                    )
        return cls._executor


class HttpInvoker(AbstractInvoker):
    """
    Implements fire-and-forget asynchronous HTTP invocation using a ThreadPool.
    """

    def __init__(self, token_fetcher: Callable[[str], str]):
        self.executor = HTTPExecutorManager.get_executor()
        self.token_fetcher = token_fetcher

    def _make_http_request(self, service_url: str, json_payload: str) -> None:
        """Worker function executed in the background thread."""

        # NOTE: Timeout is set high (300s) to allow for cold starts and large transfers
        # The calling thread only waits 5ms; the worker thread handles the full duration.
        try:
            # 1. Get Authentication Token (simulating _get_id_token logic)
            id_token_value = self.token_fetcher(service_url)

            headers = {
                "Authorization": f"Bearer {id_token_value}",
                "Content-Type": "application/json",
            }

            # 2. Execute the Blocking Request
            response = requests.post(
                service_url,
                headers=headers,
                data=json_payload,
                timeout=300
            )

            response.raise_for_status()
            # print(f"HTTP worker success: {service_url} -> {response.status_code}")

        except requests.exceptions.RequestException as e:
            # IMPORTANT: The main thread will never see this error.
            # This is the 'at-most-once' failure mode we are quantifying.
            print(f"ERROR: HTTP worker request failed for {service_url}: {e}")
        except Exception as e:
            print(f"ERROR: Failed to authorize or prepare request for {service_url}: {e}")

    def invoke(self, target_identifier: str, payload: str, **kwargs: Any) -> None:
        """Submits the HTTP request to the thread pool (fire-and-forget)."""

        # Assumes target_identifier is the full service URL for HTTP calls
        service_url = target_identifier

        future = self.executor.submit(self._make_http_request, service_url, payload)

        # Wait very briefly to ensure the task is picked up from the queue.
        # This is the 5ms block that allows the calling function to return quickly.
        try:
            future.result(timeout=DEFAULT_HTTP_FUTURE_TIMEOUT_SEC)
        except Exception:
            # Expected behavior: the call timed out, meaning it's now running asynchronously.
            pass