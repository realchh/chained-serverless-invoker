import threading
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Optional

from chained_serverless_invoker.invokers.invoker import AbstractInvoker
from chained_serverless_invoker.constants import DEFAULT_HTTP_MAX_WORKERS, DEFAULT_HTTP_REQUEST_TIMEOUT_SEC

# Shared thread pool for all HTTP invocations
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
                        thread_name_prefix="CSI-HTTP-Worker"
                    )
        return cls._executor


class HttpInvoker(AbstractInvoker):
    """
    Implements fire-and-forget asynchronous HTTP invocation using a ThreadPool.
    """

    def __init__(self, token_fetcher: Callable[[str], str]):
        self.executor = HTTPExecutorManager.get_executor()
        self.token_fetcher = token_fetcher

    def _make_http_request(self, service_url: str, json_payload: str, auth_token: Optional[str] = None) -> dict:
        """
        The actual blocking work. Returns a dict on success, raises Exception on failure.
        """
        # 1. Token Resolution
        if auth_token:
            id_token_value = auth_token
        else:
            # This might block/slow down if not cached, but it happens in the worker thread now!
            id_token_value = self.token_fetcher(service_url)

        headers = {
            "Authorization": f"Bearer {id_token_value}",
            "Content-Type": "application/json",
        }

        # 2. The Request
        response = requests.post(
            service_url,
            headers=headers,
            data=json_payload,
            timeout=DEFAULT_HTTP_REQUEST_TIMEOUT_SEC,
        )

        response.raise_for_status()

        # 3. Return something useful to the Future
        return {
            "status": response.status_code,
            "url": service_url,
            "mode": "http"
        }

    def invoke(self, target_identifier: str, payload: str, **kwargs: Any) -> Future:
        """
        Submits the task and immediately returns the Future.
        """
        service_url = target_identifier
        auth_token = kwargs.get('auth_token')

        # Return the future immediately!
        # The caller can use .result() to block if they want reliability,
        # or just ignore it for fire-and-forget.
        return self.executor.submit(self._make_http_request, service_url, payload, auth_token)