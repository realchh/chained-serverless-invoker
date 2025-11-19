from abc import ABC, abstractmethod
from typing import Any


class AbstractInvoker(ABC):
    """
    Abstract interface for a serverless function invoker.
    Guarantees a common .invoke() method.
    """

    @abstractmethod
    def invoke(self, target: str, payload: str, **kwargs: Any) -> Any:
        """
        Invokes a target function asynchronously.

        Args:
            payload: The string payload to send.
            target: The target (e.g., topic name for Pub/Sub invocations or function URL for HTTP invocations).
            **kwargs: Additional parameters specific to the invoker (e.g., auth functions).

        Returns:
            A Future-like object (concurrent.futures.Future or google.api_core.future.Future)
            that allows the caller to wait for the result or check for exceptions.
        """
        pass