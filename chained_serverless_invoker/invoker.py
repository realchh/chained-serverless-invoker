from abc import ABC, abstractmethod
from typing import Any


class AbstractInvoker(ABC):
    """
    Abstract interface for a serverless function invoker.
    Guarantees a common .invoke() method.
    """

    @abstractmethod
    def invoke(self, target_identifier: str, payload: str, **kwargs: Any) -> None:
        """
        Invokes a target function, handling all communication logistics.

        Args:
            target_identifier: The target (e.g., topic name or function URL).
            payload: The JSON string payload to send.
            **kwargs: Additional parameters specific to the invoker (e.g., auth functions).
        """
        pass