from typing import Any

from google.cloud import pubsub_v1

from invoker.chained_serverless_invoker.invokers.abstract_invoker import AbstractInvoker


class PubSubInvoker(AbstractInvoker):
    """
    Implements reliable Pub/Sub invocation, blocking until the service ACK.
    """

    def __init__(self, publisher_client: pubsub_v1.PublisherClient):
        self.client = publisher_client

    def invoke(self, target: str, payload: str, **kwargs: Any) -> Any:
        """
        Publishes the message and blocks until the future resolves (reliable ACK).

        Args:
            target: The target (full topic path).
            payload: The string payload to send.
            **kwargs: Additional parameters specific to the invoker (e.g., auth functions).

        Returns:
            A Future-like object (google.api_core.future.Future)
        """
        # Assumes target_identifier is the full topic path
        topic_path = target
        data = payload.encode("utf-8")

        # client.publish returns a future. use .result() to block.
        return self.client.publish(topic=topic_path, data=data)
