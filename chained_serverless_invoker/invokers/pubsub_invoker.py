from google.cloud import pubsub_v1

from chained_serverless_invoker.invokers.invoker import AbstractInvoker


class PubSubInvoker(AbstractInvoker):
    """
    Implements reliable Pub/Sub invocation, blocking until the service ACK.
    """

    def __init__(self, publisher_client: pubsub_v1.PublisherClient):
        self.client = publisher_client

    def invoke(self, target_identifier: str, payload: str, **kwargs) -> None:
        """
        Publishes the message and blocks until the future resolves (reliable ACK).

        Args:
            target_identifier: The target (full topic path).
            payload: The string payload to send.
        """
        # Assumes target_identifier is the full topic path
        topic_path = target_identifier
        data = payload.encode("utf-8")

        # client.publish returns a future.
        return self.client.publish(topic=topic_path, data=data)