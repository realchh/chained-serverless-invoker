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
        """
        # Assumes target_identifier is the full topic path
        topic_path = target_identifier

        # Pub/Sub requires data to be bytes
        data = payload.encode("utf-8")

        # client.publish returns a future.
        future = self.client.publish(topic=topic_path, data=data)

        try:
            message_id = future.result()
            print(f"Pub/Sub success: Message ID: {message_id}")
        except Exception as e:
            raise RuntimeError(f"Pub/Sub publish failed for topic {topic_path}: {e}") from e