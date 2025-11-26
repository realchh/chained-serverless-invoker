# Chained Serverless Invoker

<div align="center">

[//]: # ([![Build Status]&#40;https://github.com/ubc-cirrus-lab/caribou/actions/workflows/workflow.yaml/badge.svg&#41;]&#40;https://github.com/ubc-cirrus-lab/caribou/actions/workflows/workflow.yaml&#41; [![GitHub license]&#40;https://img.shields.io/badge/license-Apache%202-blue.svg&#41;]&#40;https://github.com/ubc-cirrus-lab/caribou/blob/main/LICENSE&#41;)

Chained Serverless Invoker is a lightweight client library for asynchronously invoking
chained serverless workflows on Google Cloud Platform over **HTTP** or **Pub/Sub**.

It can:
- dynamically choose between HTTP and Pub/Sub based on payload size, and
- optionally carry DAG metadata and structured logs so an external middleware
  can learn per-edge latency and rewrite the workflow configuration.

</div>

## ⚡️ Quickstart
###  Installation


For now, install from source:

```bash
git clone https://github.com/realchh/chained-serverless-invoker.git
cd chained-serverless-invoker
pip install -e .
```

### Usage

We offer two types of invocation:
- Drop-in replacement for Pub/Sub and HTTP calls.

```python
from google.cloud import pubsub_v1
from invoker.chained_serverless_invoker.client import DynamicInvoker, InvocationMode

publisher = pubsub_v1.PublisherClient()


def fetch_token(audience: str) -> str:
  # e.g., use google-auth to fetch an ID token for Cloud Run
  ...


invoker = DynamicInvoker(pubsub_client=publisher, token_fetcher=fetch_token)

payload = '{"hello": "world"}'

# Let the invoker choose based on payload size:
invoker.invoke(
  payload,
  http_url="https://my-service-xyz.a.run.app",
  pubsub_topic="projects/myproj/topics/my-topic",
  mode=InvocationMode.DYNAMIC,
)
```
In this mode, you don’t need any DAG metadata. The invoker will pick HTTP vs. Pub/Sub with the locally optimum values.

- Advanced invocation with DAG metadata and structured logs for use with the middleware.

If you want to build a middleware that learns per-edge latency and rewrites configs,
you can use the DAG-aware helpers:
- `bootstrap_from_request(request)` on the receiver side,
- `DynamicInvoker.invoke_edge(...)` on the sender side.

These:
- inject a reserved `__invoker` key into the JSON payload, carrying:
    - logical function name,
    - workflow run ID,
    - a per-edge “taint” (unique message ID),
    - the static DAG/edge configuration;

- Emit structured logs on send and receive, so an offline analysis
can reconstruct per-edge transport latency and the critical path.

Example Cloud Run-style handler:

```python
from invoker.chained_serverless_invoker.client import (
  DynamicInvoker,
  bootstrap_from_request,
)
from invoker.chained_serverless_invoker.invokers.types import InvokerMetadata, EdgeConfig

publisher = ...
invoker = DynamicInvoker(pubsub_client=publisher, token_fetcher=fetch_token)

# Static edges for this logical node (in practice, generated from a config file)
MY_EDGES = [
  EdgeConfig(
    to="my-workflow:step-b",
    strategy="dynamic",
    endpoint="https://step-b-xyz.a.run.app",
    topic="projects/myproj/topics/step-b",
    edge_id="step-a->step-b",
  )
]


def handle(request):
  # Try to bootstrap metadata from an incoming request
  meta, payload = bootstrap_from_request(request)

  if meta is None:
    # First hop: application/gateway is responsible for assigning run_id
    run_id = "some-run-id"  # e.g. uuid4().hex at the gateway
    meta = InvokerMetadata(
      fn_name="my-workflow:step-a",
      run_id=run_id,
      taint="root",
      edges=MY_EDGES,
    )
    payload = {"foo": "bar"}

  # Do your normal business logic here, mutating `payload` as needed
  payload["result"] = "ok"

  # Chain to the next node in the DAG
  invoker.invoke_edge(
    meta,
    target_fn="my-workflow:step-b",
    payload=payload,
  )

  return ("OK", 200)
```

The logs produced by this flow are designed to be consumed by a separate middleware
(e.g., a log processor that learns a latency model and rewrites the DAG config).
For more details, see [Architecture](ARCHITECTURE.md).

## Testing

To run the tests, navigate to the project root, and run:

```bash
./scripts/compliance.sh
```

## Paper.

## About

Chained Serverless Invoker is being developed at the [Cloud Infrastructure Research for Reliability, Usability, and Sustainability Lab](https://cirrus.ece.ubc.ca) at the [University of British Columbia](https://www.ubc.ca). If you have any questions or feedback, please open a GitHub issue.

##  Contributing

Contributions are welcome! Please open an issue or pull request to discuss changes.

## License

Apache License 2.0
