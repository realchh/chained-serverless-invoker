# Chained Serverless Invoker

<div align="center">

[//]: # ([![Build Status]&#40;https://github.com/ubc-cirrus-lab/caribou/actions/workflows/workflow.yaml/badge.svg&#41;]&#40;https://github.com/ubc-cirrus-lab/caribou/actions/workflows/workflow.yaml&#41; [![GitHub license]&#40;https://img.shields.io/badge/license-Apache%202-blue.svg&#41;]&#40;https://github.com/ubc-cirrus-lab/caribou/blob/main/LICENSE&#41;)

Chained Serverless Invoker is a lightweight client library for asynchronously invoking
chained serverless workflows on Google Cloud Platform over **HTTP** or **Pub/Sub**.

It can:
- dynamically choose between HTTP and Pub/Sub based on payload size, and
- optionally carry DAG metadata and structured logs so an external middleware
  can learn per-edge latency and rewrite the workflow configuration.

The repo also contains a standalone offline middleware (packaged separately) with a CLI (`csi-middleware`)
that parses the invoker logs, computes per-edge/node latency stats, and rewrites workflow configs based on the
critical path. Install both packages side by side to experiment locally.

</div>

## ⚡️ Quickstart
###  Installation


For now, install from source:

```bash
git clone https://github.com/realchh/chained-serverless-invoker.git
cd chained-serverless-invoker
pip install -e .
```

To experiment with the offline middleware + CLI as well:

```bash
pip install -e invoker -e middleware
# CLI entry point (when installed): csi-middleware --help
```

For local dev with Poetry:
```bash
cd invoker && poetry install --no-interaction --no-root
cd ../middleware && poetry install --no-interaction --no-root
# activate the Poetry venv or use `poetry run csi-middleware --help`
```

## End-to-End: Invoker + Middleware (Step by Step)

1) **Instrument your functions** with `DynamicInvoker` and `bootstrap_from_request` so send/recv logs include the `invoker` block (see example below).
2) **Run your workflow** and collect logs (e.g., export Cloud Logging to NDJSON). The middleware expects `invoker_edge_send` and `invoker_edge_recv` entries with the `invoker` payload.
3) **Summarize benchmarks (optional)**: use `python middleware/serverless_tuner_middleware/csv_summary.py --help` to see options for deriving percentile baselines and plots from the CSVs in `middleware/serverless_tuner_middleware/csv/`.
4) **Rewrite the config** using the CLI:
   ```bash
   csi-middleware \
     --logs /path/to/logs.ndjson \
     --config-in /path/to/workflow_config.json \
     --config-out /path/to/new_config.json
   ```
   This parses logs, computes per-edge/node latency stats, finds the current critical path, and chooses faster mechanisms on that path.
5) **Deploy** using the rewritten config (e.g., feed it back to your orchestrator).

Example config (`workflow_config.json`):

```json
{
  "workflow_id": "demo-workflow",
  "edges": [
    {
      "from_fn": "entry",
      "to_fn": "A",
      "strategy": "dynamic",
      "endpoint": "https://a-xyz.a.run.app",
      "topic": "projects/myproj/topics/a",
      "edge_id": "entry-A"
    },
    {
      "from_fn": "A",
      "to_fn": "B",
      "strategy": "dynamic",
      "endpoint": "https://b-xyz.a.run.app",
      "topic": "projects/myproj/topics/b",
      "edge_id": "A-B"
    },
    {
      "from_fn": "A",
      "to_fn": "C",
      "strategy": "dynamic",
      "endpoint": "https://c-xyz.a.run.app",
      "topic": "projects/myproj/topics/c",
      "edge_id": "A-C"
    }
  ]
}
```

You can also build and save a config programmatically:

```python
from middleware.serverless_tuner_middleware.config import WorkflowConfig, WorkflowEdge, dump_config

cfg = WorkflowConfig(
    workflow_id="demo-workflow",
    edges=[
        WorkflowEdge(from_fn="entry", to_fn="A", strategy="dynamic", endpoint="https://a-xyz.a.run.app", topic="projects/myproj/topics/a", edge_id="entry-A"),
        WorkflowEdge(from_fn="A", to_fn="B", strategy="dynamic", endpoint="https://b-xyz.a.run.app", topic="projects/myproj/topics/b", edge_id="A-B"),
        WorkflowEdge(from_fn="A", to_fn="C", strategy="dynamic", endpoint="https://c-xyz.a.run.app", topic="projects/myproj/topics/c", edge_id="A-C"),
    ],
)

dump_config(cfg, "workflow_config.json")
```


### Usage

We offer two types of invocation:
- Drop-in replacement for Pub/Sub and HTTP calls.

```python
from google.cloud import pubsub_v1
from chained_serverless_invoker.client import DynamicInvoker, InvocationMode

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
from chained_serverless_invoker.client import (
  DynamicInvoker,
  bootstrap_from_request,
)
from chained_serverless_invoker.invokers.types import InvokerMetadata, EdgeConfig

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
