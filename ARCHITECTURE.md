
# Architecture

This document describes the internal architecture of the `chained-serverless-invoker`
library and how it interacts with a higher-level middleware.

The library is intentionally small: it focuses on **runtime invocation and logging**.
Any heavy lifting (DAG analysis, latency modeling, configuration rewrites) is done
by an external middleware or research prototype.

---

## Goals

- Provide a **drop-in client** that can invoke GCP serverless functions over HTTP or Pub/Sub.
- Support **dynamic selection** of HTTP vs. Pub/Sub, initially based on payload size.
- Optionally, carry **DAG metadata** and emit **structured logs** so an external system can:
  - reconstruct workflow runs,
  - measure per-edge transport latency,
  - find critical paths,
  - and rewrite a configuration file accordingly.

The library itself remains **stateless** and **provider-native**.

---

## Components

### 1. `DynamicInvoker`

Located in `chained_serverless_invoker/client.py`.

- Wraps two concrete invokers:
  - `HttpInvoker` – async HTTP fire-and-forget using a shared `ThreadPoolExecutor`.
  - `PubSubInvoker` – blocking publish to a Google Cloud Pub/Sub topic.
- Exposes two main methods:
  - `invoke(payload: str, ...)` – simple size-based dynamic selection.
  - `invoke_edge(meta: InvokerMetadata, target_fn: str, payload: dict, ...)` –
    DAG-aware invocation that injects metadata and emits structured logs.

### 2. `HttpInvoker` and `PubSubInvoker`

Located in `chained_serverless_invoker/invokers/`.

- `HttpInvoker`:
  - Uses `requests.post` with an ID token fetched by a user-provided `token_fetcher`.
  - Returns a `Future` from a shared `ThreadPoolExecutor`.

- `PubSubInvoker`:
  - Wraps `google.cloud.pubsub_v1.PublisherClient.publish`.
  - Returns the Pub/Sub publish future.

### 3. `InvokerMetadata` and `EdgeConfig`

Defined in `chained_serverless_invoker/invokers/types.py`.

- `EdgeConfig` describes a directed edge in the workflow DAG:

  ```python
  @dataclass
  class EdgeConfig:
      to: str
      strategy: str                  # "http" | "pubsub" | "dynamic"
      endpoint: Optional[str] = None
      topic: Optional[str] = None
      edge_id: Optional[str] = None  # optional static DAG edge id
  ```

- `InvokerMetadata` carries per-invocation context:

  ```python
  @dataclass
  class InvokerMetadata:
      function_name: str             # logical function name
      run_id: str                    # workflow run id
      edges: Dict[str, EdgeConfig]   # static DAG edges from this function
  ```
  
`run_id` and `taint` together uniquely identify a specific edge instance within a run.

## Payload Metadata
DAG-aware mode assumes the payload is a JSON object and reserves a single key:

`DEFAULT_META_KEY = "__invoker"`

At each hop, invoke_edge injects:
```json
"__invoker": {
  "fn_name": "B",          // logical name of the *next* function (receiver)
  "run_id": "abc123",      // workflow run id
  "taint": "123123",       // unique per edge instance (message)
  "edges": [               // static DAG / edge configs
    { "to": "C", "strategy": "dynamic", "endpoint": "...", "topic": "...", "edge_id": "B->C" },
    ...
  ]
}
```

The receiver calls `bootstrap_from_request(request)` which:
- Parses the body as JSON.
- Looks up `payload["__invoker"]`.
- Reconstructs an `InvokerMetadata` object for the current node.
- Logs a `invoker_edge_recv` event.
- Returns `(meta, payload)` to the handler.

## Logging Schema

The library emits two structured log events:

### Send-side: `invoker_edge_send`
Emitted by `DynamicInvoker.invoke_edge` before sending the request.

Fields (via `extra={"invoker": {...}}`):
```json
{
  "run_id": "abc123",
  "taint": "123123",
  "from_fn": "A",
  "to_fn": "B",
  "edge_id": "A->B",
  "mechanism": "http",       // "http" | "pubsub"
  "ts_ms": 1700000000000,    // time of send in milliseconds
  "payload_size": 512        // bytes
}
```

### Receive-side: `invoker_edge_recv`
Emitted by `bootstrap_from_request` upon receiving a request.

Fields:
```json{
  "run_id": "abc123",
  "taint": "123123",
  "fn_name": "B",
  "ts_ms": 1700000000080,
  "payload_size": 512
}
```
An external middleware can join these two logs on `(run_id, taint)` to derive:
- `from_fn` and `to_fn`,
- mechanism,
- `transport_ms = recv.ts_ms - send.ts_ms`,
- per-edge latency distributions,
- the critical path for each run.

## Control Flow (DAG-aware mode)

### Ingress / First Hop

A gateway or first function creates an initial `InvokerMetadata` with a new `run_id`,
and a static edges list for the workflow.

### Receiver

- A function is triggered by HTTP or Pub/Sub.
- It calls `meta, payload = bootstrap_from_request(request)`.
- If `meta` is None, the request did not carry DAG metadata and can be treated
as a normal non-DAG invocation.
- If `meta` is present, the function is part of an instrumented workflow.

### Business Logic

The handler reads from payload, performs its normal work, and mutates payload
with any intermediate results.

### Next Hop

When ready to invoke the next node, the handler calls:
- `invoker.invoke_edge(meta, target_fn="next-step", payload=payload)`

`invoke_edge`:
- looks up the `EdgeConfig` for `target_fn`,
- decides `InvocationMode` (`FORCE_HTTP` / `FORCE_PUBSUB` / `DYNAMIC`),
- attaches a new 1 and updated metadata into `payload["__invoker"]`,
- logs `invoker_edge_send`,
- delegates to `DynamicInvoker.invoke(...)` which actually performs
the HTTP or Pub/Sub invocation.

### Middleware

Separately, a middleware (a log processor) collects send/recv logs
from Cloud Logging / Stackdriver.

- It groups them by `run_id`, joins `send/recv` on `(run_id, taint)`,
and builds a per-edge latency model.
- Based on this model, it can rewrite a config file to:
  - prefer HTTP on some edges,
  - prefer Pub/Sub on others, 
  - or change workflow structure in a future iteration.

### Middleware rewrite heuristics

The default rewrite mode (`critical-path`) uses greedy flips on the current critical path.
Heuristics applied:

- **Gain floor:** only flip an edge if the p50 gain exceeds `GAIN_THRESHOLD_MS`.
- **Path selection:** pick the longest end-to-end path (edge transports using each edge’s chosen mechanism, plus node runtimes); verbose mode logs all paths ranked by this cost.
- **Flip scope and cap:** only flip edges on the chosen path; stop when no edge clears the gain floor or when `MAX_EDGE_FLIPS_PER_RUN` (default 10) is reached.
- **Sync bottleneck reporting:** per-run bottleneck frequency on sync edges is computed with a single threshold `SYNC_BOTTLENECK_RUN_SHARE_THRESHOLD` (default 20%).
- **Fallback:** dynamic strategies normalize to HTTP after optimization.

Constants live in `middleware/serverless_tuner_middleware/constants.py`.

### Regression latency model

Unified form per quantile `q`:

lat_q = c_floor_q + a_q / (k_q + rate_rps) + d_rate_q * rate_rps + b_size_q * payload_bytes

- `q` is the target quantile; default when unspecified is median (p50).
- `c_floor_q` (>= 0): irreducible floor — half-RTT propagation, fixed proxy/server work, per-hop overhead.
- `a_q` (>= 0) and `k_q` (>= 0): control the 1/(k+rate) term that captures queueing/throughput effects (Pub/Sub benefits more here).
- `d_rate_q` (signed): linear rate slope; HTTP can slope slightly up, Pub/Sub can benefit from the reciprocal term while also allowing a linear tweak.
- `b_size_q` (>= 0): payload-size slope in bytes.
- Inputs: `rate_rps` is observed sends per second for the edge; `payload_bytes` is mean payload size for the edge.

## Non-goals / Limitations

The library does **not**:
- store or manage the global DAG config itself,
- implement the offline middleware or optimizer,
- manage carbon or cost objectives directly.

Those responsibilities are left to higher-level systems that consume the logs and
the config files this library helps target.
- JSON is currently assumed for DAG-aware mode. If a function cannot accept JSON,
it may still use the basic DynamicInvoker.invoke(...) API without metadata.