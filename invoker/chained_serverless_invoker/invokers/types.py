from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EdgeConfig:
    from_fn: Optional[str] = None  # optional source node for disambiguating parallel edges
    to: str = ""
    strategy: str = "dynamic"  # "http" | "pubsub" | "dynamic"
    endpoint: Optional[str] = None
    topic: Optional[str] = None
    edge_id: Optional[str] = None  # optional static DAG edge id


@dataclass
class InvokerMetadata:
    fn_name: str  # current logical node
    run_id: str  # workflow/run id
    taint: str  # taint for the incoming edge instance
    edges: List[EdgeConfig]  # DAG / per-edge config
