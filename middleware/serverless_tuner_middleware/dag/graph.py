from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class EdgeDef:
    """
    Static description of a workflow edge.

    This corresponds to a logical hop in the DAG, independent of mechanism
    (HTTP / Pub/Sub) or runtime measurements.

    - from_fn: logical name of the upstream function (e.g., "A")
    - to_fn:   logical name of the downstream function (e.g., "B")
    - edge_id: optional stable ID for this edge across configs / runs
               (e.g., "A->B"; if omitted, (from_fn, to_fn) is used as key)
    """

    from_fn: str
    to_fn: str
    edge_id: Optional[str] = None

    @property
    def key(self) -> str:
        """Canonical key for this edge (used in weight maps, etc.)."""
        return self.edge_id or f"{self.from_fn}->{self.to_fn}"


class WorkflowDag:
    """
    Minimal DAG representation with helpers for:

    - listing nodes / edges
    - topological order
    - critical path computation (longest path) based on edge weights

    Assumes:
      * No cycles (must be a DAG)
      * Nodes are identified by their logical fn_name strings
    """

    def __init__(self, edges: Iterable[EdgeDef]):
        self._edges: List[EdgeDef] = list(edges)
        self._nodes: Set[str] = set()
        self._out_edges: Dict[str, List[EdgeDef]] = {}
        self._in_degree: Dict[str, int] = {}

        self._build()

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # Collect nodes
        for e in self._edges:
            self._nodes.add(e.from_fn)
            self._nodes.add(e.to_fn)

        # Initialize adjacency + in-degree
        for n in self._nodes:
            self._out_edges.setdefault(n, [])
            self._in_degree.setdefault(n, 0)

        for e in self._edges:
            self._out_edges[e.from_fn].append(e)
            self._in_degree[e.to_fn] += 1

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    @property
    def edges(self) -> List[EdgeDef]:
        return list(self._edges)

    def successors(self, node: str) -> List[str]:
        return [e.to_fn for e in self._out_edges.get(node, [])]

    def predecessors(self, node: str) -> List[str]:
        return [e.from_fn for e in self._edges if e.to_fn == node]

    def sources(self) -> List[str]:
        """Nodes with in-degree 0."""
        return [n for n, deg in self._in_degree.items() if deg == 0]

    def sinks(self) -> List[str]:
        """Nodes with no outgoing edges."""
        return [n for n in self._nodes if not self._out_edges.get(n)]

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_order(self) -> List[str]:
        """
        Kahn's algorithm.

        Raises:
            ValueError if the graph has a cycle.
        """
        in_deg = dict(self._in_degree)
        queue = [n for n, deg in in_deg.items() if deg == 0]
        order: List[str] = []

        while queue:
            n = queue.pop()
            order.append(n)
            for e in self._out_edges.get(n, []):
                m = e.to_fn
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        if len(order) != len(self._nodes):
            raise ValueError("Graph is not a DAG (cycle detected)")

        return order

    # ------------------------------------------------------------------
    # Critical path (longest path in DAG)
    # ------------------------------------------------------------------

    def critical_path(
        self,
        edge_weights: Dict[str, float],
        *,
        node_weights: Optional[Dict[str, float]] = None,
        source_nodes: Optional[Iterable[str]] = None,
        sink_nodes: Optional[Iterable[str]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Compute the critical path (longest path) based on edge + node weights.

        Args:
            edge_weights:
                Mapping from edge key -> non-negative transport weight
                (e.g., p50 transport latency in ms). Keys should match EdgeDef.key.
                Missing edges default to 0.0.

            node_weights:
                Mapping from node name -> non-negative runtime weight
                (e.g., p50 function runtime in ms). Missing nodes default to 0.0.

            source_nodes:
                Optional subset of nodes to treat as possible start points.
                Defaults to all sources().

            sink_nodes:
                Optional subset of nodes to treat as possible end points.
                Defaults to all sinks().

        Returns:
            (total_weight, path_nodes) where path_nodes is a list of node names
            from start to end (inclusive). If the DAG is empty, returns (0.0, []).
        """
        if not self._nodes:
            return 0.0, []

        order = self.topological_order()

        sources = set(source_nodes) if source_nodes else set(self.sources())
        sinks = set(sink_nodes) if sink_nodes else set(self.sinks())

        # Default node weights to 0 if not provided
        nw = node_weights or {}
        node_w = {n: float(nw.get(n, 0.0)) for n in self._nodes}

        # Initialize distances and predecessors
        dist: Dict[str, float] = {n: float("-inf") for n in self._nodes}
        pred: Dict[str, Optional[str]] = {n: None for n in self._nodes}

        # Sources pay their own runtime as starting cost
        for s in sources:
            dist[s] = node_w[s]

        # Relax edges in topological order
        for u in order:
            if dist[u] == float("-inf"):
                continue  # unreachable from chosen sources
            for e in self._out_edges.get(u, []):
                v = e.to_fn
                edge_cost = float(edge_weights.get(e.key, 0.0))
                candidate = dist[u] + edge_cost + node_w[v]
                if candidate > dist[v]:
                    dist[v] = candidate
                    pred[v] = u

        # Pick best sink
        best_sink = None
        best_dist = float("-inf")
        candidate_sinks = sinks or set(self._nodes)
        for n in candidate_sinks:
            if dist[n] > best_dist:
                best_dist = dist[n]
                best_sink = n

        if best_sink is None or best_dist == float("-inf"):
            return 0.0, []

        # Reconstruct node path
        path: List[str] = []
        cur = best_sink
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()

        return best_dist, path
