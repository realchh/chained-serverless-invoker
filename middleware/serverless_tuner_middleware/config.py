from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, List


@dataclass(frozen=True)
class WorkflowEdge:
    from_fn: str
    to_fn: str
    strategy: str  # "http" | "pubsub" | "dynamic"
    endpoint: str | None = None
    topic: str | None = None
    edge_id: str | None = None

    @property
    def key(self) -> str:
        return self.edge_id or f"{self.from_fn}->{self.to_fn}"


@dataclass(frozen=True)
class WorkflowConfig:
    workflow_id: str
    edges: List[WorkflowEdge]

    @staticmethod
    def from_dict(data: dict) -> "WorkflowConfig":
        edges = [WorkflowEdge(**e) for e in data.get("edges", [])]
        return WorkflowConfig(workflow_id=data.get("workflow_id", ""), edges=edges)

    def to_dict(self) -> dict:
        return {"workflow_id": self.workflow_id, "edges": [asdict(e) for e in self.edges]}


def load_config(path_or_file: str | Path | IO[str]) -> WorkflowConfig:
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
    else:
        with open(Path(path_or_file), "r", encoding="utf-8") as f:
            raw = f.read()

    data = json.loads(raw)
    return WorkflowConfig.from_dict(data)


def dump_config(config: WorkflowConfig, path_or_file: str | Path | IO[str]) -> None:
    serialized = json.dumps(config.to_dict(), indent=2)

    if hasattr(path_or_file, "write"):
        path_or_file.write(serialized)
    else:
        Path(path_or_file).write_text(serialized, encoding="utf-8")
