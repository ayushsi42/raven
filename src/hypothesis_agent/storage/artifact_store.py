"""File system backed artifact persistence."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ArtifactStore:
    """Persist raw datasets and summaries to the local filesystem."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_path(cls, path: str) -> "ArtifactStore":
        return cls(root=Path(path).resolve())

    def write_json(self, workflow_id: str, name: str, payload: Dict[str, Any]) -> Path:
        target = self._workflow_dir(workflow_id) / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("Persisted JSON artifact %s", target)
        return target

    def write_bytes(self, workflow_id: str, name: str, content: bytes) -> Path:
        target = self._workflow_dir(workflow_id) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        logger.debug("Persisted binary artifact %s", target)
        return target

    def write_text(self, workflow_id: str, name: str, content: str, *, encoding: str = "utf-8") -> Path:
        target = self._workflow_dir(workflow_id) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        logger.debug("Persisted text artifact %s", target)
        return target

    def append_text(self, workflow_id: str, name: str, content: str, *, encoding: str = "utf-8") -> Path:
        target = self._workflow_dir(workflow_id) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding=encoding) as handle:
            handle.write(content)
        logger.debug("Appended text artifact %s", target)
        return target

    def _workflow_dir(self, workflow_id: str) -> Path:
        safe_id = workflow_id.replace("/", "_")
        return self.root / safe_id
