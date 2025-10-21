"""Alpha Vantage tool catalog loader for planning stage."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(slots=True)
class ToolDefinition:
    """Structured metadata about an Alpha Vantage Composio tool."""

    slug: str
    description: str
    inputs: Dict[str, str]
    output: str


def _default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "vantage_tools.txt"


def _parse_catalog_lines(lines: Iterable[str]) -> List[ToolDefinition]:
    entries: List[ToolDefinition] = []
    current_slug: str | None = None
    description: str = ""
    inputs: Dict[str, str] = {}
    output_lines: List[str] = []
    mode: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("ALPHA_VANTAGE"):
            if current_slug is not None:
                entries.append(ToolDefinition(slug=current_slug, description=description, inputs=inputs, output=" ".join(output_lines).strip()))
            current_slug = line
            description = ""
            inputs = {}
            output_lines = []
            mode = None
            continue
        if line.startswith("Description:"):
            description = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Input Format:"):
            mode = "input"
            continue
        if line.startswith("Output Format:"):
            mode = "output"
            continue

        if mode == "input":
            if ":" in line:
                key, value = line.split(":", 1)
                inputs[key.strip()] = value.strip()
        elif mode == "output":
            output_lines.append(line)

    if current_slug is not None:
        entries.append(ToolDefinition(slug=current_slug, description=description, inputs=inputs, output=" ".join(output_lines).strip()))
    return entries


@lru_cache(maxsize=None)
def load_tool_catalog(path: str | Path | None = None) -> Dict[str, ToolDefinition]:
    catalog_path = Path(path) if path is not None else _default_catalog_path()
    text = catalog_path.read_text(encoding="utf-8")
    definitions = _parse_catalog_lines(text.splitlines())
    return {definition.slug: definition for definition in definitions}


def get_tool_definition(slug: str) -> ToolDefinition:
    catalog = load_tool_catalog()
    if slug not in catalog:
        raise KeyError(f"Tool '{slug}' not found in Alpha Vantage catalog")
    return catalog[slug]
