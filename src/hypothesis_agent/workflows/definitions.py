"""Temporal workflow scaffolding retained solely for backward compatibility."""
from __future__ import annotations


class HypothesisValidationWorkflow:  # pragma: no cover - legacy compatibility shim
    """Placeholder to maintain import compatibility after removing Temporal."""

    def __init__(self) -> None:  # noqa: D401 - docstring not needed for stub
        raise RuntimeError(
            "Temporal workflows have been removed. Use the LangGraph pipeline via HypothesisWorkflowClient instead."
        )
