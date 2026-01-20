"""Workflow definitions retained for backward compatibility."""
from __future__ import annotations


class HypothesisValidationWorkflow:  # pragma: no cover - legacy compatibility shim
    """Placeholder to maintain import compatibility. Use HypothesisWorkflowClient instead."""

    def __init__(self) -> None:  # noqa: D401 - docstring not needed for stub
        raise RuntimeError(
            "Direct workflow instantiation is not supported. Use the LangGraph pipeline via HypothesisWorkflowClient instead."
        )
