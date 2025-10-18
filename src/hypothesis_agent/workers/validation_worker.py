"""Deprecated entrypoint retained for backward compatibility."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Raise a runtime error indicating Temporal workers are no longer supported."""

    raise RuntimeError(
        "Temporal worker execution has been removed. The LangGraph pipeline now runs directly via the API."
    )


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()
