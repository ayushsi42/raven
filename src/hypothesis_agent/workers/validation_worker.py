"""CLI entrypoint for running the hypothesis validation Temporal worker."""
from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker

from hypothesis_agent.config import get_settings
from hypothesis_agent.workflows.activities.validation import (
    perform_validation,
    run_analysis,
    run_data_ingestion,
    run_modeling,
    run_preprocessing,
    run_sentiment,
)
from hypothesis_agent.workflows.definitions import HypothesisValidationWorkflow

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    settings = get_settings()
    client = await TemporalClient.connect(
        settings.temporal_address,
        namespace=settings.temporal_namespace,
    )

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[HypothesisValidationWorkflow],
        activities=[
            run_data_ingestion,
            run_preprocessing,
            run_analysis,
            run_sentiment,
            run_modeling,
            perform_validation,
        ],
    )

    logger.info(
        "Starting Temporal worker for task_queue=%s namespace=%s",
        settings.temporal_task_queue,
        settings.temporal_namespace,
    )
    await worker.run()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()
