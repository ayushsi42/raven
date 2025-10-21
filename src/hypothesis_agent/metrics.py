"""Metrics utilities for the hypothesis agent."""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Iterator, Optional

from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter(
    "raven_http_requests_total",
    "Count of HTTP requests processed",
    labelnames=("method", "route", "status"),
)

REQUEST_LATENCY = Histogram(
    "raven_http_request_latency_seconds",
    "Latency of HTTP requests",
    labelnames=("method", "route"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

LLM_CALL_LATENCY = Histogram(
    "raven_llm_call_latency_seconds",
    "Latency observed for outbound LLM calls",
    labelnames=("operation", "model"),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

LLM_TOKEN_USAGE = Counter(
    "raven_llm_tokens_total",
    "Token usage reported by LLM providers",
    labelnames=("operation", "model", "token_type"),
)

STAGE_LATENCY = Histogram(
    "raven_workflow_stage_latency_seconds",
    "Latency observed per workflow stage",
    labelnames=("stage", "status"),
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

WORKFLOW_LATENCY = Histogram(
    "raven_workflow_total_latency_seconds",
    "Total runtime measured for entire workflows",
    labelnames=("outcome",),
    buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)


def record_request_metrics(method: str, route: str, status_code: int, latency: float) -> None:
    """Record HTTP request throughput and latency."""

    REQUEST_COUNTER.labels(method=method, route=route, status=str(status_code)).inc()
    REQUEST_LATENCY.labels(method=method, route=route).observe(latency)


@dataclass(slots=True)
class _StageAggregate:
    count: int = 0
    total_seconds: float = 0.0
    last_status: str = "pending"
    last_latency: float = 0.0


@dataclass(slots=True)
class _LlmAggregate:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    call_count: int = 0


@dataclass(slots=True)
class _WorkflowSummary:
    outcome: str
    duration_seconds: float


class LLMTelemetry:
    """Accumulates latency and token metrics across LLM calls and stages."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._llm_usage: Dict[str, _LlmAggregate] = {}
        self._stage_usage: Dict[str, Dict[str, _StageAggregate]] = {}
        self._workflow_start: Dict[str, float] = {}
        self._workflow_summary: Dict[str, _WorkflowSummary] = {}
        self._context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
            "raven_llmtelemetry_context",
            default={},
        )

    # ------------------------------------------------------------------
    # Context management helpers
    # ------------------------------------------------------------------
    @contextmanager
    def context(
        self,
        *,
        workflow_id: Optional[str] = None,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> Iterator[None]:
        """Attach metadata to downstream telemetry instrumentation."""

        current = dict(self._context.get() or {})
        if workflow_id is not None:
            current["workflow_id"] = workflow_id
        if stage is not None:
            current["stage"] = stage
        if operation is not None:
            current["operation"] = operation
        token = self._context.set(current)
        try:
            yield
        finally:
            self._context.reset(token)

    # ------------------------------------------------------------------
    # LLM instrumentation
    # ------------------------------------------------------------------
    def record_llm_call(
        self,
        *,
        operation: str,
        model: str,
        latency_seconds: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Record latency and token counts for a single LLM invocation."""

        ctx = self._context.get() or {}
        op_label = ctx.get("operation") or operation or "unknown"
        stage_label = ctx.get("stage")
        workflow_id = ctx.get("workflow_id")

        LLM_CALL_LATENCY.labels(operation=op_label, model=model or "unknown").observe(latency_seconds)

        def _inc(token_type: str, value: Optional[int]) -> None:
            if value is None:
                return
            LLM_TOKEN_USAGE.labels(operation=op_label, model=model or "unknown", token_type=token_type).inc(value)

        _inc("prompt", prompt_tokens)
        _inc("completion", completion_tokens)
        _inc("total", total_tokens)

        if workflow_id:
            with self._lock:
                agg = self._llm_usage.setdefault(workflow_id, _LlmAggregate())
                agg.call_count += 1
                agg.latency_seconds += latency_seconds
                if prompt_tokens is not None:
                    agg.prompt_tokens += prompt_tokens
                if completion_tokens is not None:
                    agg.completion_tokens += completion_tokens
                if total_tokens is not None:
                    agg.total_tokens += total_tokens
                if stage_label:
                    # ensure stage entry exists for lookups even if not yet timed
                    self._stage_usage.setdefault(workflow_id, {}).setdefault(stage_label, _StageAggregate())

    # ------------------------------------------------------------------
    # Stage instrumentation
    # ------------------------------------------------------------------
    def record_stage_duration(
        self,
        *,
        stage: str,
        duration_seconds: float,
        status: str,
        workflow_id: Optional[str] = None,
    ) -> None:
        """Record timing for a workflow stage."""

        STAGE_LATENCY.labels(stage=stage, status=status).observe(duration_seconds)
        if workflow_id is None:
            return
        with self._lock:
            stage_map = self._stage_usage.setdefault(workflow_id, {})
            agg = stage_map.setdefault(stage, _StageAggregate())
            agg.count += 1
            agg.total_seconds += duration_seconds
            agg.last_latency = duration_seconds
            agg.last_status = status

    # ------------------------------------------------------------------
    # Workflow lifecycle
    # ------------------------------------------------------------------
    def workflow_started(self, workflow_id: str) -> None:
        with self._lock:
            self._workflow_start[workflow_id] = time.perf_counter()

    def workflow_finished(self, workflow_id: str, outcome: str) -> None:
        with self._lock:
            start = self._workflow_start.pop(workflow_id, None)
            if start is None:
                # If we never saw a start just record zero duration.
                duration = 0.0
            else:
                duration = time.perf_counter() - start
            WORKFLOW_LATENCY.labels(outcome=outcome).observe(duration)
            self._workflow_summary[workflow_id] = _WorkflowSummary(outcome=outcome, duration_seconds=duration)

    # ------------------------------------------------------------------
    # Snapshot helpers for attaching telemetry to workflow metadata
    # ------------------------------------------------------------------
    def stage_summary(self, workflow_id: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            stage_map = self._stage_usage.get(workflow_id, {})
            return {
                stage: {
                    "invocations": agg.count,
                    "total_seconds": round(agg.total_seconds, 4),
                    "average_seconds": round(agg.total_seconds / agg.count, 4) if agg.count else 0.0,
                    "last_latency_seconds": round(agg.last_latency, 4),
                    "status": agg.last_status,
                }
                for stage, agg in stage_map.items()
            }

    def llm_summary(self, workflow_id: str) -> Dict[str, Any]:
        with self._lock:
            agg = self._llm_usage.get(workflow_id)
            if not agg:
                return {
                    "call_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "average_latency_seconds": 0.0,
                }
            avg_latency = agg.latency_seconds / agg.call_count if agg.call_count else 0.0
            return {
                "call_count": agg.call_count,
                "prompt_tokens": agg.prompt_tokens,
                "completion_tokens": agg.completion_tokens,
                "total_tokens": agg.total_tokens,
                "average_latency_seconds": round(avg_latency, 4),
            }

    def workflow_runtime(self, workflow_id: str) -> Optional[float]:
        with self._lock:
            if workflow_id in self._workflow_start:
                return time.perf_counter() - self._workflow_start[workflow_id]
            summary = self._workflow_summary.get(workflow_id)
            if summary:
                return summary.duration_seconds
            return None

    def workflow_outcome(self, workflow_id: str) -> Optional[str]:
        with self._lock:
            summary = self._workflow_summary.get(workflow_id)
            return summary.outcome if summary else None


# Global telemetry singleton reused across the application.
PIPELINE_TELEMETRY = LLMTelemetry()
