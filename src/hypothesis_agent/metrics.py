"""Prometheus metrics integration."""
from __future__ import annotations

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

def record_request_metrics(method: str, route: str, status_code: int, latency: float) -> None:
    REQUEST_COUNTER.labels(method=method, route=route, status=str(status_code)).inc()
    REQUEST_LATENCY.labels(method=method, route=route).observe(latency)
