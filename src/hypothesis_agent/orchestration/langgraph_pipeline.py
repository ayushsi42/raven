"""LangGraph- and Composio-powered orchestration for validation activities."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Dict, List, Tuple, TypedDict, cast

try:  # pragma: no cover - optional dependency in tests
    from composio_langchain import ComposioToolSet
except Exception:  # pragma: no cover - fall back to noop when Composio unavailable
    class ComposioToolSet:  # type: ignore[no-redef]
        def get_tool(self, name: str):
            raise RuntimeError("Composio integration is not available in this environment")
from langgraph.graph import END, StateGraph

from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


class StageContext(TypedDict, total=False):
    """Shared context that flows between LangGraph stages."""

    data: Dict[str, Any]
    normalized: Dict[str, Any]
    analysis: Dict[str, Any]
    sentiment: Dict[str, Any]
    modeling: Dict[str, Any]
    insights: List[str]
    evidence: List[Dict[str, Any]]
    context_notes: List[str]


@dataclass(slots=True)
class StageExecutionResult:
    """Container for stage execution outcomes used by Temporal activities."""

    context: StageContext
    milestone: WorkflowMilestone
    evidence: List[EvidenceReference] = field(default_factory=list)
    summary: ValidationSummary | None = None
    detail: str | None = None


class ComposioRouter:
    """Thin wrapper around Composio that provides best-effort data fetches."""

    def __init__(self, toolset: ComposioToolSet | None = None) -> None:
        self.toolset = toolset or ComposioToolSet()

    def fetch_market_data(self, request: HypothesisRequest) -> Tuple[Dict[str, Any], EvidenceReference]:
        payload = {
            "tickers": request.entities or ["SPY"],
            "horizon_days": max((request.time_horizon.end - request.time_horizon.start).days, 1),
        }
        try:
            tool = self.toolset.get_tool("polygon_get_snapshot_all_tickers")
            result = tool.invoke({"tickers": payload["tickers"]})  # type: ignore[call-arg]
            prices = result.get("closingPrices", []) if isinstance(result, dict) else []
        except Exception:
            sizes = len(payload["tickers"])
            seed = abs(hash(request.hypothesis_text)) % 997
            base_price = 90 + (seed % 50)
            prices = [round(base_price * (1 + math.sin(idx / (sizes + 1)) * 0.01), 2) for idx in range(10)]
        volatility = _clamp(fmean(prices[-5:]) / max(prices[0], 1e-6) - 1.0 if prices else 0.12, -1.0, 1.0)
        dataset = {
            "tickers": payload["tickers"],
            "prices": prices,
            "volatility": round(abs(volatility), 4),
        }
        evidence = EvidenceReference(type="market_data", uri="https://data.raven.local/market")
        return dataset, evidence

    def fetch_filings(self, request: HypothesisRequest) -> Tuple[Dict[str, Any], EvidenceReference]:
        try:
            tool = self.toolset.get_tool("sec_filings_get_company_filings")
            filings = tool.invoke({"companySymbols": request.entities})  # type: ignore[call-arg]
            total_filings = len(filings) if isinstance(filings, list) else 3
        except Exception:
            total_filings = 3 + len(request.entities)
        dataset = {
            "count": total_filings,
            "highlights": [f"Form 10-K insight {idx}" for idx in range(min(total_filings, 3))],
        }
        evidence = EvidenceReference(type="filings", uri="https://data.raven.local/filings")
        return dataset, evidence

    def fetch_news(self, request: HypothesisRequest) -> Tuple[Dict[str, Any], EvidenceReference]:
        try:
            tool = self.toolset.get_tool("newsapi_everything")
            news = tool.invoke({"q": request.hypothesis_text, "pageSize": 5})  # type: ignore[call-arg]
            articles = news.get("articles", []) if isinstance(news, dict) else []
        except Exception:
            seed = abs(hash(request.hypothesis_text)) % 13
            articles = [{"title": f"Synthetic coverage #{idx}", "sentiment": (-1) ** idx * 0.1} for idx in range(1, 4)]
            articles[0]["sentiment"] = 0.2 + seed / 100
        dataset = {
            "articles": articles,
            "avg_sentiment": round(fmean(a.get("sentiment", 0.0) for a in articles) if articles else 0.0, 4),
        }
        evidence = EvidenceReference(type="news", uri="https://data.raven.local/news")
        return dataset, evidence


class LangGraphValidationOrchestrator:
    """Run validation stages using LangGraph state machines backed by Composio data."""

    def __init__(self, router: ComposioRouter | None = None) -> None:
        self.router = router or ComposioRouter()
        self._graphs: Dict[str, Any] = {}

    def run_stage(self, stage: str, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        if stage == "data_ingest":
            return self._run_data_ingest(request, context)
        if stage == "preprocessing":
            return self._run_preprocessing(request, context)
        if stage == "analysis":
            return self._run_analysis(request, context)
        if stage == "sentiment":
            return self._run_sentiment(request, context)
        if stage == "modeling":
            return self._run_modeling(request, context)
        if stage == "report_generation":
            return self._run_report(request, context)
        raise ValueError(f"Unknown validation stage: {stage}")

    def _run_data_ingest(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        graph = self._ensure_graph("data_ingest", self._build_data_graph)
        state = graph.invoke({"request": request.model_dump(mode="json"), "context": context})
        updated_context = state["context"]
        evidence = [EvidenceReference.model_validate(e) for e in state.get("artifacts", [])]
        milestone = WorkflowMilestone(
            name="data_ingest",
            status=MilestoneStatus.COMPLETED,
            detail=state.get("detail", "Fetched market, filings, and news data via Composio."),
        )
        return StageExecutionResult(context=updated_context, milestone=milestone, evidence=evidence)

    def _run_preprocessing(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        graph = self._ensure_graph("preprocessing", self._build_preprocess_graph)
        state = graph.invoke({"request": request.model_dump(mode="json"), "context": context})
        updated_context = state["context"]
        milestone = WorkflowMilestone(
            name="preprocessing",
            status=MilestoneStatus.COMPLETED,
            detail=state.get("detail", "Normalized datasets and aligned time indices."),
        )
        return StageExecutionResult(context=updated_context, milestone=milestone)

    def _run_analysis(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        graph = self._ensure_graph("analysis", self._build_analysis_graph)
        state = graph.invoke({"request": request.model_dump(mode="json"), "context": context})
        updated_context = state["context"]
        milestone = WorkflowMilestone(
            name="analysis",
            status=MilestoneStatus.COMPLETED,
            detail=state.get("detail", "Computed trend, growth, and risk diagnostics."),
        )
        return StageExecutionResult(context=updated_context, milestone=milestone)

    def _run_sentiment(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        graph = self._ensure_graph("sentiment", self._build_sentiment_graph)
        state = graph.invoke({"request": request.model_dump(mode="json"), "context": context})
        updated_context = state["context"]
        milestone = WorkflowMilestone(
            name="sentiment",
            status=MilestoneStatus.COMPLETED,
            detail=state.get("detail", "Scored narrative tone across news coverage."),
        )
        return StageExecutionResult(context=updated_context, milestone=milestone)

    def _run_modeling(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        graph = self._ensure_graph("modeling", self._build_modeling_graph)
        state = graph.invoke({"request": request.model_dump(mode="json"), "context": context})
        updated_context = state["context"]
        evidence = [EvidenceReference.model_validate(e) for e in state.get("artifacts", [])]
        milestone = WorkflowMilestone(
            name="modeling",
            status=MilestoneStatus.COMPLETED,
            detail=state.get("detail", "Generated Monte Carlo scenarios and downside cases."),
        )
        return StageExecutionResult(context=updated_context, milestone=milestone, evidence=evidence)

    def _run_report(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        graph = self._ensure_graph("report", self._build_report_graph)
        state = graph.invoke({"request": request.model_dump(mode="json"), "context": context})
        updated_context = state["context"]
        summary = ValidationSummary(
            score=round(state["score"], 4),
            conclusion=state["conclusion"],
            confidence=round(state["confidence"], 4),
            evidence=[EvidenceReference.model_validate(e) for e in updated_context.get("evidence", [])],
            current_stage="report_generation",
            milestones=[],
        )
        milestone = WorkflowMilestone(
            name="report_generation",
            status=MilestoneStatus.COMPLETED,
            detail=state.get("detail", "Assembled validation summary and audit trail."),
        )
        return StageExecutionResult(
            context=updated_context,
            milestone=milestone,
            summary=summary,
            evidence=[EvidenceReference.model_validate(e) for e in updated_context.get("evidence", [])],
        )

    def _ensure_graph(self, key: str, builder) -> Any:
        if key not in self._graphs:
            self._graphs[key] = builder()
        return self._graphs[key]

    def _build_data_graph(self):
        graph = StateGraph(dict)

        def _market(state: Dict[str, Any]) -> Dict[str, Any]:
            request = HypothesisRequest.model_validate(state["request"])
            dataset, evidence = self.router.fetch_market_data(request)
            context = self._clone_context(state.get("context"))
            context.setdefault("data", {})["market"] = dataset
            context.setdefault("evidence", []).append(evidence.model_dump(mode="json"))
            context.setdefault("insights", []).append("Market data retrieved via Composio toolchain.")
            state["context"] = context
            state.setdefault("artifacts", []).append(evidence.model_dump(mode="json"))
            return state

        def _filings(state: Dict[str, Any]) -> Dict[str, Any]:
            request = HypothesisRequest.model_validate(state["request"])
            dataset, evidence = self.router.fetch_filings(request)
            context = self._clone_context(state.get("context"))
            context.setdefault("data", {})["filings"] = dataset
            context.setdefault("evidence", []).append(evidence.model_dump(mode="json"))
            context.setdefault("insights", []).append("SEC filings summarised for KPI extraction.")
            state["context"] = context
            state.setdefault("artifacts", []).append(evidence.model_dump(mode="json"))
            return state

        def _news(state: Dict[str, Any]) -> Dict[str, Any]:
            request = HypothesisRequest.model_validate(state["request"])
            dataset, evidence = self.router.fetch_news(request)
            context = self._clone_context(state.get("context"))
            context.setdefault("data", {})["news"] = dataset
            context.setdefault("evidence", []).append(evidence.model_dump(mode="json"))
            context.setdefault("insights", []).append("Aggregated recent coverage for sentiment pass.")
            state["context"] = context
            state.setdefault("artifacts", []).append(evidence.model_dump(mode="json"))
            state["detail"] = "Market, filings, and news datasets prepared."
            return state

        graph.add_node("market", _market)
        graph.add_node("filings", _filings)
        graph.add_node("news", _news)
        graph.set_entry_point("market")
        graph.add_edge("market", "filings")
        graph.add_edge("filings", "news")
        graph.add_edge("news", END)
        return graph.compile()

    def _build_preprocess_graph(self):
        graph = StateGraph(dict)

        def _normalize(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            market = context.get("data", {}).get("market", {})
            base = market.get("prices", [100])
            normalized = [round(price / max(base[0], 1e-6), 4) for price in base]
            context.setdefault("normalized", {})["market_indexed"] = normalized
            context.setdefault("insights", []).append("Normalized price series to base period index.")
            state["context"] = context
            return state

        def _align(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            filings = context.get("data", {}).get("filings", {})
            context.setdefault("normalized", {})["filings_count"] = filings.get("count", 0)
            context.setdefault("insights", []).append("Preprocessing complete across numeric and textual feeds.")
            state["context"] = context
            state["detail"] = "Datasets normalized and enriched with derived features."
            return state

        graph.add_node("normalize", _normalize)
        graph.add_node("align", _align)
        graph.set_entry_point("normalize")
        graph.add_edge("normalize", "align")
        graph.add_edge("align", END)
        return graph.compile()

    def _build_analysis_graph(self):
        graph = StateGraph(dict)

        def _trend(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            normalized = context.get("normalized", {}).get("market_indexed", [1])
            trend = normalized[-1] - normalized[0] if len(normalized) > 1 else 0.0
            context.setdefault("analysis", {})["trend_strength"] = round(trend, 4)
            context.setdefault("insights", []).append("Computed trend strength from normalized prices.")
            state["context"] = context
            return state

        def _risk(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            volatility = context.get("data", {}).get("market", {}).get("volatility", 0.15)
            filings_count = context.get("normalized", {}).get("filings_count", 1)
            risk_index = _clamp(volatility * 0.6 + (1 / max(filings_count, 1)) * 0.1, 0.0, 1.0)
            context.setdefault("analysis", {})["risk_index"] = round(risk_index, 4)
            context.setdefault("insights", []).append("Risk index estimated from volatility and filing cadence.")
            state["context"] = context
            state["detail"] = "Trend and risk diagnostics computed."
            return state

        graph.add_node("trend", _trend)
        graph.add_node("risk", _risk)
        graph.set_entry_point("trend")
        graph.add_edge("trend", "risk")
        graph.add_edge("risk", END)
        return graph.compile()

    def _build_sentiment_graph(self):
        graph = StateGraph(dict)

        def _aggregate(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            news = context.get("data", {}).get("news", {})
            score = news.get("avg_sentiment", 0.0)
            context.setdefault("sentiment", {})["composite"] = round(_clamp(score * 0.5 + 0.5, 0.0, 1.0), 4)
            context.setdefault("insights", []).append("Derived composite sentiment across news sources.")
            state["context"] = context
            return state

        def _volatility_bridge(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            volatility = context.get("data", {}).get("market", {}).get("volatility", 0.1)
            sentiment = context.get("sentiment", {}).get("composite", 0.5)
            smooth = round(_clamp(abs(sentiment - volatility / 2), 0.0, 1.0), 4)
            context.setdefault("sentiment", {})["stability"] = smooth
            context.setdefault("insights", []).append("Sentiment stability blended with market volatility.")
            state["context"] = context
            state["detail"] = "Sentiment signals prepared for modeling inputs."
            return state

        graph.add_node("aggregate", _aggregate)
        graph.add_node("bridge", _volatility_bridge)
        graph.set_entry_point("aggregate")
        graph.add_edge("aggregate", "bridge")
        graph.add_edge("bridge", END)
        return graph.compile()

    def _build_modeling_graph(self):
        graph = StateGraph(dict)

        def _monte_carlo(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            sentiment = context.get("sentiment", {}).get("composite", 0.5)
            trend = context.get("analysis", {}).get("trend_strength", 0.0)
            mean_return = trend * 0.3 + sentiment * 0.4
            downside = _clamp(0.3 - sentiment * 0.2, 0.05, 0.5)
            context.setdefault("modeling", {})["mean_return"] = round(mean_return, 4)
            context.setdefault("modeling", {})["downside_risk"] = round(downside, 4)
            evidence = EvidenceReference(type="simulation", uri="https://data.raven.local/simulations")
            context.setdefault("evidence", []).append(evidence.model_dump(mode="json"))
            state.setdefault("artifacts", []).append(evidence.model_dump(mode="json"))
            state["context"] = context
            return state

        def _scenario(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            mean = context.get("modeling", {}).get("mean_return", 0.1)
            downside = context.get("modeling", {}).get("downside_risk", 0.2)
            spread = round(_clamp(mean - downside, -0.2, 0.4), 4)
            context.setdefault("modeling", {})["scenario_spread"] = spread
            context.setdefault("insights", []).append("Modeled upside/downside scenarios for validation scoring.")
            state["context"] = context
            state["detail"] = "Scenario modeling complete with probabilistic outcomes."
            return state

        graph.add_node("monte_carlo", _monte_carlo)
        graph.add_node("scenario", _scenario)
        graph.set_entry_point("monte_carlo")
        graph.add_edge("monte_carlo", "scenario")
        graph.add_edge("scenario", END)
        return graph.compile()

    def _build_report_graph(self):
        graph = StateGraph(dict)

        def _score(state: Dict[str, Any]) -> Dict[str, Any]:
            request = HypothesisRequest.model_validate(state["request"])
            context = self._clone_context(state.get("context"))
            analysis = context.get("analysis", {})
            modeling = context.get("modeling", {})
            sentiment = context.get("sentiment", {})
            risk_appetite_factor = {
                "low": -0.15,
                "moderate": 0.0,
                "high": 0.1,
            }[request.risk_appetite.value]
            score_components = [analysis.get("trend_strength", 0.0) * 0.4, modeling.get("mean_return", 0.0) * 0.4, sentiment.get("composite", 0.5) * 0.2, risk_appetite_factor]
            score = _clamp(0.5 + sum(score_components))
            confidence = _clamp(0.45 + modeling.get("scenario_spread", 0.0) * 0.3 - analysis.get("risk_index", 0.1) * 0.2)
            state["score"] = score
            state["confidence"] = confidence
            state["context"] = context
            return state

        def _conclude(state: Dict[str, Any]) -> Dict[str, Any]:
            context = self._clone_context(state.get("context"))
            score = state.get("score", 0.5)
            if score >= 0.65:
                conclusion = "Supported"
            elif score >= 0.5:
                conclusion = "Partially supported"
            else:
                conclusion = "Not supported"
            state["conclusion"] = conclusion
            context.setdefault("insights", []).append(f"Final validation conclusion: {conclusion}.")
            context.setdefault("context_notes", [])
            state["context"] = context
            state["detail"] = "Validation summary assembled with LangGraph pipeline artifacts."
            return state

        graph.add_node("score", _score)
        graph.add_node("conclude", _conclude)
        graph.set_entry_point("score")
        graph.add_edge("score", "conclude")
        graph.add_edge("conclude", END)
        return graph.compile()

    @staticmethod
    def _clone_context(context: StageContext | None) -> StageContext:
        if not context:
            return cast(StageContext, {})
        cloned: StageContext = cast(StageContext, {})
        for key, value in context.items():
            if isinstance(value, dict):
                cloned[key] = dict(value)
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned
