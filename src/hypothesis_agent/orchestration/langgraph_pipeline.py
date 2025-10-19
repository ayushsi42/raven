"""LangGraph orchestration for the RAVEN validation pipeline."""
from __future__ import annotations

import io
import json
import textwrap
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Protocol, cast

import matplotlib.pyplot as plt
from composio import Composio
from composio.exceptions import ApiKeyNotProvidedError
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.connectors.news import NewsClient
from hypothesis_agent.connectors.sec import SecFilingsClient
from hypothesis_agent.connectors.yahoo import YahooFinanceClient
from hypothesis_agent.llm import BaseLLM, LLMError, OpenAILLM
from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.storage.artifact_store import ArtifactStore


StageContext = Dict[str, Any]


@dataclass(slots=True)
class StageExecutionResult:
    """Container describing the outcome of a single stage."""

    context: StageContext
    milestone: WorkflowMilestone
    evidence: List[EvidenceReference] = field(default_factory=list)
    summary: Optional[ValidationSummary] = None


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


class ToolHandle(Protocol):
    """Protocol describing the interface for a Composio-like tool."""

    def invoke(self, payload: Dict[str, Any]) -> Any:
        ...


class ToolSet(Protocol):
    """Protocol describing a minimal tool registry."""

    def get_tool(self, name: str) -> ToolHandle:
        ...


class _ComposioTool:
    """Execute a specific Composio tool slug."""

    def __init__(self, *, client: Any, slug: str) -> None:
        self._client = client
        self._slug = slug

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.tools.execute(slug=self._slug, arguments=payload)
        if not response.get("successful", False):
            error = response.get("error") or "unknown error"
            raise RuntimeError(f"Composio tool '{self._slug}' execution failed: {error}")
        return cast(Dict[str, Any], response.get("data", {}))


class _LazyComposioToolSet:
    """Lazily instantiate the Composio client when tools are requested."""

    def __init__(self) -> None:
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                self._client = Composio()
            except Exception as exc:
                if isinstance(exc, ApiKeyNotProvidedError):
                    raise RuntimeError(
                        "Composio API key must be provided via COMPOSIO_API_KEY to run the delivery stage."
                    ) from exc
                raise
        return self._client

    def get_tool(self, name: str) -> ToolHandle:
        return _ComposioTool(client=self._ensure_client(), slug=name)


class LangGraphValidationOrchestrator:
    """Run the end-to-end hypothesis validation pipeline using LangGraph."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        llm: BaseLLM | None = None,
    yahoo_client: YahooFinanceClient | None = None,
    sec_client: SecFilingsClient | None = None,
    news_client: NewsClient | None = None,
        artifact_store: ArtifactStore | None = None,
        toolset: ToolSet | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if llm is not None:
            self.llm = llm
        else:
            if not self.settings.openai_api_key:
                raise LLMError("OpenAI API key must be configured for validation pipeline")
            self.llm = OpenAILLM(api_key=self.settings.openai_api_key, model=self.settings.openai_model)
        self.yahoo = yahoo_client or YahooFinanceClient()
        self.sec = sec_client
        if not self.settings.alpha_vantage_api_key:
            raise RuntimeError("Alpha Vantage API key must be configured for news collection")
        self.news = news_client or NewsClient(api_key=self.settings.alpha_vantage_api_key)
        self.artifact_store = artifact_store or ArtifactStore.from_path(self.settings.artifact_store_path)
        self.toolset = toolset or _LazyComposioToolSet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_stage(self, stage: str, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        stage = stage.lower()
        context = self._clone_context(context)
        if stage == "plan_generation":
            return self._run_plan_generation(request, context)
        if stage == "data_collection":
            return self._run_data_collection(request, context)
        if stage == "analysis_planning":
            return self._run_analysis_planning(request, context)
        if stage == "hybrid_analysis":
            return self._run_hybrid_analysis(request, context)
        if stage == "detailed_analysis":
            return self._run_detailed_analysis(request, context)
        if stage == "report_generation":
            return self._run_report_generation(request, context)
        if stage == "delivery":
            return self._run_delivery(request, context)
        raise ValueError(f"Unknown validation stage '{stage}'")

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------
    def _run_plan_generation(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        plan_steps = self.llm.generate_data_plan(request)
        if not plan_steps:
            raise RuntimeError("LLM failed to generate data collection plan")
        context["plan"] = {"steps": plan_steps}
        context.setdefault("insights", []).append("Generated LLM-backed data acquisition plan.")
        milestone = WorkflowMilestone(
            name="plan_generation",
            status=MilestoneStatus.COMPLETED,
            detail="Validation plan drafted via OpenAI planner.",
        )
        return StageExecutionResult(context=context, milestone=milestone)

    def _run_data_collection(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        metadata = context.setdefault("metadata", {})
        workflow_id = metadata.get("workflow_id")
        if not workflow_id:
            raise RuntimeError("Workflow metadata must include a workflow_id before data collection.")
        if not request.entities:
            raise RuntimeError("At least one entity ticker is required for data collection.")

        primary_ticker = request.entities[0]
        market_info = self._collect_market_data(request, workflow_id, primary_ticker)
        filings_info = None
        if self.sec is not None:
            try:
                filings_info = self._collect_filings_data(workflow_id, primary_ticker)
            except Exception as exc:
                context.setdefault("insights", []).append(f"SEC fetch skipped: {exc}")
                filings_info = None
        news_info = self._collect_news_data(workflow_id, request.entities)

        data_sources: Dict[str, Any] = {"market": market_info, "news": news_info}
        if filings_info is not None:
            data_sources["filings"] = filings_info

        context["data_sources"] = data_sources
        insights = context.setdefault("insights", [])
        if filings_info is not None:
            insights.append("Fetched live market, SEC, and news datasets.")
        else:
            insights.append("Fetched live market and news datasets.")

        evidence = [EvidenceReference(type="market_data", uri=market_info["path"])]
        if filings_info is not None:
            evidence.append(EvidenceReference(type="sec_filings", uri=filings_info["path"]))
        evidence.append(EvidenceReference(type="news_sentiment", uri=news_info["path"]))
        for ref in evidence:
            self._record_evidence(context, ref)

        milestone = WorkflowMilestone(
            name="data_collection",
            status=MilestoneStatus.COMPLETED,
            detail="Raw datasets collected from external connectors.",
        )
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_analysis_planning(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        data_sources = context.get("data_sources") or {}
        if not data_sources:
            raise RuntimeError("Data sources missing; run data_collection stage first.")

        data_overview = {
            name: payload.get("summary")
            for name, payload in data_sources.items()
        }
        analysis_plan = self.llm.generate_analysis_plan(request, data_overview)
        if not analysis_plan:
            raise RuntimeError("LLM failed to generate analysis plan")

        context["analysis_plan"] = analysis_plan
        context.setdefault("insights", []).append("LLM produced the quantitative analysis blueprint.")
        milestone = WorkflowMilestone(
            name="analysis_planning",
            status=MilestoneStatus.COMPLETED,
            detail="Analysis plan constructed from collected dataset summaries.",
        )
        return StageExecutionResult(context=context, milestone=milestone)

    def _run_hybrid_analysis(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        data_sources = context.get("data_sources") or {}
        if not data_sources:
            raise RuntimeError("Data sources missing; run data_collection stage first.")
        metadata = context.get("metadata") or {}
        workflow_id = metadata.get("workflow_id")
        if not workflow_id:
            raise RuntimeError("Workflow ID missing from context metadata.")

        market_analysis, chart_path = self._compute_market_metrics(workflow_id, data_sources["market"])
        filings_analysis: Dict[str, Any] | None = None
        if "filings" in data_sources:
            filings_analysis = self._compute_filings_metrics(data_sources["filings"])
        news_analysis = self._compute_news_metrics(data_sources["news"])

        analysis_results = {
            "market": market_analysis,
            "news": news_analysis,
            "charts": [chart_path],
        }
        if filings_analysis is not None:
            analysis_results["filings"] = filings_analysis
        aggregated_insights = []
        aggregated_insights.extend(market_analysis.get("insights", []))
        aggregated_insights.extend(news_analysis.get("insights", []))
        if filings_analysis is not None:
            aggregated_insights.extend(filings_analysis.get("insights", []))
        analysis_results["insights"] = aggregated_insights
        metrics_path = self.artifact_store.write_json(workflow_id, "analysis_metrics", analysis_results).resolve().as_uri()
        analysis_results["metrics_path"] = metrics_path
        context["analysis_results"] = analysis_results
        insights_bucket = context.setdefault("insights", [])
        insights_bucket.append("Hybrid quantitative and qualitative analytics computed.")
        insights_bucket.extend(insight for insight in aggregated_insights[:3])

        evidence = [EvidenceReference(type="analysis_metrics", uri=metrics_path), EvidenceReference(type="chart", uri=chart_path)]
        for ref in evidence:
            self._record_evidence(context, ref)

        milestone = WorkflowMilestone(
            name="hybrid_analysis",
            status=MilestoneStatus.COMPLETED,
            detail="Quantitative diagnostics executed with chart artifacts persisted.",
        )
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_detailed_analysis(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        analysis_results = context.get("analysis_results")
        if not analysis_results:
            raise RuntimeError("Analysis results missing; run hybrid_analysis stage first.")

        narrative = self.llm.generate_detailed_analysis(request, analysis_results)
        if not narrative:
            raise RuntimeError("LLM failed to generate detailed analysis narrative")

        detailed_analysis = {
            "narrative": narrative,
            "metrics": analysis_results,
        }
        context["detailed_analysis"] = detailed_analysis
        context.setdefault("insights", []).append("LLM synthesized detailed narrative from computed metrics.")

        milestone = WorkflowMilestone(
            name="detailed_analysis",
            status=MilestoneStatus.COMPLETED,
            detail="Narrative synthesis produced from hybrid analytics.",
        )
        return StageExecutionResult(context=context, milestone=milestone)

    def _run_report_generation(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        metadata = context.get("metadata") or {}
        workflow_id = metadata.get("workflow_id")
        if not workflow_id:
            raise RuntimeError("Workflow ID missing from context metadata.")
        detailed = context.get("detailed_analysis")
        if not detailed:
            raise RuntimeError("Detailed analysis missing; run detailed_analysis stage first.")
        analysis_results = detailed.get("metrics")
        if not analysis_results:
            raise RuntimeError("Detailed analysis metrics missing; run detailed_analysis stage first.")
        if not isinstance(analysis_results, dict):
            raise RuntimeError("Detailed analysis metrics payload malformed; expected mapping")
        plan = context.get("plan")
        if not plan or not plan.get("steps"):
            raise RuntimeError("Validation plan missing; run plan_generation stage first.")
        plan_steps = plan["steps"]
        if not isinstance(plan_steps, list) or not plan_steps:
            raise RuntimeError("Plan steps payload malformed; expected non-empty list of strings.")
        if not all(isinstance(step, str) and step.strip() for step in plan_steps):
            raise RuntimeError("Plan steps must be non-empty strings")
        chart_artifacts = analysis_results.get("charts")
        if not chart_artifacts or not isinstance(chart_artifacts, list):
            raise RuntimeError("Analysis results missing chart artifacts for report generation")
        if not all(isinstance(item, str) for item in chart_artifacts):
            raise RuntimeError("Chart artifact paths must be strings")

        report_payload = self.llm.generate_report(
            request=request,
            metrics_overview=analysis_results,
            analysis_summary=detailed["narrative"],
            artifact_paths=chart_artifacts,
        )
        if not isinstance(report_payload, dict):
            raise RuntimeError("LLM report payload must be a dictionary")

        report_pdf = self._render_report_pdf(
            workflow_id=workflow_id,
            request=request,
            plan_steps=plan_steps,
            analysis_results=analysis_results,
            detailed_analysis=detailed,
            report_payload=report_payload,
        )

        context["report"] = {
            "payload": report_payload,
            "pdf_path": report_pdf,
        }
        pdf_evidence = EvidenceReference(type="report_document", uri=report_pdf)
        self._record_evidence(context, pdf_evidence)

        score, confidence = self._score_validation(analysis_results)
        summary = ValidationSummary(
            score=score,
            conclusion=self._determine_conclusion(score),
            confidence=confidence,
            evidence=[EvidenceReference.model_validate(ev) for ev in context.get("evidence", [])],
            current_stage="report_generation",
            milestones=[],
        )

        milestone = WorkflowMilestone(
            name="report_generation",
            status=MilestoneStatus.COMPLETED,
            detail="Investment memo compiled with quantitative backing.",
        )
        return StageExecutionResult(context=context, milestone=milestone, evidence=[pdf_evidence], summary=summary)

    def _run_delivery(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        report = context.get("report") or {}
        pdf_path = report.get("pdf_path")
        payload = report.get("payload") or {}
        if not pdf_path or not payload:
            raise RuntimeError("Report assets missing; run report_generation stage first.")

        recipient = self.settings.notification_email
        if not recipient:
            raise RuntimeError("notification_email must be configured for delivery stage")

        self._send_report_via_tool(
            recipient=recipient,
            request=request,
            pdf_uri=pdf_path,
            report_payload=payload,
        )
        context.setdefault("insights", []).append(f"Report delivered to {recipient} via Composio tool router.")

        milestone = WorkflowMilestone(
            name="delivery",
            status=MilestoneStatus.COMPLETED,
            detail=f"Report emailed to {recipient}.",
        )
        return StageExecutionResult(context=context, milestone=milestone)

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------
    def _collect_market_data(self, request: HypothesisRequest, workflow_id: str, ticker: str) -> Dict[str, Any]:
        series = self.yahoo.fetch_daily_prices(ticker, request.time_horizon.start, request.time_horizon.end).prices
        payload = {"ticker": ticker, "series": series}
        path = self.artifact_store.write_json(workflow_id, "market_data", payload).resolve().as_uri()
        summary = {
            "ticker": ticker,
            "observations": len(series),
            "start": series[0]["date"] if series else None,
            "end": series[-1]["date"] if series else None,
        }
        return {"path": path, "summary": summary}

    def _collect_filings_data(self, workflow_id: str, ticker: str) -> Dict[str, Any]:
        records = [self._serialize_record(record) for record in self.sec.fetch_recent_filings(ticker)]
        payload = {"ticker": ticker, "records": records}
        path = self.artifact_store.write_json(workflow_id, "sec_filings", payload).resolve().as_uri()
        summary = {
            "ticker": ticker,
            "count": len(records),
            "forms": sorted({record.get("filing_type") for record in records if record.get("filing_type")}),
        }
        return {"path": path, "summary": summary}

    def _collect_news_data(self, workflow_id: str, tickers: List[str]) -> Dict[str, Any]:
        articles = [self._serialize_record(article) for article in self.news.fetch_sentiment(tickers)]
        payload = {"tickers": tickers, "articles": articles}
        path = self.artifact_store.write_json(workflow_id, "news_sentiment", payload).resolve().as_uri()
        summary = {
            "tickers": tickers,
            "article_count": len(articles),
            "avg_sentiment": round(mean([article.get("sentiment", 0.0) for article in articles]), 4) if articles else 0.0,
        }
        return {"path": path, "summary": summary}

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def _compute_market_metrics(self, workflow_id: str, market_info: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        payload = self._load_json(market_info["path"])
        series = payload.get("series", [])

        ticker = payload.get("ticker")
        if not ticker:
            raise RuntimeError("Market data payload missing ticker symbol")

        metrics_code = textwrap.dedent(
            """
            import json
            from statistics import mean, pstdev

            label = ticker or "Asset"
            closes = [float(item["close"]) for item in series if item.get("close") is not None]
            if len(closes) < 2:
                raise ValueError("Insufficient market data for hybrid analysis.")

            returns = [(closes[idx] / closes[idx - 1]) - 1 for idx in range(1, len(closes))]
            volatility = pstdev(returns) if len(returns) > 1 else 0.0
            avg_return = mean(returns)
            trading_days = len(closes)
            years = trading_days / 252 if trading_days else 1.0
            cagr = (closes[-1] / closes[0]) ** (1 / years) - 1 if years > 0 else 0.0

            insights = [
                f"{label}: CAGR {cagr:.2%} with volatility {volatility:.2%}.",
                f"{label}: Average session return {avg_return:.2%} across {trading_days} trading days.",
            ]

            json.dumps(
                {
                    "metrics": {
                        "start_price": round(closes[0], 4),
                        "end_price": round(closes[-1], 4),
                        "cagr": round(cagr, 4),
                        "volatility": round(volatility, 4),
                        "avg_return": round(avg_return, 4),
                    },
                    "insights": insights,
                }
            )
            """
        )

        metrics_raw = self._run_python_repl(metrics_code, locals={"series": series, "ticker": ticker})
        chart_path = self._generate_price_chart(workflow_id, ticker, series)
        return self._parse_analysis_json(metrics_raw), chart_path

    def _compute_filings_metrics(self, filings_info: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._load_json(filings_info["path"])
        records = payload.get("records", [])
        if not records:
            raise RuntimeError("SEC filings dataset is empty.")
        latest = sorted(records, key=lambda item: item.get("filed", ""), reverse=True)
        metrics = {
            "filing_count": len(records),
            "recent_forms": [record.get("filing_type") for record in latest[:5]],
            "latest_filed": latest[0].get("filed"),
        }
        insights = [
            f"Filings cadence steady with {metrics['filing_count']} submissions in record set.",
            f"Most recent filing type: {metrics['recent_forms'][0]} dated {metrics['latest_filed']}."
            if metrics["recent_forms"] and metrics["latest_filed"]
            else "Reviewed recent regulatory activity for cadence context.",
        ]
        return {"metrics": metrics, "insights": insights}

    def _compute_news_metrics(self, news_info: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._load_json(news_info["path"])
        articles = payload.get("articles", [])
        if not articles:
            raise RuntimeError("News sentiment dataset is empty.")
        sentiments = [float(article.get("sentiment", 0.0)) for article in articles]
        avg_sentiment = round(mean(sentiments), 4)
        metrics = {
            "article_count": len(articles),
            "avg_sentiment": avg_sentiment,
            "sentiment_range": [round(min(sentiments), 4), round(max(sentiments), 4)],
        }
        stance = "constructive" if avg_sentiment > 0 else "cautious" if avg_sentiment < 0 else "balanced"
        insights = [
            f"News flow {stance} with average sentiment {avg_sentiment:+.2f} across {metrics['article_count']} articles.",
            "Sentiment dispersion captured via range for volatility context.",
        ]
        return {"metrics": metrics, "insights": insights}

    # ------------------------------------------------------------------
    # Rendering and delivery helpers
    # ------------------------------------------------------------------
    def _generate_price_chart(self, workflow_id: str, ticker: str, series: List[Dict[str, Any]]) -> str:
        if not ticker:
            raise RuntimeError("Market data payload missing ticker symbol for chart generation")
        if not series:
            raise RuntimeError("Price series is empty; cannot generate chart")

        dates: List[Any] = []
        closes: List[float] = []
        try:
            for row in series:
                dates.append(row["date"])
                closes.append(float(row["close"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Price series missing required fields for chart generation") from exc

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(dates, closes, color="#0B69FF", linewidth=1.6)
        ax.set_title(f"{ticker} Closing Prices", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.autofmt_xdate()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        chart_path = self.artifact_store.write_bytes(workflow_id, "price_trend.png", buffer.getvalue()).resolve().as_uri()
        buffer.close()
        return chart_path

    def _render_report_pdf(
        self,
        *,
        workflow_id: str,
        request: HypothesisRequest,
        plan_steps: List[str],
        analysis_results: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        report_payload: Dict[str, Any],
    ) -> str:
        if detailed_analysis.get("narrative") is None:
            raise RuntimeError("Detailed analysis narrative missing for report rendering")
        if not plan_steps:
            raise RuntimeError("Plan steps missing for report rendering")
        if not analysis_results:
            raise RuntimeError("Analysis results missing for report rendering")

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=LETTER)
        pdf.setTitle(f"RAVEN Validation Report - {workflow_id}")

        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, 720, "RAVEN Hypothesis Validation Report")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(72, 702, f"Workflow: {workflow_id}")
        pdf.drawString(72, 688, f"Hypothesis: {request.hypothesis_text}")

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, 660, "Plan Overview")
        text_obj = pdf.beginText(72, 644)
        text_obj.setFont("Helvetica", 10)
        for idx, step in enumerate(plan_steps, start=1):
            text_obj.textLine(f"{idx}. {step}")
        pdf.drawText(text_obj)

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, 612, "Key Metrics")
        text_obj = pdf.beginText(72, 596)
        text_obj.setFont("Helvetica", 10)
        metrics_json = json.dumps(analysis_results, indent=2)
        for line in metrics_json.splitlines():
            text_obj.textLine(line)
        pdf.drawText(text_obj)

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, 468, "Narrative Summary")
        text_obj = pdf.beginText(72, 452)
        text_obj.setFont("Helvetica", 10)
        narrative = str(detailed_analysis["narrative"])
        for line in narrative.splitlines():
            text_obj.textLine(line)
        pdf.drawText(text_obj)

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, 360, "LLM Report Highlights")
        text_obj = pdf.beginText(72, 344)
        text_obj.setFont("Helvetica", 10)
        for key in ("executive_summary", "key_findings", "risks", "next_steps"):
            value = report_payload.get(key)
            if value is None:
                continue
            heading = key.replace("_", " ").title()
            text_obj.textLine(f"{heading}:")
            if isinstance(value, list):
                for item in value:
                    text_obj.textLine(f"  - {item}")
            else:
                text_obj.textLine(f"  {value}")
            text_obj.textLine("")
        pdf.drawText(text_obj)

        pdf.showPage()
        pdf.save()
        pdf_bytes = buffer.getvalue()
        buffer.close()
        pdf_path = self.artifact_store.write_bytes(workflow_id, "validation_report.pdf", pdf_bytes).resolve().as_uri()
        return pdf_path

    def _send_report_via_tool(
        self,
        *,
        recipient: str,
        request: HypothesisRequest,
        pdf_uri: str,
        report_payload: Dict[str, Any],
    ) -> None:
        tool = self.toolset.get_tool("gmail_send_email")
        subject = f"RAVEN Validation Report: {request.hypothesis_text[:60]}"
        key_findings = report_payload.get("key_findings", [])
        body_lines = [
            "Hello,",
            "",
            "Please find the attached validation report generated by RAVEN.",
        ]
        if key_findings:
            body_lines.append("")
            body_lines.append("Key Findings:")
            for finding in key_findings[:5]:
                body_lines.append(f"- {finding}")
        body_lines.append("")
        body_lines.append("Regards,\nRAVEN Validation Platform")

        attachment_path = pdf_uri.replace("file://", "")
        tool.invoke(
            {
                "to": recipient,
                "subject": subject,
                "body": "\n".join(body_lines),
                "attachments": [attachment_path],
            }
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _run_python_repl(self, code: str, *, locals: Dict[str, Any]) -> str:
        try:
            from langchain_experimental.tools.python.tool import PythonREPLTool  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "langchain-experimental must be installed to execute Python analysis snippets."
            ) from exc

        tool = PythonREPLTool(locals=dict(locals))
        result = tool.run(code)
        if result is None:
            raise RuntimeError("Python REPL returned no output")
        return str(result).strip()

    @staticmethod
    def _parse_analysis_json(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Python REPL returned invalid JSON payload") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Python REPL analysis payload malformed; expected mapping")
        metrics = parsed.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("Python REPL analysis payload missing 'metrics' mapping")
        insights = parsed.get("insights", [])
        if not isinstance(insights, list):
            raise RuntimeError("Python REPL analysis payload 'insights' must be a list")
        payload = dict(parsed)
        payload["metrics"] = metrics
        payload["insights"] = insights
        return payload

    @staticmethod
    def _serialize_record(record: Any) -> Dict[str, Any]:
        if isinstance(record, dict):
            return dict(record)
        if hasattr(record, "model_dump"):
            return cast(Dict[str, Any], record.model_dump())
        if is_dataclass(record):
            return cast(Dict[str, Any], asdict(record))
        if hasattr(record, "__dict__"):
            return dict(vars(record))
        raise TypeError(f"Unsupported record type for serialization: {type(record)!r}")

    def _load_json(self, uri: str) -> Dict[str, Any]:
        path = Path(uri.replace("file://", ""))
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _score_validation(self, analysis_results: Dict[str, Any]) -> tuple[float, float]:
        market = (analysis_results.get("market") or {}).get("metrics", {})
        news = (analysis_results.get("news") or {}).get("metrics", {})
        filings = (analysis_results.get("filings") or {}).get("metrics", {})
        score = 0.5
        score += float(market.get("cagr", 0.0)) * 0.4
        score += float(news.get("avg_sentiment", 0.0)) * 0.2
        score -= float(market.get("volatility", 0.0)) * 0.3
        score += min(float(filings.get("filing_count", 0)), 10.0) / 100
        confidence = 0.6
        confidence -= float(market.get("volatility", 0.0)) * 0.1
        confidence += float(news.get("avg_sentiment", 0.0)) * 0.2
        confidence += min(float(filings.get("filing_count", 0)), 5.0) / 50
        return round(_clamp(score, 0.0, 1.0), 4), round(_clamp(confidence, 0.0, 1.0), 4)

    @staticmethod
    def _determine_conclusion(score: float) -> str:
        if score >= 0.7:
            return "Supported"
        if score >= 0.5:
            return "Partially supported"
        return "Not supported"

    def _record_evidence(self, context: StageContext, evidence: EvidenceReference) -> None:
        context.setdefault("evidence", []).append(evidence.model_dump(mode="json"))
        context.setdefault("artifacts", []).append(evidence.model_dump(mode="json"))

    @staticmethod
    def _record_stage_metadata(context: StageContext, stage: str, key: str, status: str, detail: str | None = None) -> None:
        metadata = context.setdefault("metadata", {})
        stages = metadata.setdefault("stages", {})
        stage_entry = stages.setdefault(stage, {"steps": {}})
        steps = stage_entry.setdefault("steps", {})
        steps[key] = {"status": status, "detail": detail}

    @staticmethod
    def _clone_context(context: StageContext | None) -> StageContext:
        if not context:
            return cast(StageContext, {})
        cloned: StageContext = cast(StageContext, {})
        for key, value in context.items():
            if isinstance(value, dict):
                cloned[key] = dict(value)
            elif isinstance(value, list):
                cloned[key] = [dict(item) if isinstance(item, dict) else item for item in value]
            else:
                cloned[key] = value
        return cloned
