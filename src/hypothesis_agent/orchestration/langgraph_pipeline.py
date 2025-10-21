"""LangGraph orchestration for the RAVEN validation pipeline."""
from __future__ import annotations

import io
import json
import math
import re
import statistics
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, cast

import matplotlib.pyplot as plt
from composio import Composio
from composio.exceptions import ApiKeyNotProvidedError
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.llm import BaseLLM, LLMError, OpenAILLM
from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.tool_catalog import ToolDefinition, load_tool_catalog
from hypothesis_agent.storage.artifact_store import ArtifactStore

StageContext = Dict[str, Any]


RESULT_PREFIX = "RESULT::"
MAX_REPL_ATTEMPTS = 10
STDOUT_CHARACTER_LIMIT = 1600
SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "print": print,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "any": any,
    "all": all,
    "round": round,
    "open": open,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
}


@dataclass(slots=True)
class StageExecutionResult:
    """Container describing the outcome of a single stage."""

    context: StageContext
    milestone: WorkflowMilestone
    evidence: List[EvidenceReference] = field(default_factory=list)
    summary: Optional[ValidationSummary] = None


class ToolHandle(Protocol):
    """Protocol describing the interface for a Composio-like tool."""

    def invoke(self, payload: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        ...


class ToolSet(Protocol):
    """Protocol describing a minimal tool registry."""

    def get_tool(self, name: str) -> ToolHandle:  # pragma: no cover - interface
        ...


class _ComposioTool:
    """Execute a specific Composio tool slug."""

    def __init__(self, *, client: Any, slug: str, user_id: str | None) -> None:
        self._client = client
        self._slug = slug
        self._user_id = user_id

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"slug": self._slug, "arguments": payload}
        if self._user_id:
            kwargs["user_id"] = self._user_id
        response = self._client.tools.execute(**kwargs)
        if not response.get("successful", False):
            error = response.get("error") or "unknown error"
            raise RuntimeError(f"Composio tool '{self._slug}' execution failed: {error}")
        return cast(Dict[str, Any], response.get("data", {}))


class _LazyComposioToolSet:
    """Lazily instantiate the Composio client when tools are requested."""

    def __init__(self, *, user_id: str | None) -> None:
        self._client: Any = None
        self._user_id = user_id

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                self._client = Composio()
            except Exception as exc:
                if isinstance(exc, ApiKeyNotProvidedError):
                    raise RuntimeError(
                        "Composio API key must be provided via COMPOSIO_API_KEY to run the pipeline."
                    ) from exc
                raise
        return self._client

    def get_tool(self, name: str) -> ToolHandle:
        return _ComposioTool(client=self._ensure_client(), slug=name, user_id=self._user_id)


class LangGraphValidationOrchestrator:
    """Run the end-to-end hypothesis validation pipeline using LangGraph."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        llm: BaseLLM | None = None,
        artifact_store: ArtifactStore | None = None,
        toolset: ToolSet | None = None,
        tool_catalog: Dict[str, ToolDefinition] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if llm is not None:
            self.llm = llm
        else:
            if not self.settings.openai_api_key:
                raise LLMError("OpenAI API key must be configured for validation pipeline")
            self.llm = OpenAILLM(api_key=self.settings.openai_api_key, model=self.settings.openai_model)
        self.artifact_store = artifact_store or ArtifactStore.from_path(self.settings.artifact_store_path)
        self.tool_catalog = tool_catalog or load_tool_catalog()
        composio_user_id = self.settings.composio_user_id or None
        self.toolset = toolset or _LazyComposioToolSet(user_id=composio_user_id)

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
        workflow_id = self._ensure_workflow_id(context)
        plan = self._build_validation_plan(request)
        plan_path = self.artifact_store.write_json(workflow_id, "validation_plan", plan).resolve().as_uri()
        context["plan"] = plan
        context.setdefault("insights", []).append(
            f"Planning stage selected {len(plan['data_fetch_tools'])} data tools and {len(plan['analysis_plan'])} analysis steps."
        )
        metadata = context.setdefault("metadata", {})
        metadata.setdefault("plan", {})["artifact"] = plan_path
        milestone = WorkflowMilestone(
            name="plan_generation",
            status=MilestoneStatus.COMPLETED,
            detail=self._format_plan_detail(plan),
        )
        evidence = [EvidenceReference(type="validation_plan", uri=plan_path)]
        for ref in evidence:
            self._record_evidence(context, ref)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_data_collection(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        plan = context.get("plan")
        if not plan or not isinstance(plan, dict):
            raise RuntimeError("Validation plan missing from context; run plan_generation first.")
        tools = plan.get("data_fetch_tools") or []
        if not tools:
            raise RuntimeError("Validation plan did not produce any data fetch tools.")

        executed: List[Dict[str, Any]] = []
        evidence: List[EvidenceReference] = []
        for index, tool in enumerate(tools, start=1):
            slug = tool.get("slug")
            arguments = cast(Dict[str, Any], tool.get("arguments", {}))
            if not slug:
                raise RuntimeError("Plan tool definition missing slug")
            handle = self.toolset.get_tool(slug)
            response = handle.invoke(arguments)
            payload = {
                "slug": slug,
                "arguments": arguments,
                "response": response,
                "description": tool.get("description"),
            }
            artifact_name = f"data_{index:02d}_{slug.lower()}"
            path = self.artifact_store.write_json(workflow_id, artifact_name, payload)
            executed.append(
                {
                    "slug": slug,
                    "arguments": arguments,
                    "artifact": str(path),
                    "uri": path.resolve().as_uri(),
                    "description": tool.get("description"),
                }
            )
            evidence_ref = EvidenceReference(type="data_fetch", uri=path.resolve().as_uri())
            evidence.append(evidence_ref)
            self._record_evidence(context, evidence_ref)

        context["data_sources"] = {
            item["slug"]: {
                "artifact_path": item["uri"],
                "arguments": item["arguments"],
                "description": item.get("description"),
            }
            for item in executed
        }
        context.setdefault("insights", []).append(
            f"Executed {len(executed)} Composio tools for data collection."
        )
        milestone = WorkflowMilestone(
            name="data_collection",
            status=MilestoneStatus.COMPLETED,
            detail=self._format_data_collection_detail(executed),
        )
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_hybrid_analysis(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        plan = context.get("plan")
        data_sources = context.get("data_sources")
        if not plan or not isinstance(plan, dict):
            raise RuntimeError("Validation plan missing from context; run plan_generation first.")
        if not data_sources or not isinstance(data_sources, dict):
            raise RuntimeError("Data sources missing; run data_collection stage first.")

        analysis_plan = cast(List[Dict[str, Any]], plan.get("analysis_plan", []))
        if not analysis_plan:
            raise RuntimeError("Analysis plan missing from validation plan output.")

        analysis_result, attempt_history = self._execute_llm_analysis(
            workflow_id=workflow_id,
            request=request,
            analysis_plan=analysis_plan,
            data_sources=data_sources,
        )

        charts = analysis_result.get("artifacts") or analysis_result.get("charts") or []
        analysis_payload = {
            "steps": analysis_result.get("steps", []),
            "aggregated": analysis_result.get("aggregated", {}),
            "insights": analysis_result.get("insights", []),
            "charts": charts,
            "history": attempt_history,
        }
        metrics_path = self.artifact_store.write_json(workflow_id, "analysis_metrics", analysis_payload).resolve().as_uri()
        evidence = [EvidenceReference(type="analysis_metrics", uri=metrics_path)]
        for ref in evidence:
            self._record_evidence(context, ref)
        context["analysis_results"] = analysis_payload
        context.setdefault("insights", []).extend(analysis_payload.get("insights", [])[:3])
        milestone = WorkflowMilestone(
            name="hybrid_analysis",
            status=MilestoneStatus.COMPLETED,
            detail=self._format_analysis_detail(analysis_payload.get("steps", [])),
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
        workflow_id = self._ensure_workflow_id(context)
        plan = context.get("plan")
        detailed = context.get("detailed_analysis")
        analysis_results = context.get("analysis_results")
        if not plan:
            raise RuntimeError("Validation plan missing; run plan_generation stage first")
        if not detailed:
            raise RuntimeError("Detailed analysis missing; run detailed_analysis stage first")
        if not analysis_results:
            raise RuntimeError("Analysis results missing; run hybrid_analysis stage first")

        plan_steps = self._plan_overview_lines(plan)
        charts = analysis_results.get("charts") or []
        if not charts:
            charts = [self._generate_fallback_chart(workflow_id, analysis_results)]
        report_payload = self.llm.generate_report(
            request=request,
            metrics_overview=analysis_results,
            analysis_summary=detailed["narrative"],
            artifact_paths=charts,
        )
        if not isinstance(report_payload, dict):
            raise RuntimeError("LLM report payload must be a dictionary")
        pdf_path = self._render_report_pdf(
            workflow_id=workflow_id,
            request=request,
            plan_steps=plan_steps,
            analysis_results=analysis_results,
            detailed_analysis=detailed,
            report_payload=report_payload,
        )
        context["report"] = {"payload": report_payload, "pdf_path": pdf_path}
        pdf_evidence = EvidenceReference(type="report_document", uri=pdf_path)
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
    # Planning helpers
    # ------------------------------------------------------------------
    def _ensure_workflow_id(self, context: StageContext) -> str:
        metadata = context.get("metadata") or {}
        workflow_id = metadata.get("workflow_id")
        if not workflow_id:
            raise RuntimeError("Workflow metadata must include a workflow_id")
        return str(workflow_id)

    def _build_validation_plan(self, request: HypothesisRequest) -> Dict[str, Any]:
        symbol = self._canonical_symbol(request.entities)
        selected_tools = self._select_data_fetch_tools(request, symbol)
        analysis_plan = self._build_analysis_plan(request, symbol, selected_tools)
        return {
            "hypothesis": request.hypothesis_text,
            "target_symbol": symbol,
            "data_fetch_tools": selected_tools,
            "analysis_plan": analysis_plan,
        }

    def _canonical_symbol(self, entities: Sequence[str]) -> str:
        if not entities:
            return "SPY"
        return entities[0].strip().upper() or "SPY"

    def _select_data_fetch_tools(self, request: HypothesisRequest, symbol: str) -> List[Dict[str, Any]]:
        text = request.hypothesis_text.lower()
        catalogue = self.tool_catalog
        tools: List[Dict[str, Any]] = []

        def add_tool(slug: str, arguments: Dict[str, Any]) -> None:
            if slug not in catalogue:
                return
            if any(item.get("slug") == slug for item in tools):
                return
            definition = catalogue[slug]
            tools.append(
                {
                    "slug": slug,
                    "description": definition.description,
                    "arguments": arguments,
                    "inputs": definition.inputs,
                }
            )

        add_tool("ALPHA_VANTAGE_TIME_SERIES_MONTHLY_ADJUSTED", {"symbol": symbol})
        add_tool("ALPHA_VANTAGE_COMPANY_OVERVIEW", {"symbol": symbol})
        add_tool("ALPHA_VANTAGE_NEWS_SENTIMENT", {"tickers": symbol})

        profitability_keywords = {"margin", "cash flow", "cashflow", "operating", "profit", "free cash"}
        if any(keyword in text for keyword in profitability_keywords):
            add_tool("ALPHA_VANTAGE_CASH_FLOW", {"symbol": symbol})
            add_tool("ALPHA_VANTAGE_BALANCE_SHEET", {"symbol": symbol})

        if "earnings" in text or "eps" in text:
            add_tool("ALPHA_VANTAGE_EARNINGS", {"symbol": symbol})

        if "calendar" in text or "guidance" in text:
            add_tool("ALPHA_VANTAGE_EARNINGS_CALENDAR", {"symbol": symbol})

        macro_keywords = {"macro", "gdp", "economy", "recession"}
        if any(keyword in text for keyword in macro_keywords):
            add_tool("ALPHA_VANTAGE_REAL_GDP", {})
            add_tool("ALPHA_VANTAGE_SECTOR", {})

        if "currency" in text or "fx" in text or "/" in symbol:
            pair = symbol
            if "/" in pair:
                base, quote = pair.split("/", 1)
            elif len(pair) == 6:
                base, quote = pair[:3], pair[3:]
            else:
                base, quote = "USD", "EUR"
            add_tool("ALPHA_VANTAGE_CURRENCY_EXCHANGE_RATE", {"from_currency": base, "to_currency": quote})
            add_tool("ALPHA_VANTAGE_FX_WEEKLY", {"from_symbol": base, "to_symbol": quote})

        return tools

    def _build_analysis_plan(
        self,
        request: HypothesisRequest,
        symbol: str,
        selected_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tool_slugs = {tool["slug"] for tool in selected_tools}
        plan: List[Dict[str, Any]] = []

        if "ALPHA_VANTAGE_TIME_SERIES_MONTHLY_ADJUSTED" in tool_slugs:
            plan.append(
                {
                    "name": "price_trend_analysis",
                    "description": f"Evaluate trailing price momentum and volatility for {symbol} using monthly adjusted closes.",
                    "source_tools": ["ALPHA_VANTAGE_TIME_SERIES_MONTHLY_ADJUSTED"],
                    "operations": [
                        {
                            "operation": "percent_change",
                            "field": "5. adjusted close",
                            "periods": 12,
                            "label": "Trailing 12m adjusted close change",
                        },
                        {
                            "operation": "volatility",
                            "field": "5. adjusted close",
                            "periods": 12,
                            "label": "Annualized volatility (monthly)",
                        },
                    ],
                }
            )

        if "ALPHA_VANTAGE_COMPANY_OVERVIEW" in tool_slugs:
            plan.append(
                {
                    "name": "profitability_snapshot",
                    "description": "Extract current profitability ratios from company overview.",
                    "source_tools": ["ALPHA_VANTAGE_COMPANY_OVERVIEW"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "OperatingMarginTTM",
                            "label": "Operating margin (TTM)",
                        },
                        {
                            "operation": "latest_value",
                            "field": "ProfitMargin",
                            "label": "Net profit margin",
                        },
                    ],
                }
            )

        if "ALPHA_VANTAGE_CASH_FLOW" in tool_slugs:
            plan.append(
                {
                    "name": "cash_flow_trend",
                    "description": "Assess annual operating cash flow momentum.",
                    "source_tools": ["ALPHA_VANTAGE_CASH_FLOW"],
                    "operations": [
                        {
                            "operation": "yoy_change",
                            "field": "operatingCashflow",
                            "label": "YoY change in operating cash flow",
                        }
                    ],
                }
            )

        if "ALPHA_VANTAGE_BALANCE_SHEET" in tool_slugs:
            plan.append(
                {
                    "name": "balance_sheet_health",
                    "description": "Review leverage and liquidity from latest balance sheet.",
                    "source_tools": ["ALPHA_VANTAGE_BALANCE_SHEET"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "totalAssets",
                            "label": "Total assets (latest)",
                        },
                        {
                            "operation": "latest_value",
                            "field": "totalLiabilities",
                            "label": "Total liabilities (latest)",
                        },
                    ],
                }
            )

        if "ALPHA_VANTAGE_EARNINGS" in tool_slugs:
            plan.append(
                {
                    "name": "earnings_consistency",
                    "description": "Inspect earnings per share trend.",
                    "source_tools": ["ALPHA_VANTAGE_EARNINGS"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "reportedEPS",
                            "label": "Most recent reported EPS",
                        }
                    ],
                }
            )

        if "ALPHA_VANTAGE_NEWS_SENTIMENT" in tool_slugs:
            plan.append(
                {
                    "name": "sentiment_monitor",
                    "description": "Summarize Alpha Vantage news sentiment feed.",
                    "source_tools": ["ALPHA_VANTAGE_NEWS_SENTIMENT"],
                    "operations": [
                        {
                            "operation": "average_sentiment",
                            "field": "overall_sentiment_score",
                            "label": "Average sentiment",
                        }
                    ],
                }
            )

        if "ALPHA_VANTAGE_CURRENCY_EXCHANGE_RATE" in tool_slugs:
            plan.append(
                {
                    "name": "fx_spot_snapshot",
                    "description": "Capture real-time FX rate for exposure analysis.",
                    "source_tools": ["ALPHA_VANTAGE_CURRENCY_EXCHANGE_RATE"],
                    "operations": [
                        {
                            "operation": "latest_rate",
                            "field": "5. Exchange Rate",
                            "label": "Spot exchange rate",
                        }
                    ],
                }
            )

        if "ALPHA_VANTAGE_REAL_GDP" in tool_slugs:
            plan.append(
                {
                    "name": "macro_context",
                    "description": "Use real GDP series for macro backdrop.",
                    "source_tools": ["ALPHA_VANTAGE_REAL_GDP"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "value",
                            "label": "Latest GDP reading",
                        }
                    ],
                }
            )

        return plan

    # ------------------------------------------------------------------
    # Analysis execution helpers
    # ------------------------------------------------------------------
    def _execute_llm_analysis(
        self,
        *,
        workflow_id: str,
        request: HypothesisRequest,
        analysis_plan: List[Dict[str, Any]],
        data_sources: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        artifact_map: Dict[str, str] = {}
        for slug, metadata in data_sources.items():
            uri = metadata.get("artifact_path") or metadata.get("uri")
            if not uri:
                raise RuntimeError(f"Artifact path missing for tool '{slug}'")
            path = Path(str(uri).replace("file://", "")).resolve()
            if not path.exists():
                raise RuntimeError(f"Artifact path for tool '{slug}' does not exist: {path}")
            artifact_map[slug] = str(path)

        history_records: List[Dict[str, Any]] = []
        prompt_history: List[Dict[str, str]] = []

        for attempt in range(1, MAX_REPL_ATTEMPTS + 1):
            response = self.llm.generate_analysis_code(
                request=request,
                analysis_plan=analysis_plan,
                data_artifacts=artifact_map,
                attempt=attempt,
                history=prompt_history,
            )
            code = self._extract_python_code(response)

            stdout_text, stderr_text, result_payload, feedback = self._run_analysis_code(
                code=code,
                artifact_map=artifact_map,
                workflow_id=workflow_id,
            )

            record_status = "success" if result_payload is not None else "error"
            record: Dict[str, Any] = {
                "attempt": attempt,
                "status": record_status,
                "code": self._tail_text(code, STDOUT_CHARACTER_LIMIT),
                "stdout": self._tail_text(stdout_text, STDOUT_CHARACTER_LIMIT),
                "stderr": self._tail_text(stderr_text, STDOUT_CHARACTER_LIMIT),
            }
            if feedback:
                record["feedback"] = self._tail_text(feedback, STDOUT_CHARACTER_LIMIT)
            history_records.append(record)

            prompt_history.append(
                {
                    "attempt": str(attempt),
                    "status": record_status,
                    "stdout": self._tail_text(stdout_text, 600),
                    "stderr": self._tail_text(stderr_text, 600),
                    "feedback": self._tail_text(feedback or record_status, 600),
                }
            )

            if result_payload is not None:
                return result_payload, history_records

        raise RuntimeError("LLM analysis failed to produce a valid result after maximum retries")

    def _run_analysis_code(
        self,
        *,
        code: str,
        artifact_map: Dict[str, str],
        workflow_id: str,
    ) -> tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result_payload: Optional[Dict[str, Any]] = None
        feedback: Optional[str] = None

        environment = self._prepare_analysis_environment(workflow_id, artifact_map)

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, environment, {})
        except Exception:
            feedback = traceback.format_exc()

        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()

        result_line: Optional[str] = None
        for line in stdout_text.splitlines():
            if line.startswith(RESULT_PREFIX):
                result_line = line[len(RESULT_PREFIX) :].strip()

        if result_line is not None:
            try:
                result_payload = json.loads(result_line)
                feedback = None
            except json.JSONDecodeError as exc:
                feedback = f"Failed to parse RESULT payload: {exc}"
        elif feedback is None:
            combined_error = stderr_text.strip()
            feedback = combined_error or "No RESULT:: line emitted in stdout"

        return stdout_text, stderr_text, result_payload, feedback

    def _prepare_analysis_environment(self, workflow_id: str, artifact_map: Dict[str, str]) -> Dict[str, Any]:
        workflow_dir = self.artifact_store.root / workflow_id.replace("/", "_")
        workflow_dir.mkdir(parents=True, exist_ok=True)

        def load_json_artifact(slug: str) -> Any:
            if slug not in artifact_map:
                raise KeyError(f"Unknown data artifact slug: {slug}")
            path = Path(artifact_map[slug])
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        def write_json_artifact(name: str, payload: Dict[str, Any]) -> str:
            target = self.artifact_store.write_json(workflow_id, name, payload)
            return str(target.resolve())

        env: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "json": json,
            "math": math,
            "statistics": statistics,
            "Path": Path,
            "DATA_ARTIFACTS": artifact_map,
            "load_json_artifact": load_json_artifact,
            "write_json_artifact": write_json_artifact,
            "ARTIFACT_OUTPUT_DIR": str(workflow_dir),
        }
        return env

    @staticmethod
    def _extract_python_code(response: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    @staticmethod
    def _tail_text(text: str, limit: int) -> str:
        snippet = text.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[-limit:]

    def _generate_time_series_chart(self, workflow_id: str, series: List[tuple[str, float]]) -> str:
        if not series:
            raise RuntimeError("Cannot render chart; time series empty")
        dates = [item[0] for item in series]
        values = [item[1] for item in series]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(dates, values, color="#0B69FF", linewidth=1.6)
        ax.set_title("Price Trend", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.autofmt_xdate()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        chart_path = self.artifact_store.write_bytes(workflow_id, "price_trend.png", buffer.getvalue()).resolve().as_uri()
        buffer.close()
        return chart_path

    def _generate_fallback_chart(self, workflow_id: str, analysis_results: Dict[str, Any]) -> str:
        steps = analysis_results.get("steps", [])
        metrics = []
        for step in steps:
            for output in step.get("outputs", []):
                label = output.get("label")
                value = output.get("value")
                if isinstance(value, (int, float)) and label:
                    metrics.append((label, float(value)))
        if not metrics:
            raise RuntimeError("No numeric metrics available to build fallback chart")
        labels, values = zip(*metrics[:8])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(labels, values, color="#10B981")
        ax.set_ylabel("Value")
        ax.set_title("Key Metrics Snapshot")
        ax.tick_params(axis="x", rotation=30)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        chart_path = self.artifact_store.write_bytes(workflow_id, "metrics_snapshot.png", buffer.getvalue()).resolve().as_uri()
        buffer.close()
        return chart_path

    # ------------------------------------------------------------------
    # Rendering and delivery helpers
    # ------------------------------------------------------------------
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
        body_lines.append("Regards,")
        body_lines.append("RAVEN Validation Platform")

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
    def _plan_overview_lines(self, plan: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for tool in plan.get("data_fetch_tools", []):
            slug = tool.get("slug")
            description = tool.get("description")
            arguments = tool.get("arguments", {})
            arg_repr = ", ".join(f"{key}={value}" for key, value in arguments.items())
            lines.append(f"Fetch {slug} ({description}) with {arg_repr}")
        for step in plan.get("analysis_plan", []):
            operations = step.get("operations", [])
            op_names = ", ".join(str(op.get("label")) for op in operations)
            lines.append(f"Analysis step '{step.get('name')}' → {op_names}")
        return lines

    def _format_plan_detail(self, plan: Dict[str, Any]) -> str:
        fetch_slugs = [tool.get("slug", "") for tool in plan.get("data_fetch_tools", [])]
        analysis_names = [step.get("name", "") for step in plan.get("analysis_plan", [])]
        return (
            f"Plan drafted with tools: {', '.join(filter(None, fetch_slugs))}; "
            f"analysis steps: {', '.join(filter(None, analysis_names))}."
        )

    def _format_data_collection_detail(self, executed: List[Dict[str, Any]]) -> str:
        if not executed:
            return "No data fetch tools executed."
        slugs = ", ".join(item["slug"] for item in executed if item.get("slug"))
        return f"Composio data ingested from: {slugs}."

    def _format_analysis_detail(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No analysis steps executed."
        names = ", ".join(step.get("name", "") for step in results)
        return f"Completed hybrid analytics: {names}."

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _load_json_from_path(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Artifact {path} did not contain a JSON object")
        return payload

    def _score_validation(self, analysis_results: Dict[str, Any]) -> tuple[float, float]:
        aggregated = analysis_results.get("aggregated", {})
        price_change = float(aggregated.get("trailing_12m_adjusted_close_change", 0.0) or 0.0)
        volatility = float(aggregated.get("annualized_volatility_(monthly)", 0.0) or 0.0)
        sentiment = float(aggregated.get("average_sentiment", 0.0) or 0.0)
        operating_margin = float(aggregated.get("operating_margin_(ttm)", 0.0) or 0.0)

        score = 0.5
        score += price_change * 0.35
        score += sentiment * 0.2
        score += operating_margin * 0.1
        score -= max(volatility, 0.0) * 0.25
        confidence = 0.6
        confidence += sentiment * 0.15
        confidence += operating_margin * 0.1
        confidence -= max(volatility, 0.0) * 0.2
        score = _clamp(score, 0.0, 1.0)
        confidence = _clamp(confidence, 0.0, 1.0)
        return round(score, 4), round(confidence, 4)

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


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
