"""LangGraph orchestration for the RAVEN validation pipeline."""
from __future__ import annotations

import io
import json
import math
import re
import statistics
import traceback
import os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast
import textwrap

import matplotlib

# Use a headless backend for deterministic chart generation in tests and workers.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from hypothesis_agent.orchestration.python_sandbox import PythonSandbox

from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.llm import BaseLLM, LLMError, OpenAILLM
from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.yfinance_tools import YFinanceToolSet, ToolHandle, ToolSet
from hypothesis_agent.orchestration.tool_catalog import ToolDefinition, load_tool_catalog
from hypothesis_agent.storage.artifact_store import ArtifactStore

StageContext = Dict[str, Any]


RESULT_PREFIX = "RESULT::"
MAX_REPL_ATTEMPTS = 5
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


def get_data_format(artifact_map: Dict[str, str]) -> Dict[str, str]:
    """Produce compact JSON schema hints for each artifact path."""
    summaries: Dict[str, str] = {}
    for slug, path_str in artifact_map.items():
        try:
            payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        except Exception:
            # Ignore unreadable artifacts; the LLM will rely on raw files.
            continue
        summary_payload = _summarize_payload(payload)
        try:
            summary_text = json.dumps(summary_payload, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            summary_text = str(summary_payload)
        if len(summary_text) > 400:
            summary_text = summary_text[:397] + "..."
        summaries[slug] = summary_text
    return summaries


def _summarize_payload(payload: Any, depth: int = 0) -> Any:
    if isinstance(payload, dict):
        keys = list(payload.keys())
        summary: Dict[str, Any] = {"keys": keys[:16]}  # Show even more keys
        first_key = keys[0] if keys else None
        if first_key is not None:
            summary["sample_key"] = first_key
            summary["sample"] = _summarize_value(payload[first_key], depth + 1)
        return summary
    if isinstance(payload, list):
        summary_list: Dict[str, Any] = {"type": "array", "length": len(payload)}
        if payload and isinstance(payload[0], dict):
            # For list of records, show the keys of the first record
            summary_list["item_keys"] = list(payload[0].keys())[:16]
            summary_list["sample"] = _summarize_value(payload[0], depth + 1)
        elif payload:
            summary_list["sample"] = _summarize_value(payload[0], depth + 1)
        return summary_list
    return _summarize_value(payload, depth)


def _summarize_value(value: Any, depth: int) -> Any:
    if depth >= 2:
        if isinstance(value, dict):
            return {"keys": list(value.keys())[:6]}
        if isinstance(value, list):
            return {"type": "array", "length": len(value)}
        return _trim_scalar(value)
    if isinstance(value, dict):
        trimmed: Dict[str, Any] = {}
        for index, (key, nested) in enumerate(value.items()):
            if index >= 4:
                break
            trimmed[key] = _summarize_value(nested, depth + 1)
        return trimmed
    if isinstance(value, list):
        if not value:
            return []
        return [_summarize_value(value[0], depth + 1)]
    return _trim_scalar(value)


def _trim_scalar(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 60:
        return value[:57] + "..."
    return value


@dataclass(slots=True)
class StageExecutionResult:
    """Container describing the outcome of a single stage."""

    context: StageContext
    milestone: WorkflowMilestone
    evidence: List[EvidenceReference] = field(default_factory=list)
    summary: Optional[ValidationSummary] = None


# ToolHandle and ToolSet protocols are imported from alpha_vantage_tools module



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
        self.toolset = toolset or YFinanceToolSet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_stage(self, stage: str, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        stage_key = stage.lower()
        context = self._clone_context(context)
        self._append_workflow_log(context, f"Stage {stage_key} started.")
        try:
            if stage_key == "plan_generation":
                result = self._run_plan_generation(request, context)
            elif stage_key == "data_collection":
                result = self._run_data_collection(request, context)
            elif stage_key == "hybrid_analysis":
                result = self._run_hybrid_analysis(request, context)
            elif stage_key == "detailed_analysis":
                result = self._run_detailed_analysis(request, context)
            elif stage_key == "report_generation":
                result = self._run_report_generation(request, context)
            elif stage_key == "delivery":
                result = self._run_delivery(request, context)
            else:
                raise ValueError(f"Unknown validation stage '{stage_key}'")
        except Exception as exc:
            try:
                failure_message = str(exc).replace("\n", " ")
                self._append_workflow_log(context, f"Stage {stage_key} failed: {failure_message}")
            except Exception:  # pragma: no cover - defensive
                pass
            raise
        self._append_workflow_log(result.context, f"Stage {stage_key} completed successfully.")
        return result

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------
    def _run_plan_generation(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        workflow_id = self._ensure_workflow_id(context)
        plan = self._build_validation_plan(request)
        plan_location = self.artifact_store.write_json(workflow_id, "validation_plan", plan).resolve()
        plan_path = str(plan_location)
        context["plan"] = plan
        context.setdefault("insights", []).append(
            f"Planning stage selected {len(plan['data_fetch_tools'])} data tools and {len(plan['analysis_plan'])} analysis steps."
        )
        metadata = context.setdefault("metadata", {})
        plan_metadata = metadata.setdefault("plan", {})
        plan_metadata["artifact"] = plan_path
        plan_metadata["artifact_uri"] = plan_location.as_uri()
        data_tool_count = len(plan.get("data_fetch_tools", []))
        analysis_step_count = len(plan.get("analysis_plan", []))
        self._append_workflow_log(
            context,
            (
                "Stage plan_generation completed; plan artifact {path} produced with "
                "{data_tools} data tool(s) and {analysis_steps} analysis step(s)."
            ).format(
                path=plan_path,
                data_tools=data_tool_count,
                analysis_steps=analysis_step_count,
            ),
        )
        milestone = WorkflowMilestone(
            name="plan_generation",
            status=MilestoneStatus.COMPLETED,
            detail=self._format_plan_detail(plan),
        )
        evidence = [EvidenceReference(type="validation_plan", uri=plan_location.as_uri())]
        self._record_evidence(context, evidence[0])
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    def _run_data_collection(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        """Fetch fundamental and market data from Yahoo Finance for evaluation."""
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
            
            # Save raw response for easier LLM consumption
            artifact_name = f"data_{index:02d}_{slug.lower()}"
            path = self.artifact_store.write_json(workflow_id, artifact_name, response)
            path_resolved = path.resolve()
            path_uri = path_resolved.as_uri()
            executed.append(
                {
                    "slug": slug,
                    "arguments": arguments,
                    "artifact": str(path_resolved),
                    "uri": path_uri,
                    "description": tool.get("description"),
                }
            )
            evidence_ref = EvidenceReference(type="data_fetch", uri=path_uri)
            evidence.append(evidence_ref)
            self._record_evidence(context, evidence_ref)

        context["data_sources"] = {
            item["slug"]: {
                "artifact_path": item["artifact"],
                "artifact_uri": item["uri"],
                "arguments": item["arguments"],
                "description": item.get("description"),
            }
            for item in executed
        }
        context.setdefault("insights", []).append(
           "Data Collection Stage - Fetches fundamental and market data from Yahoo Finance (yfinance)."
        )
        self._append_workflow_log(
            context,
            (
                "Stage data_collection completed; executed {count} tool(s) and saved artifacts to workflow {workflow}."
            ).format(
                count=len(executed),
                workflow=workflow_id,
            ),
        )
        milestone = WorkflowMilestone(
            name="data_collection",
            status=MilestoneStatus.COMPLETED,
            detail=self._format_data_collection_detail(executed),
        )
        self._maybe_add_workflow_log_evidence(context, evidence)
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

        analysis_result, attempt_history, log_uri = self._execute_llm_analysis(
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
            "log_uri": log_uri,
        }
        metrics_path = self.artifact_store.write_json(workflow_id, "analysis_metrics", analysis_payload).resolve().as_uri()
        evidence: List[EvidenceReference] = []
        metrics_evidence = EvidenceReference(type="analysis_metrics", uri=metrics_path)
        evidence.append(metrics_evidence)
        self._record_evidence(context, metrics_evidence)
        if log_uri:
            attempt_evidence = EvidenceReference(type="analysis_attempt_log", uri=log_uri)
            evidence.append(attempt_evidence)
            self._record_evidence(context, attempt_evidence)
        attempt_count = len(attempt_history)
        log_message = (
            "Stage hybrid_analysis completed after {attempts} attempt(s); metrics stored at {metrics}"
        ).format(attempts=attempt_count, metrics=metrics_path)
        if log_uri:
            log_message += f"; attempt log captured at {log_uri}"
        self._append_workflow_log(context, log_message + ".")
        self._maybe_add_workflow_log_evidence(context, evidence)
        context["analysis_results"] = analysis_payload
        if log_uri:
            context["analysis_attempt_log"] = log_uri
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
        self._append_workflow_log(
            context,
            (
                "Stage detailed_analysis completed; narrative generated with {length} character(s)."
            ).format(length=len(narrative)),
        )
        milestone = WorkflowMilestone(
            name="detailed_analysis",
            status=MilestoneStatus.COMPLETED,
            detail="Analysis Stage - Detailed Narrative (LLM) using the collected Yahoo Finance data.",
        )
        evidence: List[EvidenceReference] = []
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

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
            fallback_chart = self._generate_fallback_chart(workflow_id, analysis_results)
            if fallback_chart:
                charts = [fallback_chart]
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
        self._append_workflow_log(
            context,
            (
                "Stage report_generation completed; PDF report saved at {pdf} with conclusion score {score} and confidence {confidence}."
            ).format(pdf=pdf_path, score=score, confidence=confidence),
        )
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
        evidence = [pdf_evidence]
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence, summary=summary)

    def _run_delivery(self, request: HypothesisRequest, context: StageContext) -> StageExecutionResult:
        # Delivery stage now finalises metadata so the portal can expose the downloadable report.
        report = context.get("report") or {}
        pdf_path = report.get("pdf_path")
        payload = report.get("payload") or {}
        if not pdf_path or not payload:
            raise RuntimeError("Report assets missing; run report_generation stage first.")

        context.setdefault("insights", []).append("Report ready for download via the RAVEN portal.")
        self._append_workflow_log(
            context,
            "Stage delivery completed; report published for download.",
        )
        milestone = WorkflowMilestone(
            name="delivery",
            status=MilestoneStatus.COMPLETED,
            detail="Report available for download.",
        )
        evidence: List[EvidenceReference] = []
        self._maybe_add_workflow_log_evidence(context, evidence)
        return StageExecutionResult(context=context, milestone=milestone, evidence=evidence)

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _ensure_workflow_id(self, context: StageContext) -> str:
        metadata = context.get("metadata") or {}
        workflow_id = metadata.get("workflow_id")
        if not workflow_id:
            raise RuntimeError("Workflow metadata must include a workflow_id")
        return str(workflow_id)

    def _append_workflow_log(self, context: StageContext, message: str) -> str:
        workflow_id = self._ensure_workflow_id(context)
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message.rstrip()}\n"
        path = self.artifact_store.append_text(workflow_id, "workflow.log", line)
        log_uri = path.resolve().as_uri()
        metadata = context.setdefault("metadata", {})
        log_info = metadata.setdefault("workflow_log", {})
        log_info["artifact"] = log_uri
        context["workflow_log_uri"] = log_uri
        return log_uri

    def _maybe_add_workflow_log_evidence(
        self, context: StageContext, evidence: List[EvidenceReference]
    ) -> None:
        metadata = context.setdefault("metadata", {})
        log_info = metadata.setdefault("workflow_log", {})
        log_uri = log_info.get("artifact")
        if not log_uri or log_info.get("evidence_recorded"):
            return
        reference = EvidenceReference(type="workflow_log", uri=log_uri)
        evidence.append(reference)
        self._record_evidence(context, reference)
        log_info["evidence_recorded"] = True

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
            enriched_arguments = dict(arguments)
            tools.append(
                {
                    "slug": slug,
                    "description": definition.description,
                    "arguments": enriched_arguments,
                    "inputs": definition.inputs,
                }
            )

        add_tool("YFINANCE_HISTORICAL_PRICES", {"symbol": symbol, "period": "1y"})
        add_tool("YFINANCE_COMPANY_INFO", {"symbol": symbol})
        add_tool("YFINANCE_NEWS", {"symbol": symbol})

        profitability_keywords = {"margin", "cash flow", "cashflow", "operating", "profit", "free cash"}
        if any(keyword in text for keyword in profitability_keywords):
            add_tool("YFINANCE_CASH_FLOW", {"symbol": symbol})
            add_tool("YFINANCE_BALANCE_SHEET", {"symbol": symbol})

        if "earnings" in text or "eps" in text:
            add_tool("YFINANCE_EARNINGS", {"symbol": symbol})

        if "recommendation" in text or "analyst" in text or "rating" in text:
            add_tool("YFINANCE_RECOMMENDATIONS", {"symbol": symbol})

        if "dividend" in text or "yield" in text:
            add_tool("YFINANCE_DIVIDENDS", {"symbol": symbol})

        if "holder" in text or "institutional" in text or "insider" in text:
            add_tool("YFINANCE_HOLDERS", {"symbol": symbol})

        if "financials" in text or "income" in text or "revenue" in text:
            add_tool("YFINANCE_FINANCIALS", {"symbol": symbol})

        if "split" in text:
            add_tool("YFINANCE_SPLITS", {"symbol": symbol})

        return tools

    def _build_analysis_plan(
        self,
        request: HypothesisRequest,
        symbol: str,
        selected_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tool_slugs = {tool["slug"] for tool in selected_tools}
        plan: List[Dict[str, Any]] = []

        if "YFINANCE_HISTORICAL_PRICES" in tool_slugs:
            plan.append(
                {
                    "name": "price_trend_analysis",
                    "description": f"Evaluate trailing price momentum and volatility for {symbol} using historical adjusted closes.",
                    "source_tools": ["YFINANCE_HISTORICAL_PRICES"],
                    "operations": [
                        {
                            "operation": "percent_change",
                            "field": "close",
                            "periods": 252,
                            "label": "Trailing 1y price change",
                        },
                        {
                            "operation": "volatility",
                            "field": "close",
                            "periods": 252,
                            "label": "Annualized volatility",
                        },
                    ],
                }
            )

        if "YFINANCE_COMPANY_INFO" in tool_slugs:
            plan.append(
                {
                    "name": "profitability_snapshot",
                    "description": "Extract current profitability ratios from company info.",
                    "source_tools": ["YFINANCE_COMPANY_INFO"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "operatingMargins",
                            "label": "Operating margin",
                        },
                        {
                            "operation": "latest_value",
                            "field": "profitMargins",
                            "label": "Net profit margin",
                        },
                    ],
                }
            )

        if "YFINANCE_CASH_FLOW" in tool_slugs:
            plan.append(
                {
                    "name": "cash_flow_trend",
                    "description": "Assess annual operating cash flow momentum.",
                    "source_tools": ["YFINANCE_CASH_FLOW"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "Operating Cash Flow",
                            "label": "Operating cash flow (latest)",
                        }
                    ],
                }
            )

        if "YFINANCE_BALANCE_SHEET" in tool_slugs:
            plan.append(
                {
                    "name": "balance_sheet_health",
                    "description": "Review leverage and liquidity from latest balance sheet.",
                    "source_tools": ["YFINANCE_BALANCE_SHEET"],
                    "operations": [
                        {
                            "operation": "latest_value",
                            "field": "Total Assets",
                            "label": "Total assets (latest)",
                        },
                        {
                            "operation": "latest_value",
                            "field": "Total Liabilities",
                            "label": "Total liabilities (latest)",
                        },
                    ],
                }
            )

        if "YFINANCE_NEWS" in tool_slugs:
            plan.append(
                {
                    "name": "news_summary",
                    "description": "Review recent news headlines and publishers.",
                    "source_tools": ["YFINANCE_NEWS"],
                    "operations": [
                        {
                            "operation": "list_titles",
                            "field": "news",
                            "label": "Recent news headlines",
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
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        artifact_map: Dict[str, str] = {}
        for slug, metadata in data_sources.items():
            uri = metadata.get("artifact_path") or metadata.get("uri")
            if not uri:
                raise RuntimeError(f"Artifact path missing for tool '{slug}'")
            path = Path(uri).resolve()
            if not path.exists():
                raise RuntimeError(f"Artifact path for tool '{slug}' does not exist: {path}")
            artifact_map[slug] = str(path)

        history_records: List[Dict[str, Any]] = []
        prompt_history: List[Dict[str, str]] = []
        log_entries: List[str] = []
        log_uri: Optional[str] = None
        data_format = get_data_format(artifact_map)

        def _persist_log() -> str:
            nonlocal log_uri
            log_text = "\n\n".join(log_entries) if log_entries else "No attempts recorded."
            path = self.artifact_store.write_text(
                workflow_id,
                "hybrid_analysis_attempts.log",
                log_text,
            )
            log_uri = path.resolve().as_uri()
            return log_uri

        for attempt in range(1, MAX_REPL_ATTEMPTS + 1):
            response = self.llm.generate_analysis_code(
                request=request,
                analysis_plan=analysis_plan,
                data_artifacts=artifact_map,
                data_format=data_format,
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

            result_section = json.dumps(result_payload, indent=2) if result_payload is not None else "None"
            feedback_section = feedback or "None"
            entry_lines = [
                f"Attempt {attempt} - {record_status.upper()}",
                "=" * 80,
                "Generated code:",
                code.rstrip() or "<empty>",
                "",
                "STDOUT:",
                (stdout_text.strip() or "<none>"),
                "",
                "STDERR:",
                (stderr_text.strip() or "<none>"),
                "",
                "Feedback:",
                feedback_section.strip() or "<none>",
                "",
                "Result payload:",
                result_section,
            ]
            log_entries.append("\n".join(entry_lines))
            _persist_log()

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
                return result_payload, history_records, log_uri or _persist_log()

        _persist_log()
        raise RuntimeError("LLM analysis failed to produce a valid result after maximum retries")

    def _run_analysis_code(
        self,
        *,
        code: str,
        artifact_map: Dict[str, str],
        workflow_id: str,
    ) -> tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
        stdout_text = ""
        stderr_text = ""
        result_payload: Optional[Dict[str, Any]] = None
        feedback: Optional[str] = None

        preamble = self._prepare_analysis_environment(workflow_id, artifact_map)
        sandbox = PythonSandbox()

        try:
            sandbox.run(preamble)
        except Exception as exc:
            error_message = f"Failed to initialise analysis runtime: {exc}"
            feedback = error_message
            return "", error_message, None, feedback

        try:
            stdout_text = sandbox.run(code)
        except Exception as exc:
            stderr_text = str(exc)
            feedback = stderr_text or "Python sandbox execution raised an exception"

        result_line: Optional[str] = None
        for line in stdout_text.splitlines():
            if line.startswith(RESULT_PREFIX):
                result_line = line[len(RESULT_PREFIX) :].strip()
                break

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

    def _prepare_analysis_environment(self, workflow_id: str, artifact_map: Dict[str, str]) -> str:
        workflow_dir = self.artifact_store.root / workflow_id.replace("/", "_")
        workflow_dir.mkdir(parents=True, exist_ok=True)
        artifact_map_repr = repr(artifact_map)
        workflow_dir_str = str(workflow_dir.resolve())
        preamble = textwrap.dedent(
            f"""
            import json
            import math
            import statistics
            from pathlib import Path

            DATA_ARTIFACTS = {artifact_map_repr}
            ARTIFACT_OUTPUT_DIR = Path({workflow_dir_str!r})
            RESULT_PREFIX = {RESULT_PREFIX!r}

            def load_json_artifact(slug: str):
                if slug not in DATA_ARTIFACTS:
                    raise KeyError(f"Unknown data artifact slug: {{slug}}")
                path = Path(DATA_ARTIFACTS[slug])
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            def write_json_artifact(name: str, payload):
                target = ARTIFACT_OUTPUT_DIR / f"{{name}}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return str(target.resolve())
            """
        )
        return preamble

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

    def _generate_fallback_chart(self, workflow_id: str, analysis_results: Dict[str, Any]) -> str | None:
        steps = analysis_results.get("steps", [])
        metrics = []
        for step in steps:
            for output in step.get("outputs", []):
                label = output.get("label")
                value = output.get("value")
                if isinstance(value, (int, float)) and label:
                    metrics.append((label, float(value)))
        if not metrics:
            return None
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

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=25 * mm,
            bottomMargin=20 * mm,
        )

        # Define colors
        primary_color = HexColor("#1a1a2e")
        accent_color = HexColor("#0f3460")
        highlight_color = HexColor("#e94560")

        # Define styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=primary_color,
            spaceAfter=6,
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=accent_color,
            alignment=TA_CENTER,
            spaceAfter=20,
        ))
        styles.add(ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=primary_color,
            spaceBefore=16,
            spaceAfter=8,
            borderPadding=(0, 0, 4, 0),
        ))
        styles.add(ParagraphStyle(
            name="RavenBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            name="BulletItem",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            leftIndent=12,
            spaceAfter=3,
        ))
        styles.add(ParagraphStyle(
            name="MetricLabel",
            parent=styles["Normal"],
            fontSize=9,
            textColor=accent_color,
        ))
        styles.add(ParagraphStyle(
            name="MetricValue",
            parent=styles["Normal"],
            fontSize=12,
            textColor=primary_color,
        ))

        story = []

        # --- Header ---
        story.append(Paragraph("RAVEN", styles["ReportTitle"]))
        story.append(Paragraph("Hypothesis Validation Report", styles["ReportSubtitle"]))
        story.append(Spacer(1, 10))

        # --- Hypothesis Info Card ---
        hypothesis_text = request.hypothesis_text or "(No hypothesis text)"
        info_data = [
            [Paragraph("<b>Hypothesis:</b>", styles["RavenBody"]), Paragraph(hypothesis_text, styles["RavenBody"])],
            [Paragraph("<b>Workflow ID:</b>", styles["RavenBody"]), Paragraph(workflow_id, styles["RavenBody"])],
            [Paragraph("<b>Entities:</b>", styles["RavenBody"]), Paragraph(", ".join(request.entities) or "N/A", styles["RavenBody"])],
        ]
        info_table = Table(info_data, colWidths=[80, 400])
        info_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 16))

        # --- Key Metrics Summary Card ---
        aggregated = analysis_results.get("aggregated", {})
        if aggregated:
            story.append(Paragraph("Key Metrics at a Glance", styles["SectionHeading"]))
            metric_items = []
            for key, value in list(aggregated.items())[:6]:  # Show top 6 metrics
                display_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    display_value = f"{value:.4f}" if abs(value) < 1 else f"{value:,.2f}"
                else:
                    display_value = str(value) if value is not None else "N/A"
                metric_items.append([
                    Paragraph(display_key, styles["MetricLabel"]),
                    Paragraph(display_value, styles["MetricValue"]),
                ])
            if metric_items:
                metrics_table = Table(metric_items, colWidths=[200, 280])
                metrics_table.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.5, HexColor("#e0e0e0")),
                ]))
                story.append(metrics_table)
            story.append(Spacer(1, 12))

        # --- Executive Summary ---
        exec_summary = report_payload.get("executive_summary")
        if exec_summary:
            story.append(Paragraph("Executive Summary", styles["SectionHeading"]))
            story.append(Paragraph(str(exec_summary), styles["RavenBody"]))

        # --- Key Findings ---
        key_findings = report_payload.get("key_findings", [])
        if key_findings:
            story.append(Paragraph("Key Findings", styles["SectionHeading"]))
            for finding in key_findings:
                story.append(Paragraph(f"• {finding}", styles["BulletItem"]))

        # --- Risks ---
        risks = report_payload.get("risks", [])
        if risks:
            story.append(Paragraph("Risks", styles["SectionHeading"]))
            for risk in risks:
                story.append(Paragraph(f"• {risk}", styles["BulletItem"]))

        # --- Next Steps ---
        next_steps = report_payload.get("next_steps", [])
        if next_steps:
            story.append(Paragraph("Next Steps", styles["SectionHeading"]))
            for step in next_steps:
                story.append(Paragraph(f"• {step}", styles["BulletItem"]))

        # --- Narrative Summary ---
        narrative = str(detailed_analysis.get("narrative", ""))
        if narrative:
            story.append(Paragraph("Detailed Narrative", styles["SectionHeading"]))
            story.append(Paragraph(narrative, styles["RavenBody"]))

        # --- Plan Overview ---
        if plan_steps:
            story.append(Paragraph("Validation Plan Overview", styles["SectionHeading"]))
            for idx, step in enumerate(plan_steps[:8], start=1):  # Limit to 8 steps for readability
                # Truncate long steps
                display_step = step if len(step) < 100 else step[:97] + "..."
                story.append(Paragraph(f"{idx}. {display_step}", styles["BulletItem"]))

        # --- Footer ---
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"<i>Generated by RAVEN • {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>",
            ParagraphStyle(name="Footer", parent=styles["Normal"], fontSize=8, textColor=accent_color, alignment=TA_CENTER),
        ))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        pdf_path = self.artifact_store.write_bytes(workflow_id, "validation_report.pdf", pdf_bytes).resolve().as_uri()
        return pdf_path

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
        return f"Alpha Vantage data ingested from: {slugs}."

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
