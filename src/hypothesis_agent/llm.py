"""LLM integration helpers for the RAVEN hypothesis workflow."""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import BadRequestError, OpenAI
logger = logging.getLogger(__name__)

from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.metrics import LLMTelemetry, PIPELINE_TELEMETRY


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an unexpected payload."""


class BaseLLM(ABC):
    """Interface defining the planner / report generation contract."""

    @abstractmethod
    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:  # pragma: no cover - interface
        """Return an ordered list of data collection tasks."""

    @abstractmethod
    def generate_analysis_plan(
        self,
        request: HypothesisRequest,
        data_overview: Dict[str, Any],
    ) -> List[str]:  # pragma: no cover - interface
        """Return an ordered list of analysis steps given the available data."""

    @abstractmethod
    def generate_detailed_analysis(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
    ) -> str:  # pragma: no cover - interface
        """Return a narrative describing hypothesis validation results."""

    @abstractmethod
    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
        analysis_summary: str,
        artifact_paths: List[str],
    ) -> Dict[str, Any]:  # pragma: no cover - interface
        """Return a structured report payload summarising the findings."""

    @abstractmethod
    def generate_analysis_code(
        self,
        *,
        request: HypothesisRequest,
        analysis_plan: List[Dict[str, Any]],
        data_artifacts: Dict[str, str],
        attempt: int,
        history: List[Dict[str, str]],
    ) -> str:  # pragma: no cover - interface
        """Return Python source for executing the requested analysis."""


@dataclass(slots=True)
class OpenAILLM(BaseLLM):
    """LLM adapter backed by the OpenAI Responses API."""

    api_key: str
    model: str
    temperature: float | None = 0.2
    telemetry: LLMTelemetry | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise LLMError("OpenAI API key is not configured")
        self._client = OpenAI(api_key=self.api_key)
        if self.telemetry is None:
            self.telemetry = PIPELINE_TELEMETRY

    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:
        system_prompt = (
            "You are a financial research operations planner. Generate a concise list of data collection "
            "tasks needed to validate an investment hypothesis. Respond with a JSON array of strings."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nEntities: {entities}\nTime horizon: {start} to {end}\n"
            "Include market, fundamental, macro, and narrative data collection tasks.".format(
                hypothesis=request.hypothesis_text,
                entities=", ".join(request.entities) or "(none specified)",
                start=request.time_horizon.start.isoformat(),
                end=request.time_horizon.end.isoformat(),
            )
        )
        content = self._chat(system_prompt, user_prompt, operation="generate_data_plan")
        return self._parse_list(content)

    def generate_analysis_plan(
        self,
        request: HypothesisRequest,
        data_overview: Dict[str, Any],
    ) -> List[str]:
        system_prompt = (
            "You are a senior quantitative analyst. Based on the available data inventory, produce an ordered list "
            "of analytical steps to validate the hypothesis. Respond with a JSON array of strings."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nRisk appetite: {risk}\nData overview: {overview}".format(
                hypothesis=request.hypothesis_text,
                risk=request.risk_appetite.value,
                overview=json.dumps(data_overview, ensure_ascii=False),
            )
        )
        content = self._chat(system_prompt, user_prompt, operation="generate_analysis_plan")
        return self._parse_list(content)

    def generate_detailed_analysis(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
    ) -> str:
        system_prompt = (
            "You are an equity research associate. Write a concise narrative (max 6 sentences) summarising the current "
            "state of the hypothesis given the quantitative metrics and sentiment signals. Avoid repeated figures."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nMetrics overview: {metrics}".format(
                hypothesis=request.hypothesis_text,
                metrics=json.dumps(metrics_overview, ensure_ascii=False),
            )
        )
        return self._chat(system_prompt, user_prompt, operation="generate_detailed_analysis")

    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
        analysis_summary: str,
        artifact_paths: List[str],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are an analyst preparing a client-facing investment memo. Respond with a JSON object containing the "
            "keys 'executive_summary', 'key_findings' (array), 'risks' (array), and 'next_steps' (array)."
        )
        user_prompt = (
            "Hypothesis: {hypothesis}\nMetrics: {metrics}\nNarrative summary: {summary}\nArtifacts: {artifacts}".format(
                hypothesis=request.hypothesis_text,
                metrics=json.dumps(metrics_overview, ensure_ascii=False),
                summary=analysis_summary,
                artifacts=artifact_paths,
            )
        )
        content = self._chat(system_prompt, user_prompt, operation="generate_report")
        payload = self._parse_json(content)
        if not isinstance(payload, dict):
            raise LLMError("Report generation returned non-dict response")
        return payload

    def generate_analysis_code(
        self,
        *,
        request: HypothesisRequest,
        analysis_plan: List[Dict[str, Any]],
        data_artifacts: Dict[str, str],
        attempt: int,
        history: List[Dict[str, str]],
    ) -> str:
        system_prompt = (
            "You are an autonomous quantitative analyst operating in a constrained Python REPL. "
            "Write straight-line, top-level Python (no function or class definitions) that inspects provided datasets, "
            "computes the requested analytics, and prints the final dictionary using "
            "print(\"RESULT::\" + json.dumps(result, default=str)). Respond only with Python code inside a "
            "fenced block; do not include commentary."
        )
        user_payload = {
            "hypothesis": request.hypothesis_text,
            "attempt": attempt,
            "analysis_plan": analysis_plan,
            "data_artifacts": data_artifacts,
            "execution_history": history,
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
        return self._chat(system_prompt, user_prompt, operation="generate_analysis_code")

    def _chat(self, system_prompt: str, user_prompt: str, *, operation: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            start_time = time.perf_counter()
            response = self._client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - start_time
        except BadRequestError as exc:
            message = str(exc)
            if self.temperature is not None and "temperature" in message.lower():
                logger.warning(
                    "OpenAI model %s rejected temperature=%s; retrying with default",
                    self.model,
                    self.temperature,
                )
                self.temperature = None
                kwargs.pop("temperature", None)
                try:
                    start_time = time.perf_counter()
                    response = self._client.chat.completions.create(**kwargs)
                    latency = time.perf_counter() - start_time
                except Exception as retry_exc:  # pragma: no cover - network failures
                    raise LLMError("OpenAI request failed") from retry_exc
            else:
                raise LLMError("OpenAI request failed") from exc
        except Exception as exc:  # pragma: no cover - network failures
            raise LLMError("OpenAI request failed") from exc

        try:
            message = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
            raise LLMError("Malformed OpenAI response") from exc

        if self.telemetry is not None:
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            self.telemetry.record_llm_call(
                operation=operation,
                model=self.model,
                latency_seconds=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        return message.strip()

    @staticmethod
    def _parse_json(content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM response was not valid JSON") from exc

    def _parse_list(self, content: str) -> List[str]:
        parsed = self._parse_json(content)
        if not isinstance(parsed, list) or not parsed:
            raise LLMError("LLM response must be a non-empty list of steps")
        if not all(isinstance(item, str) and item.strip() for item in parsed):
            raise LLMError("LLM response list entries must be non-empty strings")
        return parsed
