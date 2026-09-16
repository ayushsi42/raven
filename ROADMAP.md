# Roadmap — RAVEN

## Current State
- The pipeline works end-to-end (plan → data collection → hybrid analysis → narrative → PDF report → delivery) when given a valid `OPENAI_API_KEY`, and 18 pytest tests pass with the LLM and YFinance layers stubbed out.
- Despite the names in the code, there is no LangGraph `StateGraph` and no Temporal deployment in the live path — `LangGraphValidationOrchestrator` is a hand-written `if/elif` stage dispatcher, and the Temporal-era `workflows/definitions.py` / `workflows/activities/` modules are dead code kept alive only by one test file.
- `PythonSandbox` runs LLM-generated code with `exec()` against the real Python `__builtins__` (not the `SAFE_BUILTINS` allowlist defined in `langgraph_pipeline.py`, which appears unused in the actual execution path) — there is no real sandboxing, so a malformed or adversarial LLM completion can run arbitrary code with full process privileges.
- Hypothesis records are held in `InMemoryHypothesisRepository`; a process restart silently discards all submission/status history even though on-disk artifacts (plans, data, charts, PDFs) survive.
- OpenAI is the only LLM provider, wired directly into `langgraph_pipeline.py` and `llm.py` — there's no provider abstraction beyond the already-defined `BaseLLM` interface being used for exactly one implementation.
- There is no CI workflow, no linter enforcement in CI (ruff is declared as a dev dependency but not run automatically), and no test coverage for the `yfinance` network calls or the sandboxed-code execution path itself.

## Phase 1 — Near-term (weeks)
- Remove or clearly quarantine the dead Temporal-era code (`workflows/definitions.py`, `workflows/activities/`) — either delete it or move it under a `legacy/` package with a README note, and update `tests/test_workflow_activities.py` accordingly so the test suite reflects only what's live.
- Rename `LangGraphValidationOrchestrator` (or actually port it onto a real `langgraph.graph.StateGraph`) so the class name matches its implementation — right now it's misleading to anyone reading the code cold.
- Make `PythonSandbox` actually restrict execution: pass a restricted globals dict (reuse the existing `SAFE_BUILTINS` from `langgraph_pipeline.py`, which is currently defined but never plumbed into `PythonSandbox.__init__`), disallow `import`/`open`/`__import__`, and add a wall-clock timeout around `exec()` so a runaway LLM-generated loop can't hang a stage indefinitely.
- Add retry/backoff around the `yfinance` calls in `yfinance_tools.py` — Yahoo Finance rate-limits aggressively and the current handlers have no retry, timeout, or graceful-degradation path if a ticker lookup fails mid-pipeline.
- Wire up a GitHub Actions workflow that runs `PYTHONPATH=src pytest -q` and `ruff check` on every push/PR — there's currently no CI at all, so regressions only surface locally.
- Swap `datetime.utcnow()` in `langgraph_pipeline.py` for a timezone-aware equivalent and fix the Starlette `TemplateResponse` deprecated argument order in `api/ui.py` — both currently emit `DeprecationWarning`s in the test run and will break on a future dependency bump.

## Phase 2 — Medium-term
- Replace `InMemoryHypothesisRepository` with a persistent store (SQLite via SQLAlchemy would fit the existing `ABC`-based `HypothesisRepository` interface with minimal churn) so hypothesis history survives restarts.
- Introduce an LLM provider abstraction beyond the unused-in-practice `BaseLLM` interface: add at least one second implementation (e.g. an Anthropic or local-model adapter) and make the choice configurable via `AppSettings`, removing the current hard OpenAI lock-in.
- Build an evaluation harness for report quality — currently there is no automated way to judge whether a generated executive summary, key findings, or risk list is actually good; even a small rubric-based LLM-as-judge or a golden-hypothesis regression set would let changes to prompts be validated instead of eyeballed.
- Add integration tests that exercise the real `yfinance` calls (marked `@pytest.mark.integration`, skipped by default, run on a schedule) so data-layer regressions are caught before they hit a live pipeline run.
- Expose workflow/report history and human-review actions in the web UI beyond the current single-page flow, and add basic auth so `/v1/hypotheses` isn't fully open when deployed.

## Phase 3 — Stretch
- Multi-tenant deployment: real authentication/authorization per `user_id`, persistent storage, and object storage (S3-compatible) for artifacts instead of local disk, so RAVEN can run as a shared service rather than a single-user local process.
- Pluggable data sources beyond Yahoo Finance (SEC filings, earnings call transcripts, alternative data) behind the existing `ToolSet`/`ToolHandle` protocol, so the planning stage can draw from a richer, provider-agnostic tool catalog.
- A proper async task/workflow engine (Celery, Temporal done correctly this time, or a real LangGraph graph) if concurrent multi-user load ever requires more than the current in-process `asyncio.create_task` model.

## Success Metrics
- **Phase 1**: CI is green on every PR; a fuzzed/malformed LLM code completion can no longer escape `PythonSandbox` (demonstrated by a test that tries and fails to read/write outside the sandbox); a simulated Yahoo Finance rate-limit no longer crashes a live pipeline run.
- **Phase 2**: hypothesis history survives a process restart (demonstrable via a restart-and-fetch test); at least two LLM providers can complete the full pipeline against the same hypothesis with comparable output structure; an evaluation run produces a numeric/qualitative score for report quality that can be diffed across prompt changes.
- **Phase 3**: a second concurrent user can submit and track a hypothesis independently without state collisions; at least one non-Yahoo-Finance data source is live in the planning stage's tool catalog.
