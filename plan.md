# RAVEN — System Design

**Short description:** RAVEN (Risk & Value Evaluation Network) is a multi-agent system that validates user-submitted financial hypotheses by collecting public data (historical financials, market data, news, filings, alternative data), running automated and human-in-the-loop analyses, and returning a detailed, auditable report. The stack uses **LangGraph** for agent orchestration, **Composio** as ToolRouter, and **Temporal** for resumable long-running workflows and checkpointing.

---

## Design goals

* **Accurate validation:** Reproduce thorough financial analysis (numerical, fundamental, and sentiment) using only publicly available data.
* **Reproducibility & auditability:** All steps logged, checkpoints saved, and final report traceable to data and intermediate results.
* **Resumability:** Long-running tasks can be paused/resumed, retried, and manually intervened via Temporal + ToolRouter.
* **Scalable & modular:** Easy to add new data connectors, analysis agents, or export formats.
* **Explainable:** Human-friendly explanations and structured numeric outputs.

---

## High-level architecture

Components (logical):

* **Frontend / UX**: Web app or chat UI for users to submit hypotheses, view progress, review reports, and intervene.
* **API Gateway / Orchestrator**: Receives requests and routes to LangGraph workflows.
* **LangGraph**: Multi-agent orchestration layer — defines agents, conversation/state flows, and composes tool calls.
* **Tool Router (Composio)**: Routes tool/connector calls (market data, filings, news, compute) to appropriate microservices.
* **Temporal Workflows**: Long-running workflows, checkpointing, retries, human-in-the-loop waits.
* **Agent Workers**: Microservices implementing agent logic (Data Ingest, Preprocessor, Financial Analyzer, Sentiment, Backtester, Report Generator).
* **Data Lake / Stores**:

  * Raw ingests (object storage, e.g., S3)
  * Time-series DB for market data (e.g., TimescaleDB)
  * Relational DB for metadata & audit (e.g., Postgres)
  * Vector DB for NLP embeddings (e.g., Milvus, Pinecone)
  * Cache (Redis) for fast lookups and intermediate results
* **Compute Layer**: Batch/stream compute using Kubernetes jobs, serverless functions, or specialized compute instances.
* **Monitoring & Observability**: Logging, traces (OpenTelemetry), metrics (Prometheus), alerting.
* **Auth & Governance**: Role-based access, API keys, data retention, and compliance logs.

---

## Core flow (user story)

1. User submits a hypothesis (natural language or structured): e.g., "Company X's EBITDA margin will expand by 200 bps over next 12 months due to cost savings from Y."
2. API Gateway authenticates and creates a **Temporal workflow** instance.
3. LangGraph initializes agents and hands the workflow token to the ToolRouter (Composio) integration.
4. **Data collection agents** run in parallel:

   * Historical financials & SEC filings (EDGAR connectors)
   * Market price & fundamentals (OHLC, splits, corporate actions)
   * News & press releases (news APIs, RSS, web-scrape)
   * Alternative data (job postings, satellite footfall, Google trends)
5. Each data fetch call goes through Composio which selects the best connector and returns results to the workflow. Results are stored raw in object store and indexed in metadata DB.
6. **Preprocessing agent** cleans, normalizes, maps tickers/entities, bridges different granularities, imputes missing values, and time-aligns series. Embeddings for text sources are created and stored in the vector DB.
7. **Analysis agents** run concurrent modules (numerical & NLP):

   * Ratio & trend analysis (margins, growth, leverage)
   * Forecasting & scenario models (statistical models, simple ARIMA/ETS, ML models, optionally LLM-based simulation)
   * Event & sentiment scoring (news sentiment, filing tone, CEO quotes)
   * Backtests or comparable company analysis
   * Risk & stress testing (sensitivity to macro shocks)
8. **Validator agent** maps analysis outputs to the user hypothesis and computes a validation score (quantitative and qualitative). It also lists assumptions and key drivers.
9. **Report generator** compiles traceable artifacts: data snapshots, model outputs, charts, code snippets, and natural-language explanation.
10. The Temporal workflow finalizes, stores the report, and notifies the user. If human review is required (confidence threshold or flagged items), the workflow pauses for human-in-the-loop via a Temporal activity.

---

## Agent types & responsibilities

* **User Interface Agent** — standardizes inputs (converts NL to a hypothesis schema), initial validation.
* **Data Ingest Agent** — fetches from connectors; deduplicates; tags with provenance.
* **Entity Resolver Agent** — canonicalizes names/tickers, handles delisted tickers or mergers.
* **Preprocessor Agent** — normalization, currency conversions, CPI adjustments, fiscal to calendar conversions.
* **Financial Analyzer Agent** — fundamentals, ratios, trend isolation, accounting adjustments (non-GAAP handling), peer-group selection.
* **Sentiment Agent** — NLP pipelines for news, filings, social; produces sentiment timeseries and event detection.
* **Modeling Agent** — runs forecasting models, Monte Carlo, scenario analysis, and backtests hypotheses.
* **Validator Agent** — compares hypothesis statements to outputs and assigns scores, p-values, and a confidence band.
* **Report Agent** — compiles the final deliverable in multiple formats (HTML, PDF, JSON), includes provenance links.
* **Human Review Agent** — UI hooks for reviewers to modify checkpoints and add notes.

---

## Temporal + Resumable Long-Run pattern

* **Workflow per hypothesis**: Each hypothesis runs as a Temporal workflow with these features:

  * **Checkpoint at milestones**: after raw data ingest, after preprocessing, after model runs, after validation.
  * **Idempotent activities**: activities can be retried; use deterministic Temporal code patterns.
  * **Human-in-the-loop await**: workflow can pause for human approval with a TTL and escalation.
  * **Versioning**: store the workflow & agent version used for reproducibility.
* **Checkpoint schema (example)**:

```json
{
  "workflow_id": "...",
  "milestone": "preprocessed",
  "timestamp": "2025-10-18T12:34:56Z",
  "provenance": {"data_uris": ["s3://..."], "connector_versions": {...}},
  "agent_versions": {"preprocessor": "v1.2.0", "sentiment": "v0.9.1"},
  "state_summary": {"rows": 12345, "missing_pct": 0.02},
  "next_steps": ["run_models"]
}
```

---

## Tool routing (Composio) integration patterns

* **Abstract tool interface**: define canonical tool calls like `fetch_time_series(ticker, start, end)`, `fetch_filings(entity, start, end)`, `search_news(query, start, end)`.
* **Connector registry**: Composio maps canonical calls to one or more concrete connectors with ranking (latency, freshness, cost, reliability).
* **Fallback strategies**: if a primary connector fails, retry with fallback; log reasons for fallback.
* **Rate-limiting & caching**: Composio should respect API limits; cache frequent queries.

---

## Data provenance & audit

* Store raw payloads (S3) with immutable hashes and metadata.
* Maintain lineage linking raw file → processed dataset → model inputs → model outputs → report sections.
* Expose provenance in UI as expandable traces for each assertion in the report.

---

## Example message/contract schemas

* **Hypothesis schema (input)**:

```json
{
  "user_id": "...",
  "hypothesis_text": "Company X's EBITDA margin will expand by 200 bps in next 12 months due to cost savings from Y.",
  "entities": ["Company X"],
  "time_horizon": {"start": "2025-10-01", "end": "2026-10-01"},
  "risk_appetite": "moderate",
  "requires_human_review": false
}
```

* **Validation result (output)** snippet:

```json
{
  "score": 0.21,
  "conclusion": "Partially supported",
  "key_drivers": ["gross_margin_trend", "operating_expenses_reduction"],
  "confidence": 0.65,
  "evidence": [{"type": "financials", "uri": "s3://..."}, {"type": "news", "uri": "s3://..."}]
}
```

---

## Model & algorithm choices (recommendations)

* **Baseline statistical models**: ARIMA/ETS for short-term forecasting; OLS regressions for simple driver analysis.
* **Machine learning**: tree ensembles for feature importance (XGBoost/LightGBM) when labeled outcomes exist.
* **LLM usage**: Use LLMs for hypothesis parsing, qualitative summaries, and scenario generation; do *not* use them as the sole source for numerical assertions without backing data — always link to numeric artifacts.
* **Explainability**: SHAP for feature attribution; show residual diagnostics and backtest curves.

---

## Security, compliance & privacy

* Apply rate-limited, read-only connectors for public data.
* Encrypt data at rest & transit; rotate keys.
* Redact PII from any user-provided content and log access.
* Maintain retention policies for raw and processed data.

---

## Scalability & infra

* **Kubernetes** for worker autoscaling.
* **Temporal** workers scaled per activity type (data-ingest, model-run).
* **Object storage** for raw artifacts (S3-compatible).
* **CDN** for serving generated reports.
* Use spot instances for expensive batch model runs with careful checkpointing.

---

## Observability & testing

* Structured logs with correlation IDs (workflow_id, agent_id).
* Distributed traces for end-to-end latency analysis.
* Synthetic regression tests for each agent (unit tests + integration tests using recorded fixtures).
* Dashboard showing active workflows, checkpoint states, retries, and failure reasons.

---

## UX considerations

* Show live progress with milestone checkpoints and an estimated state (not ETA) like: *Data Ingested -> Preprocessing -> Modeling -> Validation Pending Review*.
* Allow user to pin a hypothesis to re-run with different assumptions.
* Provide downloadable artifacts and JSON for reproducibility.

---

## Roadmap & incremental build plan

1. **MVP**: Accept hypotheses, simple connectors (price + historical financials + news), basic preprocessing, ratio calculations, and a deterministic validator producing JSON + HTML report.
2. **Phase 2**: Add Temporal checkpointing and human-in-the-loop approval; add more connectors (EDGAR, alternative data); introduce Composio routing.
3. **Phase 3**: Add advanced modeling agents (ML, Monte Carlo), explainability (SHAP), vector DB for filings and news similarity, and full audit UI.
4. **Phase 4**: Scale to many concurrent workflows, enterprise connectors, RBAC, and compliance additions.

---

## Appendix: Useful interfaces & sample pseudo-endpoints

* `POST /v1/hypotheses` — submit hypothesis
* `GET /v1/hypotheses/{id}/status` — workflow milestone and checkpoints
* `GET /v1/hypotheses/{id}/report` — downloadable report
* `POST /v1/hypotheses/{id}/resume` — human-approved resume (Temporal signal)

---

If you want, I can now:

* Produce a sequence diagram (SVG) of Temporal + LangGraph + Composio interactions.
* Provide a sample Temporal workflow implementation sketch (TypeScript/Python).
* Draft the Hypothesis JSON schema and database schema for audit logs.

Tell me which artifact you want next.
