# RAVEN — Hypothesis Validation Copilot

RAVEN is a LangGraph-powered copilot that helps equity analysts validate investment hypotheses fast. Paste a thesis and the agent assembles an end-to-end diligence loop: structured planning, data collection through Composio-integrated sources, hybrid LLM analytics, narrative synthesis, and a polished report published for instant download. The goal is to collapse a multi-hour research sprint into a guided interactive session.

https://github.com/user-attachments/assets/demo-placeholder

## Why It Matters
- **Actionable research in minutes**: Automates diligence steps that usually span analysts, data engineers, and PMs.
- **Audit-ready output**: Every chart, metric, and insight is backed by stored evidence and downloadable PDFs.
- **Human-in-the-loop**: Optional review gates let analysts approve or redirect the workflow before decisions are recorded.

## System Highlights
- 🔁 **LangGraph Orchestration** – deterministic workflow stages with retry-aware LLM REPL execution.
- 🧠 **LLM Hybrid Analytics** – prompt-safe Python execution, RESULT-validated outputs, and chart generation.
- 🔌 **Composio Tooling** – catalog-driven data pulls (Alpha Vantage, SEC) with full artifact capture.
- ☁️ **Firebase Persistence** – Firestore backs submissions and status tracking; no migrations to babysit.
- 🖥️ **FastAPI + Sleek UI** – submit hypotheses, watch milestones progress, and download final reports in one screen.

## Architecture Overview
```
[FastAPI / UI]
	│
	▼
[Hypothesis Service] ─────→ [Firestore]
	│
	▼
[LangGraphValidationOrchestrator]
  ├─ Planning Stage - Plan Generation (LLM) 
  ├─ Data Collection Stage - Data Collection (Composio)
  ├─ Analysis Stage - Hybrid Analysis (LLM REPL)
  ├─ Analysis Stage - Detailed Narrative (LLM)
  ├─ Report Rendering (ReportLab)
  └─ Delivery (Portal publish)
```

## Getting Started

### Requirements
- Python 3.12+
- Poetry or virtualenv of your choice
- Firebase project with service account JSON
- OpenAI API key (or compatible Azure endpoint)
- Alpha Vantage + SEC credentials (for live data pulls)

### Install & Configure
```bash
git clone https://github.com/ayushsi42/raven.git
cd raven
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Required environment variables
set -a
source .env
set +a
```

## Run It Locally

### Dependencies
- Python 3.12+
- `pip` / virtualenv (or Poetry)
- Firebase project (service account JSON for Firestore access)
- OpenAI API key (or compatible Azure endpoint) for LLM calls
- Alpha Vantage API key & SEC user agent for market data
- Composio account (user ID + API key) to broker third-party tool executions

### Installation
```bash
git clone https://github.com/ayushsi42/raven.git
cd raven
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Environment Variables
Create a `.env` file in the project root (or export variables manually). At minimum:

```bash
RAVEN_OPENAI_API_KEY="sk-..."
RAVEN_OPENAI_MODEL="gpt-4o-mini"             # optional override
RAVEN_ALPHA_VANTAGE_API_KEY="your-av-key"
RAVEN_SEC_USER_AGENT="Your Firm Contact"
RAVEN_FIREBASE_CREDENTIALS_PATH="/path/to/service-account.json"
RAVEN_FIREBASE_PROJECT_ID="your-firebase-project"
RAVEN_COMPOSIO_USER_ID="your-composio-user"
RAVEN_COMPOSIO_API_KEY="cpso_..."             # or set COMPOSIO_API_KEY
RAVEN_REQUIRE_AUTHENTICATION=false            # flip to true for Firebase auth gate
```

Load the configuration when developing:

```bash
set -a
source .env
set +a
```

### Start the API & UI
```bash
PYTHONPATH=src uvicorn hypothesis_agent.main:app --reload
```

Visit `http://localhost:8000/` for the landing page. The REST API is exposed beneath `/v1` (see cheat sheet below).

### Run Tests
```bash
PYTHONPATH=src pytest
```

The suite uses lightweight Firestore fakes, so no emulator is required for unit tests.

## Connect Composio Tool Router
RAVEN leans on Composio to orchestrate external data pulls and notifications.

1. Sign in to the Composio dashboard and create an API key plus user (or reuse the default).
2. Share those credentials with the agent by setting either `RAVEN_COMPOSIO_API_KEY`/`RAVEN_COMPOSIO_USER_ID` or the generic `COMPOSIO_API_KEY` environment variable before launching the app.
3. During demos, testers can paste temporary keys into the `.env` file or export them in the terminal; the UI will automatically surface data pulled through Composio once the credentials are present.
4. To add additional tool integrations, update `src/hypothesis_agent/orchestration/tool_catalog.py` and re-run the workflow.

Without credentials, the delivery stage will still produce the downloadable PDF, but upstream data fetches may fall back to stub responses.
2. Launch shared dashboards for portfolio teams.
3. Add cost-aware execution plans for budget-constrained analyses.

## License & Attribution
This project was produced during Hacktualization 2025 by Team RAVEN. Usage is internal-only unless granted explicit permission.