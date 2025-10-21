# RAVEN — Hypothesis Validation Copilot

RAVEN is a LangGraph-powered validation copilot built during Hacktualization 2025. Drop in an investment hypothesis and the agent orchestrates a full diligence loop: planning, data collection via Composio tools, hybrid LLM analysis, narrative synthesis, and polished delivery straight to your inbox.

https://github.com/user-attachments/assets/demo-placeholder

## Why It Matters
- **Actionable research in minutes**: Automates equity research workflows that normally take hours.
- **Audit-ready output**: Every chart, metric, and insight is backed by stored evidence and downloadable PDFs.
- **Human-in-the-loop**: Optional review gates let analysts approve or redirect before reports ship.

## System Highlights
- 🔁 **LangGraph Orchestration** – deterministic workflow stages with retry-aware LLM REPL execution.
- 🧠 **LLM Hybrid Analytics** – prompt-safe Python execution, RESULT-validated outputs, and chart generation.
- 🔌 **Composio Tooling** – catalog-driven data pulls (Alpha Vantage, SEC, Gmail) with full artifact capture.
- ☁️ **Firebase Persistence** – Firestore backs submissions and status tracking; no migrations to babysit.
- 🖥️ **FastAPI + Sleek UI** – submit hypotheses, watch milestones progress, download reports, email delivery in one screen.

## Architecture Overview
```
[FastAPI / UI]
	│
	▼
[Hypothesis Service] ─────→ [Firestore]
	│
	▼
[LangGraphValidationOrchestrator]
  ├─ Plan Generation (LLM)
  ├─ Data Collection (Composio)
  ├─ Hybrid Analysis (LLM REPL)
  ├─ Detailed Narrative (LLM)
  ├─ Report Rendering (ReportLab)
  └─ Delivery (Composio Gmail)
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

### Run the Stack
```bash
PYTHONPATH="src" uvicorn hypothesis_agent.main:app --reload
```

Navigate to `http://localhost:8000/` for the live UI or hit the API directly under `/v1`.

### Smoke Tests
```bash
PYTHONPATH=src pytest
```

Unit tests rely on lightweight Firestore fakes—no emulator required.

## API Cheat Sheet
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `POST` | `/v1/hypotheses` | Submit a new hypothesis |
| `GET` | `/v1/hypotheses/{id}` | Retrieve stored submission |
| `GET` | `/v1/hypotheses/{id}/status` | Live workflow status |
| `GET` | `/v1/hypotheses/{id}/report` | Final validation summary |
| `POST` | `/v1/hypotheses/{id}/resume` | Resolve human review gates |
| `POST` | `/v1/hypotheses/{id}/cancel` | Cancel an in-flight workflow |

All routes honor the optional `x-api-key` header defined in `AppSettings`.

## Development Notes
- LLM interaction lives in `src/hypothesis_agent/llm.py` with abstract hooks for swapping providers.
- Workflow orchestration code: `src/hypothesis_agent/orchestration/langgraph_pipeline.py`.
- Firestore wiring and repository layer: `src/hypothesis_agent/db/firebase.py` and `.../repositories/hypothesis_repository.py`.
- UI template: `src/hypothesis_agent/api/templates/landing.html`.

## Roadmap After Hackathon
1. Expand tool catalog to onboard additional market data sources.
2. Launch shared dashboards for portfolio teams.
3. Add cost-aware execution plans for budget-constrained analyses.

## License & Attribution
This project was produced during Hacktualization 2025 by Team RAVEN. Usage is internal-only unless granted explicit permission.