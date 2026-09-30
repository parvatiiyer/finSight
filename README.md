# FinSight — Quantitative Equity Research & Causal Attribution Engine

FinSight is a research-desk tool designed to answer one narrow question well: **"Why did this stock move, and how sure are we?"**

Unlike unconstrained LLMs that hallucinate explanations from nearby headlines, FinSight employs factor decomposition to isolate idiosyncratic return variance and enforces a deterministic 3-gate adjudicator before evidence can be accepted.

---

## Quickstart

### 1. Local Development (Fastest)

```bash
# Clone the repository
git clone <repo-url>
cd finSight

# Install dependencies (virtual environment recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server (serves both API & Dashboard UI)
python3 -m uvicorn src.api.main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### 2. Docker Deployment (Recommended for Cloud/Production)

```bash
# Build and run with Docker Compose
docker compose up --build
```
The application will be live at `http://localhost:8000`.

---

### 3. Deploying to Free/Cloud Hosting (Render / Railway)

1. Push your repository to GitHub.
2. In **[Render](https://render.com)** or **[Railway](https://railway.app)**:
   - Create a new **Web Service**.
   - Select your GitHub repo.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
3. Visiting your assigned URL (e.g. `https://finsight.onrender.com`) will automatically serve the full interactive UI and API.

---

## Running the Automated Test Suite

Run all 33 unit, property, and contract tests:

```bash
PYTHONPATH=. pytest tests/
```

---

## Architecture Overview

1. **Ticker Resolution (`src/data/resolve_ticker.py`)**: Multi-layered search (exact ticker, fuzzy name, yfinance live resolution).
2. **Factor Decomposition (`src/features/factor_model.py`)**: Rolling OLS multi-factor model (market + sector) with cross-calendar/timezone alignment (`_align_returns`).
3. **Multi-Agent Evidence Proposals (`src/agents/`)**:
   - `news_agent`: Google News RSS + FinBERT sentiment analysis.
   - `fundamentals_agent`: Quarter-over-quarter ratio deltas + earnings proximity.
   - `technical_agent`: Bollinger Band breakouts, RSI extremes, volume spikes, and Isolation Forest anomaly detection.
4. **Deterministic Adjudicator (`src/agents/adjudicator.py`)**:
   - Temporal Alignment Gate ($Score \ge 0.4$)
   - Magnitude Adequacy Gate ($Hint \ge 0.30$)
   - Directional Consistency Gate (Residual sign match)
   - Auditable confidence formula with independent source-type corroboration bonuses.
5. **Monitoring Loop & Persistence (`src/monitoring/`)**:
   - SQLite run persistence (`store.py`).
   - NetworkX graph engine (`graph_tracker.py`) tracking contradiction edges and source reliability.
