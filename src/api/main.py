"""
FastAPI backend for the Move Desk dashboard.

    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /health                    -> liveness check
    GET  /resolve?query=...         -> validates arbitrary input ("HDFC Bank",
                                        "tcs", "AAPL") into a real ticker BEFORE
                                        committing to a full analysis run
    GET  /explain?query=...         -> resolves + runs the full live pipeline
                                        for ANY stock (src/agents/orchestrator.py
                                        explain_move_live)
    GET  /explain_demo?ticker=...   -> the original curated-universe synthetic
                                        path, kept for demoing offline / when
                                        live data isn't reachable

`/explain` is intentionally two steps under the hood (resolve, then explain)
even though it's one HTTP call, so a failed resolution returns a fast, clear
404 with a helpful message instead of only failing deep inside a slow live
data fetch.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.universe import all_tickers, sector_of, UNIVERSE, peers_of
from src.data.resolve_ticker import resolve_ticker
from src.data.synthetic_factors import generate_factor_world
from src.data.synthetic import generate_fundamentals
from src.features.pipeline import build_feature_table
from src.agents.orchestrator import explain_move, explain_move_live
from src.agents.evidence_types import AdjudicationResult
from src.agents.thesis_writer import write_narrative_offline
from src.api.schemas import ExplainResponse, ResolveResponse, HealthResponse
from src.monitoring.store import ExplanationStore
from src.monitoring.graph_tracker import MonitoringGraph

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="FinSight Move Desk API")

store = ExplanationStore()
monitoring_graph = MonitoringGraph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten before any real deployment
    allow_methods=["GET"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(FRONTEND_DIR / "dashboard.html"))

# --- Demo-mode world (unchanged from before, kept for offline/demo use) ----
_sector_map = {t: sector_of(t) for t in all_tickers()}
_DEMO_WORLD = generate_factor_world(list(_sector_map.keys()), _sector_map, n_days=500, seed=23)
_DEMO_FEATURE_TABLES = {}


def _demo_series(ticker: str):
    sector = _sector_map[ticker]
    dates = _DEMO_WORLD["dates"]
    stock_data = _DEMO_WORLD["stocks"][ticker]
    stock_returns = pd.Series(stock_data["returns"], index=dates)
    market_returns = pd.Series(_DEMO_WORLD["market_returns"], index=dates)
    sector_returns = pd.Series(_DEMO_WORLD["sector_returns"][sector], index=dates)
    return stock_data, stock_returns, market_returns, sector_returns


def _demo_feature_table(ticker: str) -> pd.DataFrame:
    if ticker not in _DEMO_FEATURE_TABLES:
        stock_data, _, _, _ = _demo_series(ticker)
        dates = _DEMO_WORLD["dates"]
        prices = stock_data["prices"]
        ohlcv = pd.DataFrame({
            "Open": prices * 0.998, "High": prices * 1.01, "Low": prices * 0.99, "Close": prices,
            "Volume": (2_000_000 * (1 + 5 * pd.Series(stock_data["returns"]).abs())).astype(int).values,
        }, index=dates)
        _DEMO_FEATURE_TABLES[ticker] = build_feature_table(ohlcv, generate_fundamentals(ticker))
    return _DEMO_FEATURE_TABLES[ticker]


def _demo_most_significant_window(ticker: str, lookback_days: int = 90, window_len: int = 5):
    from src.features.factor_model import find_most_significant_window
    _, stock_returns, market_returns, sector_returns = _demo_series(ticker)
    return find_most_significant_window(ticker, stock_returns, market_returns, sector_returns,
                                         _DEMO_WORLD["dates"], lookback_days=lookback_days, window_len=window_len)


def _resolve_source_path(evidence, ticker: str) -> str:
    if getattr(evidence, "source_path", None):
        return evidence.source_path
    if getattr(evidence, "source_url", None):
        return evidence.source_url
    tag = evidence.tags[0] if getattr(evidence, "tags", None) else "event"
    return f"{evidence.source_agent}://{ticker}#{tag}"


def _adjudication_to_response(r: AdjudicationResult) -> dict:
    m = r.move
    accepted = [v for v in r.verdicts if v.accepted]
    rejected = [v for v in r.verdicts if not v.accepted]
    return {
        "ticker": m.ticker, "start_date": str(m.start_date), "end_date": str(m.end_date),
        "raw_return": m.raw_return, "market_beta": m.market_beta, "sector_beta": m.sector_beta,
        "market_explained_return": m.market_explained_return,
        "sector_explained_return": m.sector_explained_return,
        "residual_return": m.residual_return, "residual_zscore": m.residual_zscore,
        "systematic_fraction": m.systematic_fraction, "has_sector_factor": m.has_sector_factor,
        "confidence": r.confidence, "unexplained": r.unexplained,
        "corroboration_count": r.corroboration_count, "confidence_breakdown": r.confidence_breakdown,
        "accepted_evidence": [
            {
                "source_agent": v.evidence.source_agent,
                "event_date": str(v.evidence.event_date),
                "direction": v.evidence.direction.value,
                "magnitude_hint": v.evidence.magnitude_hint,
                "raw_text": v.evidence.raw_text,
                "source_url": v.evidence.source_url,
                "source_path": _resolve_source_path(v.evidence, m.ticker),
                "contribution": v.contribution,
                "temporal_alignment": v.temporal_alignment,
            }
            for v in accepted
        ],
        "rejected_evidence": [
            {
                "source_agent": v.evidence.source_agent,
                "event_date": str(v.evidence.event_date),
                "raw_text": v.evidence.raw_text,
                "rejection_reason": v.rejection_reason,
                "source_url": v.evidence.source_url,
                "source_path": _resolve_source_path(v.evidence, m.ticker),
                "contribution": 0.0,
                "temporal_alignment": v.temporal_alignment,
            }
            for v in rejected
        ],
        "narrative": write_narrative_offline(r),
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/universe")
def universe():
    """Suggested tickers for quick-pick UI — no longer a hard restriction on what /explain accepts."""
    return {"sectors": {sector: [t for t, _ in stocks] for sector, stocks in UNIVERSE.items()}}


@app.get("/resolve", response_model=ResolveResponse)
def resolve(query: str = Query(..., description="Company name or ticker, e.g. 'HDFC Bank' or 'AAPL'")):
    """
    Validates arbitrary input WITHOUT running the full (slower) analysis —
    lets the frontend show "Resolved to: HDFC Bank Ltd (HDFCBANK.NS)" and
    let the person confirm before committing to a live data pull.
    """
    result = resolve_ticker(query)
    if not result.resolved:
        raise HTTPException(status_code=404, detail=result.error)
    return {
        "query": result.query, "ticker": result.ticker, "company_name": result.company_name,
        "exchange": result.exchange, "resolution_method": result.resolution_method,
        "alternate_candidates": result.alternate_candidates,
    }


@app.get("/explain", response_model=ExplainResponse)
def explain(
    query: str = Query(..., description="Any company name or ticker — resolved automatically"),
    start: str | None = Query(None, description="YYYY-MM-DD; omit to auto-select the most significant recent move"),
    end: str | None = Query(None, description="YYYY-MM-DD; required if start is given"),
):
    """
    The general "any possible stock" endpoint. Resolves `query` first (fast,
    fails clearly if it can't), then runs the LIVE pipeline — real yfinance
    data, real news, real factor decomposition with market-only fallback
    when no sector index applies.
    """
    resolution = resolve_ticker(query)
    if not resolution.resolved:
        raise HTTPException(status_code=404, detail=resolution.error)

    window_start = pd.Timestamp(start) if start else None
    window_end = pd.Timestamp(end) if end else None

    try:
        report = explain_move_live(resolution.ticker, window_start=window_start, window_end=window_end)
    except Exception as e:
        # Live data unreachable, ticker too new/thin for the beta lookback, etc.
        # Surface a clear, actionable error instead of a raw 500 traceback.
        raise HTTPException(
            status_code=502,
            detail=f"Resolved '{query}' to {resolution.ticker}, but the live analysis failed: {e}",
        )

    response = _adjudication_to_response(report.adjudication)
    response["resolved_from_query"] = query
    response["resolution_method"] = resolution.resolution_method

    # Persist in SQLite store and update monitoring graph
    try:
        move_id = store.save_run(report.adjudication, narrative=response["narrative"])
        monitoring_graph.ingest_run(report.adjudication, run_id=move_id, narrative=response["narrative"])
    except Exception as err:
        print(f"[api] Error saving run to persistence store: {err}")

    return response


@app.get("/explain_demo", response_model=ExplainResponse)
def explain_demo(
    ticker: str = Query(..., description="One of the curated demo tickers, e.g. HDFCBANK.NS"),
    start: str | None = Query(None),
    end: str | None = Query(None),
):
    """Original synthetic-data path — kept so the dashboard still has a zero-setup demo mode."""
    if ticker not in _sector_map:
        raise HTTPException(status_code=404, detail=f"{ticker} is not in the demo universe. See /universe.")

    if start and end:
        window_start, window_end = pd.Timestamp(start), pd.Timestamp(end)
    else:
        window_start, window_end = _demo_most_significant_window(ticker)

    stock_data, stock_returns, market_returns, sector_returns = _demo_series(ticker)
    feature_table = _demo_feature_table(ticker)
    report = explain_move(ticker, stock_returns, market_returns, sector_returns, feature_table,
                           window_start=window_start, window_end=window_end, injected_events=stock_data["events"])
    response = _adjudication_to_response(report.adjudication)

    # Persist in SQLite store and update monitoring graph
    try:
        move_id = store.save_run(report.adjudication, narrative=response["narrative"])
        monitoring_graph.ingest_run(report.adjudication, run_id=move_id, narrative=response["narrative"])
    except Exception as err:
        print(f"[api] Error saving demo run: {err}")

    return response


@app.get("/history")
def history(ticker: str = Query(..., description="Ticker to retrieve historical runs for")):
    """Retrieves previous explanation runs for a stock from the persistence layer."""
    runs = store.get_runs_for_ticker(ticker)
    return {"ticker": ticker, "total_runs": len(runs), "runs": runs}


@app.get("/contradictions")
def contradictions(ticker: str | None = Query(None, description="Optional ticker filter")):
    """Retrieves moves where earlier accepted evidence was subsequently contradicted by new evidence."""
    return {"contradicted_moves": monitoring_graph.get_contradicted_moves(ticker=ticker)}


@app.get("/peers")
def peers(ticker: str = Query(..., description="Ticker to retrieve peer group for")):
    """Returns sector peers for comparison mode."""
    sector = sector_of(ticker)
    peer_list = peers_of(ticker)
    return {"ticker": ticker, "sector": sector, "peers": peer_list}

