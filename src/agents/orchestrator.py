"""
Orchestrator for the move-explanation pipeline.

Two entry points now:

  explain_move(...)       — the original synthetic-data path, used for the
                             curated 25-stock demo universe and for testing
                             the adjudicator against known ground truth.

  explain_move_live(...)  — NEW: the "any possible stock" path. Takes a raw
                             ticker, resolves it (src/data/resolve_ticker.py),
                             fetches real data (src/data/ingest.py), figures
                             out its sector or falls back to market-only
                             (src/features/factor_model.py), and gathers real
                             news evidence (src/agents/news_agent.py's
                             production path) instead of synthetic events.

Both converge on the same adjudicate() call and the same MoveExplanationReport
shape — the "any stock" capability is additive, not a fork of the pipeline.
"""

from concurrent import futures
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.universe import sector_of, sector_index_of, sector_index_from_yf_sector, MARKET_INDEX
from src.features.factor_model import decompose_move, find_most_significant_window
from src.features.pipeline import build_feature_table
from src.agents.news_agent import gather_news_evidence_synthetic, gather_news_evidence_production
from src.agents.fundamentals_agent import gather_fundamentals_evidence
from src.agents.technical_agent import gather_technical_evidence
from src.agents.filing_agent import gather_filing_evidence
from src.agents.adjudicator import adjudicate
from src.agents.evidence_types import AdjudicationResult


@dataclass
class MoveExplanationReport:
    adjudication: AdjudicationResult
    narrative: str | None = None


def explain_move(ticker, stock_returns, market_returns, sector_returns, feature_table,
                  window_start, window_end, injected_events=None, current_fundamentals=None,
                  prior_fundamentals=None, earnings_dates=None, narrate=False):
    """Synthetic-data path — see module docstring."""
    move = decompose_move(ticker, stock_returns, market_returns, sector_returns,
                           start_date=window_start, end_date=window_end)
    
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        if injected_events is not None:
            futures["news"] = pool.submit(gather_news_evidence_synthetic, ticker, injected_events,
                                           window_start, window_end, feature_table.index)
        if current_fundamentals is not None:
            futures["fundamentals"] = pool.submit(gather_fundamentals_evidence, ticker, window_start, window_end,
                                        current_fundamentals, prior_fundamentals, earnings_dates)
        futures["technical"] = pool.submit(gather_technical_evidence, ticker, feature_table, window_start, window_end)
        futures["filing"] = pool.submit(gather_filing_evidence, ticker, window_start, window_end, move.residual_zscore)

        all_evidence = []
        for name, fut in futures.items():
            try:
                all_evidence.extend(fut.result())
            except Exception as e:
                print(f"[orchestrator] {name} agent failed: {e}")

    result = adjudicate(move, all_evidence)
    narrative = None
    if narrate:
        from src.agents.thesis_writer import write_narrative
        narrative = write_narrative(result)
    return MoveExplanationReport(adjudication=result, narrative=narrative)


def _validate_returns(name: str, ticker: str, returns: pd.Series, min_obs: int = 30) -> None:
    """
    Fail fast, at the source, with a specific message — instead of letting a
    bad/thin/empty return series travel silently through _to_log_returns(),
    decompose_move(), and _align_returns() and surface 40+ calls later as
    the generic "no overlapping trading dates" error. That generic error is
    still correct when it fires for a genuine calendar-mismatch reason (as
    with AAPL vs NSEI), but it was also the ONLY signal for completely
    unrelated causes upstream (a bad reindex, a wrong resolved ticker, a
    thin/stale cached fetch) — which is what made each of those look like a
    new, ticker-specific bug instead of the same class of problem.
    """
    if returns.empty:
        raise ValueError(
            f"{name} return series for {ticker!r} is empty after fetch + log-return computation — "
            f"the OHLCV fetch returned no usable 'Close' data (check for an empty or malformed "
            f"DataFrame from the ingest layer)."
        )
    if len(returns) < min_obs:
        raise ValueError(
            f"{name} return series for {ticker!r} only has {len(returns)} observations "
            f"(range {returns.index.min().date()} to {returns.index.max().date()}) — too thin to be "
            f"a legitimate fetch for the requested period. This usually means a partial/rate-limited "
            f"fetch got cached as if it succeeded, not a real data gap."
        )


def explain_move_live(
    ticker: str,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    lookback_days: int = 90,
    window_len: int = 5,
    period: str = "2y",
) -> MoveExplanationReport:
    """
    The "any possible stock" entry point. `ticker` must already be a
    RESOLVED, validated yfinance symbol (see src/data/resolve_ticker.py) —
    resolution is a separate, explicit step on purpose, so a caller (like
    the API) can show the person what it resolved their input to before
    committing to the slower live-data fetch and evidence gathering.

    If window_start/window_end aren't given, auto-selects the most
    significant recent move via find_most_significant_window — the same
    "what should I even be looking at" behavior as the demo path.
    """
    from src.data import ingest

    MIN_SUPPORTED_DATE = pd.Timestamp("2015-01-01")
    if window_start is not None:
        ws_norm = window_start.tz_localize(None) if getattr(window_start, "tz", None) else window_start
        if ws_norm < MIN_SUPPORTED_DATE:
            raise ValueError(
                f"Requested start date {ws_norm.date()} is prior to 2015-01-01. "
                "FinSight supports historical moves from 2015 onwards to ensure reliable multi-factor market and sector alignment."
            )
        # If looking at earlier years (e.g. 2016-2023), pull period="max" so historical dates are available
        now_dt = pd.Timestamp.now()
        if (now_dt - ws_norm).days > 700:
            period = "max"

    ohlcv = ingest.fetch_ohlcv(ticker, period=period)
    market_ohlcv = ingest.fetch_ohlcv(MARKET_INDEX, period=period)
    market_returns = _to_log_returns(market_ohlcv)
    stock_returns = _to_log_returns(ohlcv)

    # Fail fast and specifically here, at the source of each series, rather
    # than letting a bad fetch travel five layers down into a generic
    # "no overlapping trading dates" error that could mean anything.
    _validate_returns("Stock", ticker, stock_returns)
    _validate_returns("Market", MARKET_INDEX, market_returns)

    # Sector factor: try the curated mapping first (covers our 25 tickers
    # exactly as before), then fall back to yfinance's reported sector for
    # anything else. Either way, if no sector index can be determined, the
    # factor model runs market-only rather than erroring out.
    sector_index = sector_index_of(ticker)
    if sector_index is None:
        yf_sector = ingest.fetch_company_sector(ticker)
        sector_index = sector_index_from_yf_sector(yf_sector)

    sector_returns = None
    if sector_index is not None:
        try:
            sector_ohlcv = ingest.fetch_ohlcv(sector_index, period=period)
            # Passed through on its OWN native calendar/timezone, deliberately
            # NOT reindexed onto stock_returns.index here — _align_returns()
            # in factor_model.py is the one place alignment happens, and it
            # normalizes tz/calendar consistently for stock, market, and
            # sector alike. Reindexing here first (as a previous version did)
            # silently produced an all-NaN sector series whenever the sector
            # index's timezone differed from the stock's.
            sector_returns = _to_log_returns(sector_ohlcv)
            _validate_returns("Sector", sector_index, sector_returns)
        except Exception as e:
            print(f"[orchestrator] sector index fetch failed ({sector_index}): {e} — falling back to market-only")
            sector_returns = None

    dates = stock_returns.index
    if window_start is None or window_end is None:
        window_start, window_end = find_most_significant_window(
            ticker, stock_returns, market_returns, sector_returns, dates,
            lookback_days=lookback_days, window_len=window_len,
        )

    move = decompose_move(ticker, stock_returns, market_returns, sector_returns, window_start, window_end)

    fundamentals = ingest.fetch_fundamentals(ticker)
    company_name = fundamentals.get("longName") or fundamentals.get("shortName") or ticker
    feature_table = build_feature_table(ohlcv, fundamentals)

    quarterly_history = ingest.fetch_quarterly_fundamentals_history(ticker)
    prior_fundamentals = quarterly_history[1] if len(quarterly_history) > 1 else None
    earnings_dates = ingest.fetch_earnings_dates(ticker)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "news": pool.submit(gather_news_evidence_production, ticker, company_name, window_start, window_end),
            "fundamentals": pool.submit(gather_fundamentals_evidence, ticker, window_start, window_end,
                             fundamentals, prior_fundamentals, earnings_dates),
            "technical": pool.submit(gather_technical_evidence, ticker, feature_table, window_start, window_end),
            "filing": pool.submit(gather_filing_evidence, ticker, window_start, window_end, move.residual_zscore),
        }
        all_evidence = []
        for name, fut in futures.items():
            try:
                all_evidence.extend(fut.result())
            except Exception as e:
                print(f"[orchestrator] {name} agent failed for {ticker}: {e}")

    result = adjudicate(move, all_evidence)
    return MoveExplanationReport(adjudication=result)


def _to_log_returns(ohlcv: pd.DataFrame) -> pd.Series:
    import numpy as np
    return np.log(ohlcv["Close"] / ohlcv["Close"].shift(1)).dropna()


def print_report(report: MoveExplanationReport) -> None:
    r = report.adjudication
    m = r.move
    print(f"\n{'='*70}\n{m.ticker}: {m.start_date} to {m.end_date}")
    print(f"Raw return: {m.raw_return:+.2%}  |  Residual: {m.residual_return:+.2%} (z={m.residual_zscore:+.2f})")
    print(f"Systematic fraction: {m.systematic_fraction:.0%}  |  Sector factor available: {m.has_sector_factor}")
    print(f"Confidence: {r.confidence:.0%}  |  Unexplained: {r.unexplained}")
    for e in r.accepted_evidence:
        print(f"  ACCEPTED [{e.source_agent}] {e.event_date} {e.raw_text}")
    for v in r.rejected_evidence:
        print(f"  REJECTED [{v.evidence.source_agent}] {v.evidence.raw_text} -- {v.rejection_reason}")
    if report.narrative:
        print(f"\n{report.narrative}")
    print("=" * 70)
