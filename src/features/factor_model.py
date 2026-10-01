"""
Factor model: decomposes a stock's return over a window into

    raw_return = market_explained + sector_explained + residual

NEW in this version: sector_returns is now OPTIONAL. The curated 25-stock
universe always has a mapped sector index (see config/universe.py), but an
arbitrary ticker typed into the dashboard might not — a company yfinance
tags as, say, "Industrials" or "Real Estate" has no NSE sector-index analogy
in our SECTOR_INDICES map. Rather than refuse to analyze it, or silently
pretend a sector factor exists, the model degrades to a single-factor CAPM
regression against the market alone, and sets sector_beta / sector_explained
to exactly 0.0 with has_sector_factor=False on the resulting MoveWindow, so
every downstream consumer (adjudicator, dashboard) can be honest about a
weaker decomposition rather than hiding it.

Betas are still estimated on a TRAILING window ending before the move
window starts — using the move window's own data to estimate the beta that
then "explains" that same window would be circular.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agents.evidence_types import MoveWindow


def _normalize_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """tz-strip + floor-to-midnight, so a start/end date can be compared against _align_returns' output."""
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _align_returns(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: pd.Series | None,
) -> pd.DataFrame:
    """
    Aligns stock/market/(sector) return Series onto a single, timezone-naive,
    calendar-date index.

    A stock and its benchmark index frequently trade on different calendars
    — different countries' market holidays, or (as with AAPL vs the NSEI
    Nifty 50 used as this project's market factor) entirely different
    exchanges and timezones. Two return series over "the same" 2 years can
    have a different NUMBER of trading days (500 vs 494 was the case that
    surfaced this), and their raw DatetimeIndex objects aren't directly
    comparable (-04:00 vs +05:30).

    Every place that combines these Series — estimating betas, selecting a
    move window, or computing trailing residual volatility — must operate on
    values that actually correspond to the same calendar date. This is the
    ONE place that alignment happens; both estimate_betas() and
    decompose_move() call it rather than each doing their own (previously
    inconsistent) version of it, which is what let this bug hide in
    estimate_betas() after decompose_move()'s window-selection was fixed but
    estimate_betas() wasn't.
    """
    def _normalize_index(s: pd.Series) -> pd.Series:
        s = s.copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        return s[~s.index.duplicated(keep="last")]

    data = {"stock": _normalize_index(stock_returns), "market": _normalize_index(market_returns)}
    if sector_returns is not None:
        data["sector"] = _normalize_index(sector_returns)

    aligned = pd.DataFrame(data).dropna()
    if aligned.empty:
        raise ValueError(
            "Stock and market/sector return series share no overlapping trading dates after "
            "calendar alignment — check both cover the same date range and were fetched correctly."
        )
    return aligned


def estimate_betas(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: pd.Series | None,
    as_of: pd.Timestamp,
    lookback: int = 120,
) -> tuple[float, float, float]:
    """
    OLS-fits alpha, beta_mkt, and (if sector_returns is given) beta_sector on
    the `lookback` ALIGNED trading days strictly BEFORE `as_of`. When
    sector_returns is None, fits a single-factor market model instead and
    returns beta_sector=0.0.

    Previously sliced stock/market/sector independently with .iloc using
    integer positions derived from stock_returns' own index — silently wrong
    whenever the three Series don't share an identical calendar, since
    position N in one Series and position N in another don't correspond to
    the same date. Now takes the trailing window from the jointly-aligned
    frame, so every row used in the regression is guaranteed to be the same
    calendar date across stock/market/sector.
    """
    aligned = _align_returns(stock_returns, market_returns, sector_returns)
    as_of_norm = _normalize_timestamp(as_of)

    trailing = aligned.loc[aligned.index < as_of_norm].tail(lookback)
    if len(trailing) < 30 and "sector" in aligned.columns:
        # Fall back to single-factor market alignment if sector history has gaps
        try:
            mkt_aligned = _align_returns(stock_returns, market_returns, None)
            mkt_trailing = mkt_aligned.loc[mkt_aligned.index < as_of_norm].tail(lookback)
            if len(mkt_trailing) >= 30:
                aligned = mkt_aligned
                trailing = mkt_trailing
        except Exception:
            pass

    if len(trailing) < 30:
        raise ValueError(
            f"Not enough ALIGNED trailing history before {as_of_norm.date()} to estimate betas "
            f"(need >= 30 overlapping trading days, found {len(trailing)}). "
            f"Please choose an analysis window from 2015-01-01 onwards where continuous trading data is available."
        )

    y = trailing["stock"].values
    x_mkt = trailing["market"].values

    if "sector" in trailing.columns:
        x_sector = trailing["sector"].values
        X = np.column_stack([np.ones_like(x_mkt), x_mkt, x_sector])
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta_mkt, beta_sector = coeffs
    else:
        X = np.column_stack([np.ones_like(x_mkt), x_mkt])
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta_mkt = coeffs
        beta_sector = 0.0

    return float(alpha), float(beta_mkt), float(beta_sector)


def decompose_move(
    ticker: str,
    stock_returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: pd.Series | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    beta_lookback: int = 120,
    residual_vol_lookback: int = 60,
) -> MoveWindow:
    """
    Decompose the stock's cumulative return over [start_date, end_date].
    `sector_returns=None` runs a market-only decomposition — the correct,
    explicit behavior for a ticker with no mapped sector index, rather than
    an error or a silent zero-filled sector series that would look identical
    to "this stock genuinely has zero sector exposure" (a different, false
    claim).
    """
    has_sector_factor = sector_returns is not None
    alpha, beta_mkt, beta_sector = estimate_betas(
        stock_returns, market_returns, sector_returns, as_of=start_date, lookback=beta_lookback
    )

    # Same alignment as estimate_betas() — deliberately re-derived from the
    # shared helper rather than passed through, so decompose_move() stays
    # correct even when called on its own (as it is directly in tests and
    # in ad-hoc diagnostics) without relying on a caller to have aligned
    # anything first.
    aligned = _align_returns(stock_returns, market_returns, sector_returns)
    start_norm = _normalize_timestamp(start_date)
    end_norm = _normalize_timestamp(end_date)

    window = aligned.loc[(aligned.index >= start_norm) & (aligned.index <= end_norm)]
    if window.empty:
        raise ValueError(
            f"No overlapping trading days found for {ticker} between {start_date} and {end_date} "
            f"after aligning stock/market{'/sector' if has_sector_factor else ''} calendars"
        )

    stock_window = window["stock"]
    mkt_window = window["market"]
    sector_window = window["sector"] if has_sector_factor else None

    raw_return = float(np.exp(stock_window.sum()) - 1)
    market_explained = float(np.exp(beta_mkt * mkt_window.sum()) - 1)
    sector_explained = float(np.exp(beta_sector * sector_window.sum()) - 1) if has_sector_factor else 0.0

    sector_component = beta_sector * sector_window.sum() if has_sector_factor else 0.0
    residual_log = stock_window.sum() - beta_mkt * mkt_window.sum() - sector_component
    residual_return = float(np.exp(residual_log) - 1)

    trailing = aligned.loc[aligned.index < start_norm].tail(residual_vol_lookback)
    trailing_residuals = trailing["stock"].values - beta_mkt * trailing["market"].values
    if has_sector_factor:
        trailing_residuals = trailing_residuals - beta_sector * trailing["sector"].values

    residual_daily_vol = float(np.std(trailing_residuals)) if len(trailing_residuals) > 0 else 0.0
    horizon_days = len(stock_window)
    residual_window_vol = residual_daily_vol * np.sqrt(max(horizon_days, 1))
    residual_zscore = residual_log / residual_window_vol if residual_window_vol > 0 else 0.0

    return MoveWindow(
        ticker=ticker,
        start_date=start_date.date() if hasattr(start_date, "date") else start_date,
        end_date=end_date.date() if hasattr(end_date, "date") else end_date,
        raw_return=raw_return,
        market_beta=beta_mkt,
        sector_beta=beta_sector,
        market_explained_return=market_explained,
        sector_explained_return=sector_explained,
        residual_return=residual_return,
        residual_zscore=float(residual_zscore),
        has_sector_factor=has_sector_factor,
    )


def find_most_significant_window(
    ticker: str,
    stock_returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: pd.Series | None,
    dates: pd.DatetimeIndex,
    lookback_days: int = 90,
    window_len: int = 5,
    stride: int = 2,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Scans the last `lookback_days` for the `window_len`-day span with the
    largest |residual z-score|. Shared by the API and the CLI/orchestrator
    so "what should I even be looking at" logic lives in exactly one place
    instead of being copy-pasted between callers.
    """
    candidates = dates[-lookback_days:-window_len]
    best_window, best_score = None, -1.0
    errors_seen: dict[str, int] = {}
    for start in candidates[::stride]:
        end = start + pd.Timedelta(days=window_len - 1)
        try:
            move = decompose_move(ticker, stock_returns, market_returns, sector_returns, start, end)
        except ValueError as e:
            # Previously a bare `continue` here silently discarded the real
            # reason EVERY candidate failed (this is exactly what hid the
            # estimate_betas() alignment bug — every window raised the same
            # ValueError, and the loop just reported "nothing found" with no
            # way to tell why). Now the distinct failure reasons are tallied
            # and surfaced if nothing ever scores.
            errors_seen[str(e)] = errors_seen.get(str(e), 0) + 1
            continue
        if abs(move.residual_zscore) > best_score:
            best_score, best_window = abs(move.residual_zscore), (start, end)
    if best_window is None:
        if errors_seen:
            detail = "; ".join(f"{msg} (x{count})" for msg, count in errors_seen.items())
            raise ValueError(
                f"Could not find a scoreable window for {ticker} in the last {lookback_days} days — "
                f"every candidate window failed: {detail}"
            )
        raise ValueError(
            f"Could not find a scoreable window for {ticker} in the last {lookback_days} days "
            f"(no candidate windows were attempted — check `dates` covers enough history)."
        )
    return best_window

def _validate_returns(name: str, ticker: str, returns: pd.Series) -> None:
    if returns.empty:
        raise ValueError(
            f"{name} return series for {ticker!r} is empty after fetch + log-return "
            f"computation — the OHLCV fetch likely returned no usable 'Close' data "
            f"(check for an empty/malformed DataFrame, e.g. unexpected MultiIndex columns)."
        )
    span = (returns.index.min(), returns.index.max())
    if len(returns) < 30:
        raise ValueError(
            f"{name} return series for {ticker!r} only has {len(returns)} observations "
            f"(range {span[0].date()} to {span[1].date()}) — too thin to be a real fetch failure vs. a "
            f"genuinely new/thinly-traded listing; inspect the raw fetch."
        )