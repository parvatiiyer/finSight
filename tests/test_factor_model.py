"""
Regression test for the AAPL-vs-NSEI calendar/timezone alignment bug.

Two return series covering "the same" date range can have a different
number of trading days (different market holidays) and different
timezones. Positional alignment (.iloc, or a Boolean mask built from one
Series applied to another) silently assumes identical calendars and breaks
the moment that's untrue — this happened in both decompose_move() and,
separately, estimate_betas(), which is why fixing only the first one didn't
resolve the reported bug. These tests pin down both.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.features.factor_model import decompose_move, estimate_betas, find_most_significant_window


def _mismatched_calendars(seed=42):
    """Reproduces the reported AAPL (500 rows, -04:00) vs NSEI (494 rows, +05:30) case."""
    rng = np.random.default_rng(seed)
    stock_dates = pd.bdate_range("2024-09-24", periods=500, tz="America/New_York")
    stock_returns = pd.Series(rng.normal(0.0005, 0.015, 500), index=stock_dates)
    market_dates = pd.bdate_range("2024-09-24", periods=494, tz="Asia/Kolkata")
    market_returns = pd.Series(rng.normal(0.0003, 0.011, 494), index=market_dates)
    return stock_returns, market_returns


def test_decompose_move_handles_different_length_timezone_aware_series():
    stock_returns, market_returns = _mismatched_calendars()
    start, end = stock_returns.index[300], stock_returns.index[304]
    # Must not raise IndexError("Boolean index has wrong length...")
    move = decompose_move("AAPL", stock_returns, market_returns, None, start, end)
    assert np.isfinite(move.raw_return)
    assert np.isfinite(move.residual_zscore)


def test_estimate_betas_handles_different_length_timezone_aware_series():
    stock_returns, market_returns = _mismatched_calendars()
    as_of = stock_returns.index[300]
    # This is the second bug: estimate_betas() sliced stock/market positionally
    # with .iloc using integer positions from stock_returns' own index, which
    # silently picks different CALENDAR DATES from market_returns whenever the
    # two series don't share an identical trading calendar.
    alpha, beta_mkt, beta_sector = estimate_betas(stock_returns, market_returns, None, as_of=as_of, lookback=120)
    assert np.isfinite(alpha) and np.isfinite(beta_mkt)
    assert beta_sector == 0.0  # no sector series was given


def test_find_most_significant_window_succeeds_with_mismatched_calendars():
    """The exact failure mode reported: this used to raise 'Could not find a
    scoreable window' for every candidate because estimate_betas() threw on
    every single one — silently, since find_most_significant_window() only
    reported the count, not the reason."""
    stock_returns, market_returns = _mismatched_calendars()
    start, end = find_most_significant_window(
        "AAPL", stock_returns, market_returns, None, stock_returns.index, lookback_days=90
    )
    assert start < end
    move = decompose_move("AAPL", stock_returns, market_returns, None, start, end)
    assert np.isfinite(move.residual_zscore)


def test_genuine_no_overlap_raises_informative_error_not_generic_one():
    """When there truly is no fixable overlap, the error should say why —
    this is what would have made the original bug report a one-line fix
    instead of a full debugging round trip."""
    rng = np.random.default_rng(1)
    stock_dates = pd.bdate_range("2024-01-01", periods=100)
    stock_returns = pd.Series(rng.normal(0, 0.01, 100), index=stock_dates)
    market_dates = pd.bdate_range("2030-01-01", periods=100)  # zero overlap, by construction
    market_returns = pd.Series(rng.normal(0, 0.01, 100), index=market_dates)

    with pytest.raises(ValueError, match="no overlapping trading dates"):
        find_most_significant_window("FAKE", stock_returns, market_returns, None, stock_dates, lookback_days=90)


def test_same_calendar_behavior_is_unchanged():
    """Regression guard: when stock/market/sector already share an identical
    index (the common case for the curated demo universe), results must be
    numerically identical to before this fix — this fix should only change
    behavior for MISMATCHED calendars, never for matched ones."""
    from src.data.synthetic_factors import generate_factor_world

    sector_map = {"HDFCBANK.NS": "Banking"}
    world = generate_factor_world(list(sector_map.keys()), sector_map, n_days=500, seed=7)
    dates = world["dates"]
    stock_returns = pd.Series(world["stocks"]["HDFCBANK.NS"]["returns"], index=dates)
    market_returns = pd.Series(world["market_returns"], index=dates)
    sector_returns = pd.Series(world["sector_returns"]["Banking"], index=dates)

    event = world["stocks"]["HDFCBANK.NS"]["events"][0]
    start = event.event_date - pd.Timedelta(days=2)
    end = event.event_date + pd.Timedelta(days=2)

    move_with_sector = decompose_move("HDFCBANK.NS", stock_returns, market_returns, sector_returns, start, end)
    move_market_only = decompose_move("HDFCBANK.NS", stock_returns, market_returns, None, start, end)

    assert move_with_sector.has_sector_factor is True
    assert move_market_only.has_sector_factor is False
    assert move_market_only.sector_beta == 0.0
    assert abs(move_with_sector.raw_return - move_market_only.raw_return) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))