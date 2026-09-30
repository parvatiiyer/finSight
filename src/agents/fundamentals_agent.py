"""Fundamentals evidence agent — earnings-date proximity and ratio deltas."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agents.evidence_types import Evidence, Direction

def check_earnings_proximity(ticker, window_start, window_end, earnings_dates, proximity_days=3):
    for edate in earnings_dates:
        edate = pd.Timestamp(edate)
        if (window_start - pd.Timedelta(days=proximity_days)) <= edate <= window_end:
            return Evidence("fundamentals", edate.date(), Direction.NEUTRAL, 0.7,
                             f"Quarterly results date ({edate.date()}) falls within the move window",
                             1.0, ["earnings_date"],
                             source_url=f"https://finance.yahoo.com/calendar/earnings?symbol={ticker}",
                             source_path=f"sec_edgar/earnings_calendar/{ticker}")
    return None

def check_ratio_deltas(ticker, current_fundamentals, prior_fundamentals, as_of_date, material_change_pct=0.15):
    watched_ratios = {
        "returnOnEquity": ("ROE", True), "profitMargins": ("Profit margin", True),
        "debtToEquity": ("Debt/Equity", False), "currentRatio": ("Current ratio", True),
        "revenueGrowth": ("Revenue growth", True),
    }
    evidence = []
    for field, (label, higher_is_better) in watched_ratios.items():
        cur = current_fundamentals.get(field)
        prior = prior_fundamentals.get(field)
        if cur is None or prior is None or prior == 0:
            continue
        pct_change = (cur - prior) / abs(prior)
        if abs(pct_change) < material_change_pct:
            continue
        improved = (pct_change > 0) == higher_is_better
        evidence.append(Evidence("fundamentals", as_of_date,
                                  Direction.POSITIVE if improved else Direction.NEGATIVE,
                                  min(1.0, abs(pct_change) / 0.5),
                                  f"{label} changed {pct_change:+.1%} quarter-over-quarter ({prior:.3g} -> {cur:.3g})",
                                  1.0, ["ratio_delta", field],
                                  source_url=f"https://finance.yahoo.com/quote/{ticker}/key-statistics",
                                  source_path=f"financial_statements/quarterly_ratios/{ticker}#{field}"))
    return evidence

def gather_fundamentals_evidence(ticker, window_start, window_end, current_fundamentals,
                                  prior_fundamentals=None, earnings_dates=None):
    evidence = []
    if earnings_dates:
        e = check_earnings_proximity(ticker, window_start, window_end, earnings_dates)
        if e:
            evidence.append(e)
    if prior_fundamentals:
        evidence.extend(check_ratio_deltas(ticker, current_fundamentals, prior_fundamentals, as_of_date=window_end.date()))
    return evidence