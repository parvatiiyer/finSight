"""Fundamental ratio features, broadcast from a point-in-time snapshot."""
import pandas as pd

def compute_derived_ratios(fundamentals: dict) -> dict:
    f = dict(fundamentals)
    debt, cash, mcap = f.get("totalDebt"), f.get("totalCash"), f.get("marketCap")
    f["net_debt"] = (debt - cash) if (debt is not None and cash is not None) else None
    f["net_debt_to_mcap"] = f["net_debt"] / mcap if (f["net_debt"] is not None and mcap not in (None, 0)) else None
    fcf = f.get("freeCashflow")
    f["fcf_yield"] = fcf / mcap if (fcf is not None and mcap not in (None, 0)) else None
    return f

def fundamentals_to_frame(fundamentals: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    enriched = compute_derived_ratios(fundamentals)
    cols = {k: v for k, v in enriched.items() if k not in ("ticker", "fetched_at", "sector", "longName", "shortName")}
    return pd.DataFrame([cols] * len(index), index=index)
