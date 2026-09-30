"""Production data ingestion via yfinance, with retry/backoff and parquet caching."""
import time, logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finsight.ingest")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(ticker: str, kind: str) -> Path:
    safe_ticker = ticker.replace(".", "_").replace("^", "_")
    return RAW_DIR / f"{safe_ticker}__{kind}.parquet"

MIN_TRADING_DAYS = 30  # anything less almost certainly means a bad/partial fetch, not a real listing

def fetch_ohlcv(ticker, period="3y", interval="1d", max_retries=3, use_cache=True, cache_max_age_hours=12):
    cache_file = _cache_path(ticker, f"ohlcv_{period}_{interval}")
    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_max_age_hours:
            cached = pd.read_parquet(cache_file)
            if len(cached) >= MIN_TRADING_DAYS:
                return cached
            logger.warning(f"{ticker}: cached file has only {len(cached)} rows — treating as stale, refetching")
            # fall through to refetch rather than trusting a thin cache

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"yfinance returned empty history for {ticker}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "date"
            if len(df) < MIN_TRADING_DAYS:
                # Non-empty but too thin to be a legitimate multi-year fetch for
                # `period` — almost always a transient/partial response, not a
                # real "this stock only has 5 days of history" situation. Don't
                # cache it, so the next call retries instead of being stuck
                # replaying the same bad data for up to cache_max_age_hours.
                raise ValueError(
                    f"yfinance returned only {len(df)} rows for {ticker} over period={period!r} "
                    f"(expected roughly {MIN_TRADING_DAYS}+) — treating as a partial/failed fetch"
                )
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning(f"{ticker}: attempt {attempt}/{max_retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch OHLCV for {ticker} after {max_retries} attempts: {last_err}")

def fetch_fundamentals(ticker, use_cache=True, cache_max_age_hours=24):
    cache_file = _cache_path(ticker, "fundamentals")
    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_max_age_hours:
            return pd.read_parquet(cache_file).iloc[0].to_dict()
    info = yf.Ticker(ticker).info
    fields = ["trailingPE", "priceToBook", "returnOnEquity", "debtToEquity", "currentRatio", "quickRatio",
              "profitMargins", "operatingMargins", "revenueGrowth", "earningsGrowth", "beta", "marketCap",
              "dividendYield", "freeCashflow", "totalDebt", "totalCash", "sector", "longName", "shortName"]
    row = {f: info.get(f) for f in fields}
    row["ticker"] = ticker
    row["fetched_at"] = datetime.utcnow().isoformat()
    pd.DataFrame([row]).to_parquet(cache_file)
    return row

def fetch_company_sector(ticker: str) -> str | None:
    """Just the .info['sector'] field — used to pick a sector index for arbitrary tickers."""
    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:
        return None

def fetch_quarterly_fundamentals_history(ticker: str, n_quarters: int = 2) -> list[dict]:
    """
    Pulls the last n_quarters of quarterly financials/balance sheet and
    derives the same ratio fields fundamentals_agent.check_ratio_deltas
    expects, so ratio-delta evidence works for ANY ticker, not just ones
    with a hand-maintained fundamentals history.
    """
    t = yf.Ticker(ticker)
    fin = t.quarterly_financials
    bs = t.quarterly_balance_sheet
    if fin.empty or bs.empty:
        return []
    results = []
    cols = fin.columns[:n_quarters]
    for col in cols:
        try:
            net_income = fin.loc["Net Income", col] if "Net Income" in fin.index else None
            revenue = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
            equity = bs.loc["Stockholders Equity", col] if "Stockholders Equity" in bs.index else None
            total_debt = bs.loc["Total Debt", col] if "Total Debt" in bs.index else None
            current_assets = bs.loc["Current Assets", col] if "Current Assets" in bs.index else None
            current_liab = bs.loc["Current Liabilities", col] if "Current Liabilities" in bs.index else None
            results.append({
                "date": col,
                "returnOnEquity": float(net_income / equity) if net_income is not None and equity else None,
                "profitMargins": float(net_income / revenue) if net_income is not None and revenue else None,
                "debtToEquity": float(total_debt / equity * 100) if total_debt is not None and equity else None,
                "currentRatio": float(current_assets / current_liab) if current_assets is not None and current_liab else None,
                "revenueGrowth": None,  # filled in below once we have >=2 quarters
            })
        except Exception:
            continue
    for i in range(len(results) - 1):
        cur_rev = fin.loc["Total Revenue", cols[i]] if "Total Revenue" in fin.index else None
        prior_rev = fin.loc["Total Revenue", cols[i + 1]] if "Total Revenue" in fin.index else None
        if cur_rev and prior_rev:
            results[i]["revenueGrowth"] = float((cur_rev - prior_rev) / abs(prior_rev))
    return results

def fetch_earnings_dates(ticker: str, limit: int = 4) -> list:
    try:
        t = yf.Ticker(ticker)
        edf = t.get_earnings_dates(limit=limit)
        return list(edf.index) if edf is not None else []
    except Exception:
        return []

def fetch_universe(tickers, period="3y", sleep_between=0.5):
    results, failed = {}, []
    for i, ticker in enumerate(tickers, 1):
        try:
            ohlcv = fetch_ohlcv(ticker, period=period)
            fundamentals = fetch_fundamentals(ticker)
            results[ticker] = {"ohlcv": ohlcv, "fundamentals": fundamentals}
        except Exception as e:
            logger.error(f"{ticker}: giving up — {e}")
            failed.append(ticker)
        time.sleep(sleep_between)
    return results
