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

def _covers_range(df: pd.DataFrame, start=None, end=None) -> bool:
    if start is None:
        return True
    idx = df.index
    idx_min = idx.min().tz_localize(None) if idx.tz else idx.min()
    start_ts = pd.Timestamp(start).tz_localize(None) if getattr(pd.Timestamp(start), "tz", None) else pd.Timestamp(start)
    if idx_min > start_ts:
        return False
    if end is not None:
        idx_max = idx.max().tz_localize(None) if idx.tz else idx.max()
        end_ts = pd.Timestamp(end).tz_localize(None) if getattr(pd.Timestamp(end), "tz", None) else pd.Timestamp(end)
        if idx_max < end_ts:
            return False
    return True

def fetch_ohlcv(
    ticker,
    period="3y",
    interval="1d",
    max_retries=3,
    use_cache=True,
    cache_max_age_hours=24,
    start=None,
    end=None,
):
    safe_ticker = ticker.replace(".", "_").replace("^", "_")

    # Fast path 1: Check if ANY existing cached parquet for this ticker covers the requested window
    if use_cache:
        # Check specific period cache file first
        cache_file = _cache_path(ticker, f"ohlcv_{period}_{interval}")
        if cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                if len(cached) >= MIN_TRADING_DAYS:
                    if _covers_range(cached, start, end):
                        return cached
                else:
                    cache_file.unlink(missing_ok=True)
            except Exception:
                pass

        # Check all existing parquet files for this ticker on disk (e.g. ohlcv_max_1d, ohlcv_2y_1d)
        for p in sorted(RAW_DIR.glob(f"{safe_ticker}__ohlcv_*.parquet"), key=lambda x: x.stat().st_size, reverse=True):
            try:
                cached = pd.read_parquet(p)
                if len(cached) >= MIN_TRADING_DAYS and _covers_range(cached, start, end):
                    return cached
            except Exception:
                continue

    # For optional sector indices (e.g. ^CNXENERGY, ^CNXIT, ^NSEBANK), do NOT wait through 3 slow exponential retries (14s)
    is_optional_sector = ticker.startswith("^") and ticker != "^NSEI"
    effective_retries = 1 if is_optional_sector else max_retries

    # Prepare fetch arguments: bounded start/end if available to avoid downloading 30 years over rate-limited connections
    fetch_kwargs = {"interval": interval, "auto_adjust": True}
    if start is not None:
        start_ts = pd.Timestamp(start).tz_localize(None) if getattr(pd.Timestamp(start), "tz", None) else pd.Timestamp(start)
        # Pull 1 year of trailing history before start to compute betas reliably
        bounded_start = (start_ts - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        if end is not None:
            end_ts = pd.Timestamp(end).tz_localize(None) if getattr(pd.Timestamp(end), "tz", None) else pd.Timestamp(end)
            bounded_end = (end_ts + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            bounded_end = None
        fetch_kwargs["start"] = bounded_start
        if bounded_end:
            fetch_kwargs["end"] = bounded_end
    else:
        fetch_kwargs["period"] = period

    last_err = None
    for attempt in range(1, effective_retries + 1):
        try:
            df = yf.Ticker(ticker).history(**fetch_kwargs)
            if df.empty:
                raise ValueError(f"yfinance returned empty history for {ticker}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "date"
            if len(df) < MIN_TRADING_DAYS:
                raise ValueError(
                    f"yfinance returned only {len(df)} rows for {ticker} (expected roughly {MIN_TRADING_DAYS}+)"
                )
            # Save to cache
            cache_file = _cache_path(ticker, f"ohlcv_{period}_{interval}")
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            last_err = e
            if attempt < effective_retries:
                wait = 2 ** attempt
                logger.warning(f"{ticker}: attempt {attempt}/{effective_retries} failed ({e}); retrying in {wait}s")
                time.sleep(wait)

    # Fallback to ANY cached parquet on disk (even if partial)
    fallback_caches = sorted(RAW_DIR.glob(f"{safe_ticker}__ohlcv_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fallback_caches:
        for fb in fallback_caches:
            try:
                cached = pd.read_parquet(fb)
                if len(cached) >= MIN_TRADING_DAYS:
                    logger.warning(f"{ticker}: live fetch failed ({last_err}); falling back to cached file {fb.name}")
                    return cached
            except Exception:
                continue

    raise RuntimeError(f"Failed to fetch OHLCV for {ticker} after {effective_retries} attempts: {last_err}")

def fetch_fundamentals(ticker, use_cache=True, cache_max_age_hours=24):
    cache_file = _cache_path(ticker, "fundamentals")
    safe_ticker = ticker.replace(".", "_").replace("^", "_")

    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_max_age_hours:
            return pd.read_parquet(cache_file).iloc[0].to_dict()

    info = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        logger.warning(f"{ticker}: yfinance info fetch failed ({e})")

    fields = ["trailingPE", "priceToBook", "returnOnEquity", "debtToEquity", "currentRatio", "quickRatio",
              "profitMargins", "operatingMargins", "revenueGrowth", "earningsGrowth", "beta", "marketCap",
              "dividendYield", "freeCashflow", "totalDebt", "totalCash", "sector", "longName", "shortName"]

    if not info and cache_file.exists():
        logger.warning(f"{ticker}: using cached fundamentals as fallback")
        return pd.read_parquet(cache_file).iloc[0].to_dict()

    row = {f: info.get(f) for f in fields}
    row["ticker"] = ticker
    row["fetched_at"] = datetime.utcnow().isoformat()
    if info:
        try:
            pd.DataFrame([row]).to_parquet(cache_file)
        except Exception:
            pass
    return row

def fetch_company_sector(ticker: str) -> str | None:
    """Just the .info['sector'] field — used to pick a sector index for arbitrary tickers."""
    # Check cached fundamentals first to avoid redundant network call
    cache_file = _cache_path(ticker, "fundamentals")
    if cache_file.exists():
        try:
            cached_sector = pd.read_parquet(cache_file).iloc[0].to_dict().get("sector")
            if cached_sector:
                return cached_sector
        except Exception:
            pass
    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:
        return None

def fetch_quarterly_fundamentals_history(ticker: str, n_quarters: int = 2) -> list[dict]:
    """
    Pulls the last n_quarters of quarterly financials/balance sheet and
    derives the same ratio fields fundamentals_agent.check_ratio_deltas
    expects, with parquet caching so repeated calls don't hang on rate-limited connections.
    """
    cache_file = _cache_path(ticker, "quarterly_history")
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file).to_dict(orient="records")
        except Exception:
            pass

    try:
        t = yf.Ticker(ticker)
        fin = t.quarterly_financials
        bs = t.quarterly_balance_sheet
        if fin is None or bs is None or fin.empty or bs.empty:
            return []
        results = []
        cols = list(fin.columns[:n_quarters])
        for col in cols:
            try:
                net_income = fin.loc["Net Income", col] if "Net Income" in fin.index else None
                revenue = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
                equity = bs.loc["Stockholders Equity", col] if "Stockholders Equity" in bs.index else None
                total_debt = bs.loc["Total Debt", col] if "Total Debt" in bs.index else None
                current_assets = bs.loc["Current Assets", col] if "Current Assets" in bs.index else None
                current_liab = bs.loc["Current Liabilities", col] if "Current Liabilities" in bs.index else None
                results.append({
                    "date": str(col),
                    "returnOnEquity": float(net_income / equity) if net_income is not None and equity else None,
                    "profitMargins": float(net_income / revenue) if net_income is not None and revenue else None,
                    "debtToEquity": float(total_debt / equity * 100) if total_debt is not None and equity else None,
                    "currentRatio": float(current_assets / current_liab) if current_assets is not None and current_liab else None,
                    "revenueGrowth": None,
                })
            except Exception:
                continue
        for i in range(len(results) - 1):
            cur_rev = fin.loc["Total Revenue", cols[i]] if "Total Revenue" in fin.index else None
            prior_rev = fin.loc["Total Revenue", cols[i + 1]] if "Total Revenue" in fin.index else None
            if cur_rev and prior_rev:
                results[i]["revenueGrowth"] = float((cur_rev - prior_rev) / abs(prior_rev))
        if results:
            try:
                pd.DataFrame(results).to_parquet(cache_file)
            except Exception:
                pass
        return results
    except Exception as e:
        logger.debug(f"{ticker}: quarterly fundamentals fetch failed ({e})")
        return []

def fetch_earnings_dates(ticker: str, limit: int = 4) -> list:
    """Fetches quarterly results dates with parquet caching."""
    cache_file = _cache_path(ticker, "earnings_dates")
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            return [pd.Timestamp(d) for d in df["date"]]
        except Exception:
            pass

    try:
        t = yf.Ticker(ticker)
        edf = t.get_earnings_dates(limit=limit)
        if edf is not None and not edf.empty:
            dates = [pd.Timestamp(d) for d in edf.index]
            try:
                pd.DataFrame({"date": [str(d) for d in dates]}).to_parquet(cache_file)
            except Exception:
                pass
            return dates
        return []
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
