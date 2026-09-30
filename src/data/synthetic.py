"""DEV-ONLY synthetic OHLCV/fundamentals fixture (see earlier project turns for full rationale)."""
import numpy as np
import pandas as pd

RNG_SEED = 42

def generate_ohlcv(ticker: str, n_days: int = 750, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else (hash(ticker) % (2**32)))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    log_vol = np.zeros(n_days)
    log_vol[0] = np.log(0.015)
    for t in range(1, n_days):
        log_vol[t] = 0.95 * log_vol[t - 1] + 0.05 * np.log(0.015) + rng.normal(0, 0.15)
    daily_vol = np.exp(log_vol)
    drift = np.full(n_days, 0.0003)
    n_shocks = rng.integers(1, 4)
    for _ in range(n_shocks):
        shock_start = rng.integers(50, n_days - 50)
        shock_len = rng.integers(10, 40)
        shock_mag = rng.normal(0, 0.002)
        drift[shock_start:shock_start + shock_len] += shock_mag
    log_returns = drift + daily_vol * rng.standard_normal(n_days)
    price = 100 * np.exp(np.cumsum(log_returns))
    close = price
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    intraday_range = np.abs(rng.normal(0, 1, n_days)) * daily_vol * close
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range
    base_volume = 2_000_000
    volume = (base_volume * (1 + 8 * np.abs(log_returns) / daily_vol.mean())
              * (1 + rng.normal(0, 0.2, n_days))).clip(min=10_000).astype(int)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    df.index.name = "date"
    return df

def generate_fundamentals(ticker: str, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed if seed is not None else (hash(ticker) % (2**32)))
    return {
        "trailingPE": float(rng.uniform(12, 45)), "priceToBook": float(rng.uniform(1.5, 12)),
        "returnOnEquity": float(rng.uniform(0.08, 0.28)), "debtToEquity": float(rng.uniform(5, 120)),
        "currentRatio": float(rng.uniform(0.8, 2.5)), "quickRatio": float(rng.uniform(0.6, 2.0)),
        "profitMargins": float(rng.uniform(0.05, 0.30)), "operatingMargins": float(rng.uniform(0.10, 0.35)),
        "revenueGrowth": float(rng.uniform(-0.05, 0.25)), "earningsGrowth": float(rng.uniform(-0.10, 0.30)),
        "beta": float(rng.uniform(0.6, 1.6)), "marketCap": float(rng.uniform(2e11, 1.5e13)),
        "dividendYield": float(rng.uniform(0.0, 0.03)), "freeCashflow": float(rng.uniform(-5e9, 5e10)),
        "totalDebt": float(rng.uniform(1e9, 8e11)), "totalCash": float(rng.uniform(1e9, 3e11)),
        "ticker": ticker, "fetched_at": "synthetic",
    }

def generate_universe(tickers: list[str], n_days: int = 750) -> dict:
    return {t: {"ohlcv": generate_ohlcv(t, n_days=n_days), "fundamentals": generate_fundamentals(t)} for t in tickers}
