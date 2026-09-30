"""DEV-ONLY factor-driven synthetic data with injected labeled events (see earlier turns for full rationale)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class InjectedEvent:
    event_date: pd.Timestamp
    direction: str
    magnitude: float
    headline: str
    tag: str

def generate_factor_world(tickers, sector_of_ticker, n_days=500, market_vol=0.011, sector_vol=0.007,
                           idio_vol=0.013, n_events_per_stock=(1, 2), seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    market_returns = rng.normal(0.0003, market_vol, n_days)
    sectors = sorted(set(sector_of_ticker.values()))
    sector_returns = {}
    for sector in sectors:
        sector_beta_to_mkt = rng.uniform(0.7, 1.3)
        sector_idio = rng.normal(0, sector_vol, n_days)
        sector_returns[sector] = sector_beta_to_mkt * market_returns + sector_idio
    stocks = {}
    for ticker in tickers:
        sector = sector_of_ticker[ticker]
        beta_mkt = rng.uniform(0.5, 1.5)
        beta_sector = rng.uniform(0.4, 1.1)
        idio = rng.normal(0, idio_vol, n_days)
        n_events = rng.integers(n_events_per_stock[0], n_events_per_stock[1] + 1)
        events = []
        used_positions = set()
        for _ in range(n_events):
            pos = rng.integers(60, n_days - 10)
            while pos in used_positions:
                pos = rng.integers(60, n_days - 10)
            used_positions.add(pos)
            is_negative = rng.random() < 0.5
            magnitude = rng.uniform(0.025, 0.06) * (-1 if is_negative else 1)
            idio[pos] += magnitude
            topic = rng.choice(["asset quality", "margin outlook", "capex plans"])
            sector_label = sector if sector == "IT" else sector.lower()
            template_idx = rng.integers(0, 3)
            if template_idx == 0:
                tag = "earnings_surprise"
                headline = "Weak quarterly results, missing street estimates" if is_negative else "Strong quarterly results, beating street estimates"
            elif template_idx == 1:
                tag = "regulatory"
                headline = f"Regulator flags concerns over {sector_label} sector practices" if is_negative else f"Regulator clears {sector_label} sector of prior compliance concerns"
            else:
                tag = "management"
                headline = f"Cautious management guidance issued on {topic}" if is_negative else f"Optimistic management guidance issued on {topic}"
            events.append(InjectedEvent(event_date=dates[pos], direction="negative" if is_negative else "positive",
                                         magnitude=magnitude, headline=headline, tag=tag))
        total_returns = beta_mkt * market_returns + beta_sector * sector_returns[sector] + idio
        prices = 100 * np.exp(np.cumsum(total_returns))
        stocks[ticker] = {"returns": total_returns, "prices": prices, "beta_mkt": beta_mkt,
                           "beta_sector": beta_sector, "events": events}
    return {"dates": dates, "market_returns": market_returns, "sector_returns": sector_returns, "stocks": stocks}
