"""
FinSight stock universe.

Deliberately curated (not scraped) so the project has predictable, sector-diverse
coverage for a portfolio demo: enough large-caps to build robust features, spread
across sectors so peer-comparison mode (Section G, stretch goal) has real peers to
compare within.

yfinance ticker convention for NSE: SYMBOL.NS
"""

UNIVERSE = {
    "Banking": [
        ("HDFCBANK.NS", "HDFC Bank"),
        ("ICICIBANK.NS", "ICICI Bank"),
        ("KOTAKBANK.NS", "Kotak Mahindra Bank"),
        ("AXISBANK.NS", "Axis Bank"),
        ("SBIN.NS", "State Bank of India"),
    ],
    "IT": [
        ("TCS.NS", "Tata Consultancy Services"),
        ("INFY.NS", "Infosys"),
        ("WIPRO.NS", "Wipro"),
        ("HCLTECH.NS", "HCL Technologies"),
        ("TECHM.NS", "Tech Mahindra"),
    ],
    "Energy": [
        ("RELIANCE.NS", "Reliance Industries"),
        ("ONGC.NS", "Oil & Natural Gas Corp"),
        ("NTPC.NS", "NTPC"),
        ("POWERGRID.NS", "Power Grid Corp"),
    ],
    "FMCG": [
        ("HINDUNILVR.NS", "Hindustan Unilever"),
        ("ITC.NS", "ITC"),
        ("NESTLEIND.NS", "Nestle India"),
        ("BRITANNIA.NS", "Britannia Industries"),
    ],
    "Auto": [
        ("MARUTI.NS", "Maruti Suzuki"),
        ("TATAMOTORS.NS", "Tata Motors"),
        ("M&M.NS", "Mahindra & Mahindra"),
        ("BAJAJ-AUTO.NS", "Bajaj Auto"),
    ],
    "Pharma": [
        ("SUNPHARMA.NS", "Sun Pharmaceutical"),
        ("DRREDDY.NS", "Dr. Reddy's Labs"),
        ("CIPLA.NS", "Cipla"),
    ],
}

MARKET_INDEX = "^NSEI"  # Nifty 50 — the market factor

SECTOR_INDICES = {
    "Banking": "^NSEBANK",
    "IT": "^CNXIT",
    "Energy": "^CNXENERGY",
    "FMCG": "^CNXFMCG",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
}

# Maps yfinance's free-text .info['sector'] values (Yahoo's GICS-ish taxonomy,
# which covers ANY global stock, not just our curated 25) onto the NSE sector
# indices above. This is what lets an arbitrary ticker still get a sector
# factor instead of falling back to market-only. Deliberately conservative:
# a mapping only exists where the analogy is reasonably tight; anything else
# resolves to None and the factor model degrades gracefully (see
# src/features/factor_model.py's sector_returns=None path).
YFINANCE_SECTOR_MAP = {
    "Financial Services": "Banking",
    "Financial": "Banking",
    "Technology": "IT",
    "Information Technology": "IT",
    "Energy": "Energy",
    "Utilities": "Energy",
    "Consumer Defensive": "FMCG",
    "Consumer Staples": "FMCG",
    "Consumer Cyclical": "Auto",
    "Consumer Discretionary": "Auto",
    "Healthcare": "Pharma",
}


def sector_index_of(ticker):
    """The sector index ticker to regress `ticker` against, for curated tickers."""
    sector = sector_of(ticker)
    return SECTOR_INDICES.get(sector)


def sector_index_from_yf_sector(yf_sector: str | None):
    """
    For an arbitrary (non-curated) ticker: map yfinance's reported .info
    sector string to one of our sector indices, or None if there's no good
    analogy — callers must treat None as "use market-only factor model."
    """
    if not yf_sector:
        return None
    mapped_sector = YFINANCE_SECTOR_MAP.get(yf_sector)
    return SECTOR_INDICES.get(mapped_sector) if mapped_sector else None


def all_tickers():
    return [t for sector in UNIVERSE.values() for t, _ in sector]


def sector_of(ticker):
    for sector, stocks in UNIVERSE.items():
        for t, _ in stocks:
            if t == ticker:
                return sector
    return None


def peers_of(ticker, exclude_self=True):
    sector = sector_of(ticker)
    if sector is None:
        return []
    peers = [t for t, _ in UNIVERSE[sector]]
    if exclude_self and ticker in peers:
        peers.remove(ticker)
    return peers


DEMO_STOCKS = {
    "steady": "HDFCBANK.NS",
    "volatile": "TATAMOTORS.NS",
    "negative_event": "PAYTM.NS",
}

if __name__ == "__main__":
    tickers = all_tickers()
    print(f"Universe size: {len(tickers)} stocks across {len(UNIVERSE)} sectors")
    print(f"Peers of HDFCBANK.NS: {peers_of('HDFCBANK.NS')}")
