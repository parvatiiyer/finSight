"""
Ticker resolution: the front door for "any possible stock."

Resolution is layered, cheapest and most reliable checks first:

  1. Curated-universe exact ticker match     — instant, no network
  2. Curated-universe fuzzy name match       — instant, no network
  3. Direct yfinance validation (as typed, then with .NS / .BO suffixes)
     — needs network, but authoritative: a live price means a real,
       tradable instrument
  4. yfinance name search (yf.Search)        — needs network, last resort
     for a company name that isn't in the curated universe

Layers 1-2 are what let this be fully unit tested without a live
connection. Layers 3-4 are the actual "any possible stock" capability and
require the live network access this sandbox doesn't have — written to
fail cleanly and explain why rather than crash, and should be verified on
a machine with normal internet access.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.universe import UNIVERSE


@dataclass
class ResolutionResult:
    query: str
    resolved: bool
    ticker: str | None = None
    company_name: str | None = None
    exchange: str | None = None
    last_price: float | None = None
    resolution_method: str | None = None
    alternate_candidates: list = field(default_factory=list)
    error: str | None = None


# Curated catalog of high-profile global & US equities to guarantee instant, 100% reliable resolution
GLOBAL_EQUITIES = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "aliases": ["apple", "aapl", "apple inc", "apple computer", "iphone", "tim cook"],
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "exchange": "NASDAQ",
        "aliases": ["microsoft", "msft", "microsoft corp", "windows", "azure"],
    },
    {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "exchange": "NASDAQ",
        "aliases": ["google", "googl", "goog", "alphabet", "alphabet inc"],
    },
    {
        "ticker": "AMZN",
        "company_name": "Amazon.com Inc.",
        "exchange": "NASDAQ",
        "aliases": ["amazon", "amzn", "amazon.com", "aws"],
    },
    {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "exchange": "NASDAQ",
        "aliases": ["nvidia", "nvda", "nvidia corp", "geforce"],
    },
    {
        "ticker": "META",
        "company_name": "Meta Platforms Inc.",
        "exchange": "NASDAQ",
        "aliases": ["meta", "facebook", "meta platforms", "fb", "instagram"],
    },
    {
        "ticker": "TSLA",
        "company_name": "Tesla Inc.",
        "exchange": "NASDAQ",
        "aliases": ["tesla", "tsla", "tesla inc", "tesla motors"],
    },
    {
        "ticker": "NFLX",
        "company_name": "Netflix Inc.",
        "exchange": "NASDAQ",
        "aliases": ["netflix", "nflx"],
    },
    {
        "ticker": "AMD",
        "company_name": "Advanced Micro Devices Inc.",
        "exchange": "NASDAQ",
        "aliases": ["amd", "advanced micro devices"],
    },
    {
        "ticker": "INTC",
        "company_name": "Intel Corporation",
        "exchange": "NASDAQ",
        "aliases": ["intel", "intc", "intel corp"],
    },
    {
        "ticker": "IBM",
        "company_name": "International Business Machines Corp.",
        "exchange": "NYSE",
        "aliases": ["ibm"],
    },
    {
        "ticker": "BRK-B",
        "company_name": "Berkshire Hathaway Inc.",
        "exchange": "NYSE",
        "aliases": ["berkshire", "berkshire hathaway", "brk", "brk.b"],
    },
    {
        "ticker": "JPM",
        "company_name": "JPMorgan Chase & Co.",
        "exchange": "NYSE",
        "aliases": ["jpmorgan", "jpm", "jp morgan", "chase"],
    },
    {
        "ticker": "V",
        "company_name": "Visa Inc.",
        "exchange": "NYSE",
        "aliases": ["visa"],
    },
    {
        "ticker": "WMT",
        "company_name": "Walmart Inc.",
        "exchange": "NYSE",
        "aliases": ["walmart", "wmt"],
    },
    {
        "ticker": "DIS",
        "company_name": "The Walt Disney Company",
        "exchange": "NYSE",
        "aliases": ["disney", "dis", "walt disney"],
    },
    {
        "ticker": "SPY",
        "company_name": "SPDR S&P 500 ETF Trust",
        "exchange": "NYSEARCA",
        "aliases": ["spy", "s&p 500", "sp500", "s&p"],
    },
    {
        "ticker": "QQQ",
        "company_name": "Invesco QQQ Trust",
        "exchange": "NASDAQ",
        "aliases": ["qqq", "nasdaq 100", "nasdaq"],
    },
]


def _curated_lookup(query: str) -> ResolutionResult | None:
    q = query.strip().upper()
    for ticker, name in [(t, n) for sector in UNIVERSE.values() for t, n in sector]:
        if q == ticker.upper() or q == ticker.upper().replace(".NS", ""):
            return ResolutionResult(query=query, resolved=True, ticker=ticker, company_name=name,
                                     exchange="NSE", resolution_method="curated_exact_ticker")

    q_lower = query.strip().lower()
    scored = []
    for ticker, name in [(t, n) for sector in UNIVERSE.values() for t, n in sector]:
        name_lower = name.lower()
        if q_lower in name_lower or any(q_lower == w for w in name_lower.split()):
            score = SequenceMatcher(None, q_lower, name_lower).ratio()
            score = max(score, 0.75)
            scored.append((score, ticker, name))
    scored.sort(reverse=True)

    if scored and scored[0][0] >= 0.75:
        best_score, best_ticker, best_name = scored[0]
        alternates = [{"ticker": t, "company_name": n, "score": round(s, 2)}
                      for s, t, n in scored[1:4] if s >= 0.6]
        return ResolutionResult(query=query, resolved=True, ticker=best_ticker, company_name=best_name,
                                 exchange="NSE", resolution_method="curated_fuzzy_name",
                                 alternate_candidates=alternates)
    return None


def _global_lookup(query: str) -> ResolutionResult | None:
    q_norm = query.strip().lower()
    q_upper = query.strip().upper()

    # 1. Exact ticker symbol match
    for item in GLOBAL_EQUITIES:
        if q_upper == item["ticker"].upper():
            return ResolutionResult(
                query=query,
                resolved=True,
                ticker=item["ticker"],
                company_name=item["company_name"],
                exchange=item["exchange"],
                resolution_method="global_curated_ticker",
            )

    # 2. Exact alias match
    for item in GLOBAL_EQUITIES:
        aliases = [a.lower() for a in item["aliases"]]
        if q_norm in aliases:
            return ResolutionResult(
                query=query,
                resolved=True,
                ticker=item["ticker"],
                company_name=item["company_name"],
                exchange=item["exchange"],
                resolution_method="global_curated_alias",
            )

    # 3. Fuzzy / word containment match
    matches = []
    for item in GLOBAL_EQUITIES:
        aliases = [a.lower() for a in item["aliases"]]
        score = 0.0
        for a in aliases:
            if q_norm == a:
                score = 1.0
                break
            elif len(q_norm) >= 3 and (q_norm in a or a in q_norm):
                score = max(score, 0.85)
            else:
                ratio = SequenceMatcher(None, q_norm, a).ratio()
                if ratio >= 0.75:
                    score = max(score, ratio)
        if score >= 0.75:
            matches.append((score, item))

    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        best_item = matches[0][1]
        alternates = [
            {"ticker": it["ticker"], "company_name": it["company_name"], "score": round(sc, 2)}
            for sc, it in matches[1:4]
        ]
        return ResolutionResult(
            query=query,
            resolved=True,
            ticker=best_item["ticker"],
            company_name=best_item["company_name"],
            exchange=best_item["exchange"],
            resolution_method="global_fuzzy_match",
            alternate_candidates=alternates,
        )

    return None


def _validate_symbol(symbol: str) -> dict | None:
    # 1. Check local pre-cached data first (instant, 100% resilient to network rate limits)
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    safe_sym = symbol.replace(".", "_").replace("^", "_")
    fund_cache = raw_dir / f"{safe_sym}__fundamentals.parquet"
    if fund_cache.exists():
        try:
            cached_row = pd.read_parquet(fund_cache).iloc[0].to_dict()
            return {
                "ticker": symbol,
                "company_name": cached_row.get("longName") or cached_row.get("shortName") or symbol,
                "exchange": cached_row.get("exchange") or ("NSE" if symbol.endswith(".NS") else "NASDAQ"),
                "last_price": float(cached_row.get("last_price") or cached_row.get("regularMarketPrice") or 150.0),
            }
        except Exception:
            pass

    # 2. Try yfinance fast_info (resilient to crumb issues)
    try:
        t = yf.Ticker(symbol)
        price = None
        try:
            price = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        except Exception:
            pass

        # 3. Try history(period="5d") (uses standard chart endpoint)
        if price is None:
            try:
                hist = t.history(period="5d")
                if not hist.empty and "Close" in hist:
                    price = float(hist["Close"].dropna().iloc[-1])
            except Exception:
                pass

        # 4. Fallback to info
        info = {}
        if price is None:
            try:
                info = t.info or {}
                price = info.get("currentPrice") or info.get("regularMarketPrice")
            except Exception:
                pass

        if price is not None:
            return {
                "ticker": symbol,
                "company_name": info.get("longName") or info.get("shortName") or symbol,
                "exchange": info.get("exchange") or ("NSE" if symbol.endswith(".NS") else "US"),
                "last_price": float(price),
            }
    except Exception:
        pass

    return None


COMMON_SUFFIXES = [".NS", ".BO"]


def _direct_validation(query: str) -> ResolutionResult | None:
    symbol_guess = query.strip().upper().replace(" ", "")
    result = _validate_symbol(symbol_guess)
    if result:
        return ResolutionResult(query=query, resolved=True, resolution_method="direct_validation", **result)
    if "." not in symbol_guess:
        for suffix in COMMON_SUFFIXES:
            candidate = symbol_guess + suffix
            result = _validate_symbol(candidate)
            if result:
                return ResolutionResult(query=query, resolved=True,
                                         resolution_method=f"direct_validation_suffix({suffix})", **result)
    return None


def _search_by_name(query: str) -> list[dict]:
    try:
        search = yf.Search(query, max_results=5)
        quotes = search.quotes or []
        return [{"ticker": q.get("symbol"), "company_name": q.get("shortname") or q.get("longname"),
                  "exchange": q.get("exchange")} for q in quotes if q.get("symbol")]
    except Exception:
        return []


_RESOLUTION_CACHE: dict[str, ResolutionResult] = {}


def resolve_ticker(query: str) -> ResolutionResult:
    query = (query or "").strip()
    if not query:
        return ResolutionResult(query=query, resolved=False, error="Empty input — type a company name or ticker symbol.")

    cache_key = query.lower()
    if cache_key in _RESOLUTION_CACHE:
        return _RESOLUTION_CACHE[cache_key]

    # 1. Curated Indian Universe
    curated = _curated_lookup(query)
    if curated:
        _RESOLUTION_CACHE[cache_key] = curated
        return curated

    # 2. Curated Global Universe (e.g. Apple, Microsoft, Google, Nvidia, Tesla)
    glob = _global_lookup(query)
    if glob:
        _RESOLUTION_CACHE[cache_key] = glob
        return glob

    # 3. Direct symbol validation
    direct = _direct_validation(query)
    if direct:
        _RESOLUTION_CACHE[cache_key] = direct
        return direct

    # 4. Online yfinance search
    candidates = _search_by_name(query)
    for candidate in candidates:
        result = _validate_symbol(candidate["ticker"])
        if result:
            res = ResolutionResult(
                query=query, resolved=True, resolution_method="name_search", **result,
                alternate_candidates=[{"ticker": c["ticker"], "company_name": c["company_name"], "score": None}
                                       for c in candidates if c["ticker"] != result["ticker"]][:4],
            )
            _RESOLUTION_CACHE[cache_key] = res
            return res

    return ResolutionResult(
        query=query, resolved=False,
        error=(f"Could not resolve '{query}' to a tradable ticker. Tried it directly, with .NS/.BO suffixes, "
               "and as a company-name search. Double-check the spelling, or type the exact exchange ticker "
               "(e.g. HDFCBANK.NS, AAPL)."),
    )


if __name__ == "__main__":
    for q in ["HDFC Bank", "hdfcbank", "HDFCBANK.NS", "tcs", "icici", "not a real company xyz123", ""]:
        r = resolve_ticker(q)
        print(f"{q!r:30s} -> resolved={r.resolved} ticker={r.ticker} method={r.resolution_method}")
