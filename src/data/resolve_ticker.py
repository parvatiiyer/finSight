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
        # Word-boundary containment only — avoids char-level anagram false
        # positives like "nvidia" ~ "nestle india" that pure SequenceMatcher
        # ratio can't distinguish from a real substring/name match.
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

def _validate_symbol(symbol: str) -> dict | None:
    try:
        t = yf.Ticker(symbol)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            return None
        return {
            "ticker": symbol,
            "company_name": info.get("longName") or info.get("shortName") or symbol,
            "exchange": info.get("exchange"),
            "last_price": float(price),
        }
    except Exception:
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


def resolve_ticker(query: str) -> ResolutionResult:
    query = (query or "").strip()
    if not query:
        return ResolutionResult(query=query, resolved=False, error="Empty input — type a company name or ticker symbol.")

    curated = _curated_lookup(query)
    if curated:
        return curated

    direct = _direct_validation(query)
    if direct:
        return direct

    candidates = _search_by_name(query)
    for candidate in candidates:
        result = _validate_symbol(candidate["ticker"])
        if result:
            return ResolutionResult(
                query=query, resolved=True, resolution_method="name_search", **result,
                alternate_candidates=[{"ticker": c["ticker"], "company_name": c["company_name"], "score": None}
                                       for c in candidates if c["ticker"] != result["ticker"]][:4],
            )

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
