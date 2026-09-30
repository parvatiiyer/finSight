"""
News evidence agent.

Two modes, same output contract (Evidence objects):

  - Synthetic (demo): draws from injected ground-truth events for the
    curated universe — see src/data/synthetic_factors.py. Used for testing
    the adjudicator's logic against known-correct answers.

  - Production (gather_news_evidence_production): a REAL, working
    implementation using Google News' public RSS search endpoint, which
    needs no API key and works for any company name — this is what makes
    "any possible stock" actually have real news evidence instead of only
    working for the 25 curated demo tickers. Not reachable from this sandbox
    (news.google.com isn't in the network allowlist here, same class of
    restriction as yfinance's query1/query2 hosts elsewhere in this
    project), so it's written for correctness and needs verification on a
    machine with normal internet access.

Sentiment here is a lightweight keyword heuristic, not FinBERT — deliberately
kept simple and dependency-light (no torch/transformers download required to
get the pipeline working end-to-end). Swapping in FinBERT is a one-function
change: replace `_keyword_polarity` with a call to a transformers sentiment
pipeline; nothing else in this file or its callers needs to change, since
both return the same (Direction, magnitude_hint) shape.
"""

import sys
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agents.evidence_types import Evidence, Direction


# --- Lightweight keyword-based polarity (swap for FinBERT when available) --

_POSITIVE_WORDS = [
    "beat", "beats", "beating", "surge", "surges", "jump", "jumps", "rally", "rallies",
    "upgrade", "upgraded", "gain", "gains", "record profit", "strong", "outperform",
    "clears", "cleared", "approval", "approved", "optimistic", "raises guidance",
]
_NEGATIVE_WORDS = [
    "miss", "misses", "missing", "plunge", "plunges", "fall", "falls", "falling",
    "downgrade", "downgraded", "loss", "losses", "weak", "underperform", "probe",
    "investigation", "fraud", "concerns", "flags", "cautious", "cuts guidance", "lawsuit",
]


def _keyword_polarity(headline: str) -> tuple[Direction, float]:
    """
    Returns (Direction, magnitude_hint) from simple keyword matching.
    magnitude_hint scales with how many sentiment-bearing words matched —
    a headline hitting multiple negative keywords is treated as a bigger
    deal than one with a single mild keyword, which is a crude but
    defensible proxy in the absence of a real sentiment model.
    """
    text = headline.lower()
    pos_hits = sum(1 for w in _POSITIVE_WORDS if w in text)
    neg_hits = sum(1 for w in _NEGATIVE_WORDS if w in text)

    if pos_hits == 0 and neg_hits == 0:
        return Direction.NEUTRAL, 0.2
    if pos_hits > neg_hits:
        return Direction.POSITIVE, min(1.0, 0.35 + 0.2 * pos_hits)
    if neg_hits > pos_hits:
        return Direction.NEGATIVE, min(1.0, 0.35 + 0.2 * neg_hits)
    return Direction.NEUTRAL, 0.3  # equal hits both ways — genuinely mixed signal


# --- FinBERT sentiment analysis with lightweight fallback -------------------

_FINBERT_PIPELINE = None
_FINBERT_FAILED = False


def score_headline_sentiment_production(headline: str) -> tuple[Direction, float]:
    """
    Scores sentiment using ProsusAI/finbert via transformers if available/cached,
    falling back seamlessly to _keyword_polarity if offline or uninstalled.
    """
    global _FINBERT_PIPELINE, _FINBERT_FAILED
    if not _FINBERT_FAILED:
        if _FINBERT_PIPELINE is None:
            try:
                from transformers import pipeline
                _FINBERT_PIPELINE = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    top_k=None,
                )
            except Exception:
                _FINBERT_FAILED = True

        if _FINBERT_PIPELINE is not None:
            try:
                preds = _FINBERT_PIPELINE(headline)[0]
                best = max(preds, key=lambda x: x["score"])
                label_map = {
                    "positive": Direction.POSITIVE,
                    "negative": Direction.NEGATIVE,
                    "neutral": Direction.NEUTRAL,
                }
                direction = label_map.get(best["label"].lower(), Direction.NEUTRAL)
                magnitude = float(best["score"])
                return direction, magnitude
            except Exception:
                pass

    return _keyword_polarity(headline)


# --- Production path: Google News RSS, no API key ---------------------------


def _fetch_google_news_rss(company_name: str, timeout: int = 8) -> list[dict]:
    import requests
    import xml.etree.ElementTree as ET

    query = quote(f"{company_name} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link")          # NEW
        pub_date_raw = item.findtext("pubDate")
        source_el = item.find("source")
        source = source_el.text if source_el is not None else "Google News"
        try:
            published_at = parsedate_to_datetime(pub_date_raw).date() if pub_date_raw else None
        except Exception:
            published_at = None
        if title and published_at:
            items.append({"headline": title, "published_at": published_at, "source": source, "link": link})
    return items


def gather_news_evidence_production(
    ticker: str,
    company_name: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    lookback_buffer_days: int = 5,
    max_items: int = 15,
) -> list[Evidence]:
    try:
        raw_items = _fetch_google_news_rss(company_name)
    except Exception as e:
        print(f"[news_agent] Google News RSS fetch failed for '{company_name}': {e}")
        return []

    window_start_d = window_start.date() if hasattr(window_start, "date") else window_start
    window_end_d = window_end.date() if hasattr(window_end, "date") else window_end
    lookback_start = window_start_d - pd.Timedelta(days=lookback_buffer_days)

    evidence = []
    for item in raw_items[:max_items]:
        pub_date = item["published_at"]
        if not (lookback_start <= pub_date <= window_end_d):
            continue
        direction, magnitude = score_headline_sentiment_production(item["headline"])
        tag_sentiment = "finbert" if _FINBERT_PIPELINE is not None else "keyword_sentiment"
        source_name = item.get("source", "Google News")
        evidence.append(Evidence(
            source_agent="news",
            event_date=pub_date,
            direction=direction,
            magnitude_hint=magnitude,
            raw_text=f"{item['headline']} ({source_name})",
            reliability_weight=0.75,
            tags=["rss", tag_sentiment],
            source_url=item.get("link"),
            source_path=f"news_rss://{quote(source_name.lower().replace(' ', '_'))}/{pub_date}",
        ))
    return evidence

# --- Synthetic path (demo/testing, tied to ground truth) --------------------

def gather_news_evidence_synthetic(
    ticker: str,
    injected_events: list,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    n_noise_headlines: int = 2,
    seed: int | None = None,
) -> list[Evidence]:
    rng = np.random.default_rng(seed if seed is not None else hash(ticker) % (2**32))
    evidence = []
    lookback_buffer = pd.Timedelta(days=3)
    for event in injected_events:
        if window_start - lookback_buffer <= event.event_date <= window_end:
            evidence.append(Evidence(
                source_agent="news", event_date=event.event_date.date(),
                direction=Direction.NEGATIVE if event.direction == "negative" else Direction.POSITIVE,
                magnitude_hint=min(1.0, abs(event.magnitude) / 0.05),
                raw_text=event.headline, reliability_weight=1.0, tags=[event.tag],
                source_path=f"news_feed://ground_truth/{ticker}#{event.tag}",
            ))
    noise_templates = [
        "Analyst maintains neutral rating, no change to estimates",
        "Peer company announces unrelated product launch",
        "Broader market commentary mentions sector in passing",
    ]
    for _ in range(n_noise_headlines):
        offset_days = int(rng.integers(-15, -5))
        noise_date = (window_start + pd.Timedelta(days=offset_days)).date()
        evidence.append(Evidence(
            source_agent="news", event_date=noise_date,
            direction=rng.choice([Direction.POSITIVE, Direction.NEGATIVE, Direction.NEUTRAL]),
            magnitude_hint=float(rng.uniform(0.05, 0.25)),
            raw_text=str(rng.choice(noise_templates)), reliability_weight=0.6, tags=["low_relevance"],
        ))
    return evidence


if __name__ == "__main__":
    # Smoke test the keyword polarity function (no network needed)
    for h in ["Company beats estimates, shares surge", "Regulator flags concerns over compliance",
              "Company announces new office location"]:
        d, m = _keyword_polarity(h)
        print(f"{h!r:55s} -> {d.value:8s} magnitude={m:.2f}")
