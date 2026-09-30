"""
Filing Evidence Agent — extracts causal disclosures and risk factors from Annual Reports (10-K/MD&A).
"""
import sys
from datetime import date
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agents.evidence_types import Evidence, Direction
from src.rag.filing_index import FilingRetriever, get_default_filing_retriever


def gather_filing_evidence(
    ticker: str,
    window_start: pd.Timestamp | date | None = None,
    window_end: pd.Timestamp | date | None = None,
    residual_zscore: float = 0.0,
    retriever: FilingRetriever | None = None,
    min_similarity: float = 0.20,
    event_date: pd.Timestamp | date | None = None,
) -> list[Evidence]:
    """
    Queries vector-indexed annual report sections for management disclosures,
    risk factors, and segment guidance that corroborate or contextualize a stock move.
    """
    if retriever is None:
        retriever = get_default_filing_retriever()

    # Resolve date
    effective_date = event_date or window_start or pd.Timestamp.now()
    if hasattr(effective_date, "date"):
        effective_date = effective_date.date()

    # Formulate domain query based on idiosyncratic residual direction
    if residual_zscore <= -0.5:
        query_text = "margin compression revenue slowdown discretionary spend cuts credit slippage headwinds risk"
        expected_direction = Direction.NEGATIVE
    elif residual_zscore >= 0.5:
        query_text = "order book deal wins margin expansion revenue growth acceleration asset quality resilience"
        expected_direction = Direction.POSITIVE
    else:
        query_text = "forward guidance segment performance credit deposit ratio capital adequacy"
        expected_direction = Direction.NEUTRAL

    try:
        matches = retriever.query(query_text=query_text, ticker=ticker, top_k=2)
    except Exception as e:
        print(f"[filing_agent] RAG retrieval failed for {ticker}: {e}")
        return []

    evidence_items = []
    window_start_d = effective_date

    for chunk, similarity in matches:
        if similarity < min_similarity:
            continue

        # Map sentiment_hint to Direction
        if chunk.sentiment_hint == "negative":
            direction = Direction.NEGATIVE
        elif chunk.sentiment_hint == "positive":
            direction = Direction.POSITIVE
        else:
            direction = Direction.NEUTRAL

        # Normalize magnitude hint to a meaningful scale [0.35, 0.90]
        magnitude = float(min(0.95, max(0.35, similarity * 1.2)))
        section_clean = chunk.section.replace(" ", "_").lower()

        evidence_items.append(Evidence(
            source_agent="filing",
            event_date=window_start_d,
            direction=direction,
            magnitude_hint=magnitude,
            raw_text=f"Annual Report ({chunk.fiscal_year}) [{chunk.section}]: {chunk.content}",
            reliability_weight=0.85,
            tags=["annual_report", "rag_disclosure", section_clean],
            source_path=f"filings://annual_report/{ticker}/{chunk.fiscal_year}#{section_clean}",
        ))

    return evidence_items
