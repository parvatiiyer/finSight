"""
Tests for RAG over annual report filings and filing evidence agent.
"""
from datetime import date
import pytest

from src.agents.evidence_types import Direction
from src.agents.filing_agent import gather_filing_evidence
from src.rag.filing_index import FilingChunk, FilingRetriever, get_default_filing_retriever


def test_filing_retriever_indexing_and_search(tmp_path):
    # Initialize a retriever in a temporary directory
    retriever = FilingRetriever(collection_name="test_filings", persist_dir=str(tmp_path / "chroma"))

    chunks = [
        FilingChunk(
            ticker="TEST.NS",
            fiscal_year=2025,
            section="MD&A",
            chunk_id="test_mda_1",
            text="Operating margins contracted by 180 bps due to elevated wage hikes and input costs.",
        ),
        FilingChunk(
            ticker="TEST.NS",
            fiscal_year=2025,
            section="Outlook",
            chunk_id="test_outlook_1",
            text="Order book reached all-time high of 40 billion dollars driven by cloud migration contracts.",
        ),
    ]
    retriever.index_chunks(chunks)

    # Query for margin pressure
    margin_results = retriever.search("operating margins contracted cost pressure", ticker="TEST.NS", top_k=1)
    assert len(margin_results) == 1
    chunk, score = margin_results[0]
    assert chunk.chunk_id == "test_mda_1"
    assert "Operating margins contracted" in chunk.text
    assert score > 0.0

    # Query for order book
    order_results = retriever.search("order book cloud expansion growth", ticker="TEST.NS", top_k=1)
    assert len(order_results) == 1
    assert order_results[0][0].chunk_id == "test_outlook_1"


def test_gather_filing_evidence_negative_residual():
    # Residual z-score is negative -> expect negative/risk evidence
    evidence = gather_filing_evidence(
        ticker="HDFCBANK.NS",
        event_date=date(2025, 4, 15),
        residual_zscore=-2.2,
    )
    assert len(evidence) > 0
    top = evidence[0]
    assert top.source_agent == "filing"
    assert top.direction == Direction.NEGATIVE
    assert top.source_path is not None
    assert "filings://annual_report/HDFCBANK.NS/" in top.source_path
    assert "#" in top.source_path
    assert top.magnitude_hint >= 0.3
    assert "HDFC Bank" in top.raw_text or "NIM" in top.raw_text or "margin" in top.raw_text or "deposit" in top.raw_text


def test_gather_filing_evidence_positive_residual():
    # Residual z-score is positive -> expect positive evidence
    evidence = gather_filing_evidence(
        ticker="TCS.NS",
        event_date=date(2025, 6, 20),
        residual_zscore=2.5,
    )
    assert len(evidence) > 0
    top = evidence[0]
    assert top.source_agent == "filing"
    assert top.direction == Direction.POSITIVE
    assert top.source_path is not None
    assert "filings://annual_report/TCS.NS/" in top.source_path
    assert top.magnitude_hint >= 0.3


def test_default_filing_retriever_singleton():
    r1 = get_default_filing_retriever()
    r2 = get_default_filing_retriever()
    assert r1 is r2
