"""
Contract tests between FastAPI responses, Pydantic schemas, and frontend assumptions.
Checks that all required fields (e.g. has_sector_factor, contribution, temporal_alignment, etc.)
are present and strictly validated.
"""
from fastapi.testclient import TestClient
import pytest

from src.api.main import app
from src.api.schemas import ExplainResponse, ResolveResponse, HealthResponse

client = TestClient(app)


def test_health_contract():
    res = client.get("/health")
    assert res.status_code == 200
    data = HealthResponse(**res.json())
    assert data.status == "ok"


def test_resolve_contract():
    # Indian stock in curated universe
    res = client.get("/resolve?query=HDFC Bank")
    assert res.status_code == 200
    data = ResolveResponse(**res.json())
    assert data.ticker == "HDFCBANK.NS"
    assert "HDFC" in data.company_name
    assert data.resolution_method

    # Apple resolution (query='apple', 'aapl', 'Apple Inc.')
    for q in ["apple", "aapl", "Apple", "Apple Inc."]:
        res_apple = client.get(f"/resolve?query={q}")
        assert res_apple.status_code == 200
        data_apple = ResolveResponse(**res_apple.json())
        assert data_apple.ticker == "AAPL"
        assert "Apple" in data_apple.company_name
        assert data_apple.exchange == "NASDAQ"

    # Other global mega-caps
    res_nvda = client.get("/resolve?query=nvidia")
    assert res_nvda.status_code == 200
    assert ResolveResponse(**res_nvda.json()).ticker == "NVDA"


def test_explain_demo_contract():
    res = client.get("/explain_demo?ticker=HDFCBANK.NS")
    assert res.status_code == 200
    raw_json = res.json()

    # Validate against strict Pydantic model
    validated = ExplainResponse(**raw_json)

    # Specific fields required by frontend/dashboard.html
    assert validated.ticker == "HDFCBANK.NS"
    assert isinstance(validated.has_sector_factor, bool)
    assert 0.0 <= validated.confidence <= 1.0
    assert isinstance(validated.unexplained, bool)
    assert validated.confidence_breakdown.move_is_significant in (True, False)
    assert validated.confidence_breakdown.has_sector_factor in (True, False)

    # Check evidence contract
    for ev in validated.accepted_evidence:
        assert ev.source_agent in ("news", "fundamentals", "technical", "filing")
        assert ev.event_date
        assert 0.0 <= ev.temporal_alignment <= 1.0
        assert ev.contribution >= 0.0
        assert ev.source_path is not None

    for ev in validated.rejected_evidence:
        assert ev.rejection_reason is not None
        assert ev.contribution == 0.0


def test_history_and_peers_endpoints():
    # After running demo, history should contain records
    res_hist = client.get("/history?ticker=HDFCBANK.NS")
    assert res_hist.status_code == 200
    hist_json = res_hist.json()
    assert hist_json["ticker"] == "HDFCBANK.NS"
    assert hist_json["total_runs"] >= 1

    # Peers endpoint
    res_peers = client.get("/peers?ticker=HDFCBANK.NS")
    assert res_peers.status_code == 200
    peers_json = res_peers.json()
    assert peers_json["sector"] == "Banking"
    assert "ICICIBANK.NS" in peers_json["peers"]
