"""
Tests for persistence layer and monitoring loop graph tracking.
"""
from datetime import date
import pytest

from src.agents.evidence_types import AdjudicationResult, Direction, Evidence, MoveWindow
from src.agents.adjudicator import adjudicate
from src.monitoring.store import ExplanationStore
from src.monitoring.graph_tracker import MonitoringGraph


def _make_test_adjudication(ticker="HDFCBANK.NS", residual_zscore=1.5, ev_dir=Direction.POSITIVE, ev_text="Earnings surge"):
    move = MoveWindow(
        ticker=ticker,
        start_date=date(2025, 4, 1),
        end_date=date(2025, 4, 5),
        raw_return=0.04,
        market_beta=1.2,
        sector_beta=0.8,
        market_explained_return=0.01,
        sector_explained_return=0.01,
        residual_return=0.02,
        residual_zscore=residual_zscore,
        has_sector_factor=True,
    )
    ev1 = Evidence("news", date(2025, 4, 2), ev_dir, 0.7, ev_text)
    ev2 = Evidence("technical", date(2025, 4, 3), Direction.NEUTRAL, 0.5, "Volume breakout")
    return adjudicate(move, [ev1, ev2])


def test_sqlite_store_roundtrip(tmp_path):
    db_file = tmp_path / "test_history.db"
    store = ExplanationStore(db_path=db_file)

    adj = _make_test_adjudication()
    move_id = store.save_run(adj, narrative="Test narrative for HDFC Bank")
    assert move_id > 0

    runs = store.get_runs_for_ticker("HDFCBANK.NS")
    assert len(runs) == 1
    assert runs[0]["ticker"] == "HDFCBANK.NS"
    assert runs[0]["narrative"] == "Test narrative for HDFC Bank"

    details = store.get_run_details(move_id)
    assert details is not None
    assert len(details["evidence"]) == 2
    assert any(e["source_agent"] == "news" for e in details["evidence"])


def test_monitoring_graph_ingest_and_corroboration():
    mg = MonitoringGraph()
    adj = _make_test_adjudication()
    move_id = mg.ingest_run(adj, run_id="run_1", narrative="First run")

    assert move_id in mg.g.nodes
    # Check that corroboration edge exists between news and technical
    edges = list(mg.g.edges(data=True))
    corrob_edges = [d for _, _, d in edges if d.get("type") == "CORROBORATES"]
    assert len(corrob_edges) >= 2  # bidirectional


def test_monitoring_graph_contradiction_detection():
    mg = MonitoringGraph()

    # Step 1: Initial move explained by NPA / probe concerns
    adj_prior = _make_test_adjudication(
        residual_zscore=-1.8,
        ev_dir=Direction.NEGATIVE,
        ev_text="Regulator initiates probe into NPA classification",
    )
    id1 = mg.ingest_run(adj_prior, run_id="run_probe", narrative="Drop driven by probe")

    # Step 2: Subsequent move where regulator clears bank of probe
    adj_new = _make_test_adjudication(
        residual_zscore=2.1,
        ev_dir=Direction.POSITIVE,
        ev_text="Regulator probe cleared and concluded with no penalties",
    )
    id2 = mg.ingest_run(adj_new, run_id="run_cleared", narrative="Rally on cleared probe")

    # Contradiction should be detected
    contradicted = mg.get_contradicted_moves("HDFCBANK.NS")
    assert len(contradicted) == 1
    assert "probe" in contradicted[0]["invalidated_evidence"]
    assert "cleared" in contradicted[0]["contradicting_evidence"]

    # Diff explanations
    diff = mg.diff_explanations(id1, id2)
    assert "confidence_delta" in diff
    assert diff["prior_confidence"] == adj_prior.confidence
    assert diff["new_confidence"] == adj_new.confidence


def test_monitoring_graph_cypher_export(tmp_path):
    mg = MonitoringGraph()
    adj = _make_test_adjudication(ticker="HDFCBANK.NS", residual_zscore=1.8, ev_dir=Direction.POSITIVE, ev_text="Earnings surge")
    mg.ingest_run(adj, run_id="run_cypher_test", narrative="Rally on earnings")

    cypher = mg.export_to_cypher()
    assert "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company)" in cypher
    assert "MERGE (c:Company {ticker: \"HDFCBANK.NS\"})" in cypher
    assert "MERGE (m:MoveWindow" in cypher
    assert "MERGE (e:Evidence" in cypher
    assert "-[r:HAD_MOVE]->" in cypher
    assert "-[r:EXPLAINED_BY" in cypher

    export_file = tmp_path / "test_export.cypher"
    mg.save_cypher_export(str(export_file))
    assert export_file.exists()
    assert export_file.read_text(encoding="utf-8") == cypher

