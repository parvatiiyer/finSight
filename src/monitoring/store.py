"""
Persistence layer for FinSight move explanation runs using SQLite.
Stores MoveWindows, AdjudicationResults, Evidence, and narratives.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from src.agents.evidence_types import AdjudicationResult, MoveWindow, EvidenceVerdict

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "finsight_history.db"


class ExplanationStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS move_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    raw_return REAL NOT NULL,
                    market_beta REAL NOT NULL,
                    sector_beta REAL NOT NULL,
                    market_explained REAL NOT NULL,
                    sector_explained REAL NOT NULL,
                    residual_return REAL NOT NULL,
                    residual_zscore REAL NOT NULL,
                    has_sector_factor INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    unexplained INTEGER NOT NULL,
                    corroboration_count INTEGER NOT NULL,
                    confidence_breakdown TEXT,
                    narrative TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    move_id INTEGER NOT NULL,
                    source_agent TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    magnitude_hint REAL NOT NULL,
                    raw_text TEXT NOT NULL,
                    source_url TEXT,
                    accepted INTEGER NOT NULL,
                    rejection_reason TEXT,
                    contribution REAL NOT NULL DEFAULT 0.0,
                    temporal_alignment REAL NOT NULL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (move_id) REFERENCES move_runs (id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_moves_ticker ON move_runs(ticker);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_moves_dates ON move_runs(start_date, end_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_move ON evidence_records(move_id);")

    def save_run(self, result: AdjudicationResult, narrative: str | None = None) -> int:
        """Saves a MoveExplanation run and returns the assigned move_id."""
        m = result.move
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO move_runs (
                    ticker, start_date, end_date, raw_return, market_beta, sector_beta,
                    market_explained, sector_explained, residual_return, residual_zscore,
                    has_sector_factor, confidence, unexplained, corroboration_count,
                    confidence_breakdown, narrative
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.ticker, str(m.start_date), str(m.end_date), m.raw_return,
                m.market_beta, m.sector_beta, m.market_explained_return,
                m.sector_explained_return, m.residual_return, m.residual_zscore,
                1 if m.has_sector_factor else 0, result.confidence,
                1 if result.unexplained else 0, result.corroboration_count,
                json.dumps(result.confidence_breakdown), narrative or ""
            ))
            move_id = cursor.lastrowid

            for v in result.verdicts:
                cursor.execute("""
                    INSERT INTO evidence_records (
                        move_id, source_agent, event_date, direction, magnitude_hint,
                        raw_text, source_url, accepted, rejection_reason, contribution,
                        temporal_alignment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    move_id, v.evidence.source_agent, str(v.evidence.event_date),
                    v.evidence.direction.value, v.evidence.magnitude_hint,
                    v.evidence.raw_text, v.evidence.source_url,
                    1 if v.accepted else 0, v.rejection_reason,
                    v.contribution, v.temporal_alignment,
                ))
            return move_id

    def get_runs_for_ticker(self, ticker: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM move_runs WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
                (ticker, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_run_details(self, move_id: int) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            move = conn.execute("SELECT * FROM move_runs WHERE id = ?", (move_id,)).fetchone()
            if not move:
                return None
            res = dict(move)
            evidence_rows = conn.execute(
                "SELECT * FROM evidence_records WHERE move_id = ?", (move_id,)
            ).fetchall()
            res["evidence"] = [dict(e) for e in evidence_rows]
            return res
