"""Shared data contracts between evidence agents and the adjudicator."""
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

class Direction(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class Evidence:
    source_agent: str
    event_date: date
    direction: Direction
    magnitude_hint: float
    raw_text: str
    reliability_weight: float = 1.0
    tags: list[str] = field(default_factory=list)
    source_url: str | None = None   # link the UI can show alongside the claim
    source_path: str | None = None  # explicit source path / audit trail identifier

@dataclass
class MoveWindow:
    ticker: str
    start_date: date
    end_date: date
    raw_return: float
    market_beta: float
    sector_beta: float
    market_explained_return: float
    sector_explained_return: float
    residual_return: float
    residual_zscore: float
    has_sector_factor: bool = True  # False when no sector index was available (arbitrary ticker fallback)

    @property
    def systematic_fraction(self) -> float:
        if self.raw_return == 0:
            return 0.0
        return (self.market_explained_return + self.sector_explained_return) / self.raw_return

@dataclass
class EvidenceVerdict:
    evidence: Evidence
    temporal_alignment: float
    directional_match: bool
    magnitude_ok: bool
    accepted: bool
    rejection_reason: str | None = None
    contribution: float = 0.0  # Literal share of the final confidence score

@dataclass
class AdjudicationResult:
    move: MoveWindow
    verdicts: list
    corroboration_count: int
    confidence: float
    confidence_breakdown: dict
    unexplained: bool

    @property
    def accepted_evidence(self):
        return [v.evidence for v in self.verdicts if v.accepted]

    @property
    def rejected_evidence(self):
        return [v for v in self.verdicts if not v.accepted]
