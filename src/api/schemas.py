"""
Pydantic contracts for FinSight API endpoints and frontend consumption.
Ensures frontend expectations (has_sector_factor, contribution, temporal_alignment, etc.)
are strongly validated and cannot silently diverge.
"""

from typing import Any
from pydantic import BaseModel, Field


class EvidenceItemResponse(BaseModel):
    source_agent: str
    event_date: str
    direction: str | None = None
    magnitude_hint: float | None = None
    raw_text: str
    source_url: str | None = None
    source_path: str | None = None
    contribution: float = 0.0
    temporal_alignment: float = 1.0
    rejection_reason: str | None = None


class ConfidenceBreakdown(BaseModel):
    systematic_fraction: float
    residual_zscore: float
    move_is_significant: bool
    directional_strength: float
    corroboration_count: int
    corroboration_bonus: float
    n_evidence_considered: int
    n_evidence_accepted: int
    n_evidence_rejected: int
    has_sector_factor: bool = True


class ExplainResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    raw_return: float
    market_beta: float
    sector_beta: float
    market_explained_return: float
    sector_explained_return: float
    residual_return: float
    residual_zscore: float
    systematic_fraction: float
    has_sector_factor: bool
    confidence: float
    unexplained: bool
    corroboration_count: int
    confidence_breakdown: ConfidenceBreakdown
    accepted_evidence: list[EvidenceItemResponse]
    rejected_evidence: list[EvidenceItemResponse]
    narrative: str
    resolved_from_query: str | None = None
    resolution_method: str | None = None


class ResolveResponse(BaseModel):
    query: str
    ticker: str
    company_name: str
    exchange: str
    resolution_method: str
    alternate_candidates: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
