"""The deterministic evidence adjudicator (see earlier turns for full design rationale)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agents.evidence_types import Direction, MoveWindow, EvidenceVerdict, AdjudicationResult

def temporal_alignment_score(evidence, move: MoveWindow, decay_days: int = 5) -> float:
    if move.start_date <= evidence.event_date <= move.end_date:
        return 1.0
    if evidence.event_date < move.start_date:
        days_off = (move.start_date - evidence.event_date).days
    else:
        days_off = (evidence.event_date - move.end_date).days
    return max(0.0, 1.0 - days_off / decay_days)

def magnitude_adequate(evidence, min_magnitude: float = 0.30) -> bool:
    return evidence.magnitude_hint >= min_magnitude

def directional_consistency(evidence, move: MoveWindow, dead_zone_z: float = 0.5) -> bool:
    if evidence.direction == Direction.NEUTRAL:
        return True
    if abs(move.residual_zscore) < dead_zone_z:
        return False
    residual_is_positive = move.residual_zscore > 0
    return (evidence.direction == Direction.POSITIVE) == residual_is_positive

def adjudicate(move, evidence_list, min_magnitude=0.30, temporal_decay_days=5, dead_zone_z=0.5, significance_z=1.0):
    verdicts = []
    for ev in evidence_list:
        t_score = temporal_alignment_score(ev, move, decay_days=temporal_decay_days)
        mag_ok = magnitude_adequate(ev, min_magnitude=min_magnitude)
        dir_ok = directional_consistency(ev, move, dead_zone_z=dead_zone_z)
        temporal_ok = t_score >= 0.4
        accepted = temporal_ok and mag_ok and dir_ok
        reason = None
        if not accepted:
            reasons = []
            if not temporal_ok:
                reasons.append(f"event_date {ev.event_date} too far from window [{move.start_date}, {move.end_date}]")
            if not mag_ok:
                reasons.append(f"magnitude_hint {ev.magnitude_hint:.2f} below floor {min_magnitude}")
            if not dir_ok:
                reasons.append(f"direction {ev.direction.value} inconsistent with residual (z={move.residual_zscore:+.2f})")
            reason = "; ".join(reasons)
        verdicts.append(EvidenceVerdict(evidence=ev, temporal_alignment=t_score, directional_match=dir_ok,
                                         magnitude_ok=mag_ok, accepted=accepted, rejection_reason=reason))

    accepted_verdicts = [v for v in verdicts if v.accepted]
    directional_strength = 0.0
    raw_weights = []
    for v in accepted_verdicts:
        weight = v.evidence.reliability_weight * v.temporal_alignment * v.evidence.magnitude_hint
        if v.evidence.direction == Direction.NEUTRAL:
            weight *= 0.5
        raw_weights.append(weight)
        directional_strength += weight
    directional_strength = min(1.0, directional_strength)

    corroborating_agent_types = {v.evidence.source_agent for v in accepted_verdicts}
    corroboration_count = len(corroborating_agent_types)
    corroboration_bonus = min(0.15 * max(0, corroboration_count - 1), 0.30)
    confidence = min(1.0, directional_strength * (1 + corroboration_bonus))

    move_is_significant = abs(move.residual_zscore) >= significance_z
    unexplained = move_is_significant and len(accepted_verdicts) == 0
    if not move_is_significant:
        confidence = 1.0 if len(accepted_verdicts) == 0 else confidence
        unexplained = False

    confidence_rounded = round(confidence, 3)
    total_raw_weight = sum(raw_weights)
    for v, w in zip(accepted_verdicts, raw_weights):
        v.contribution = round((w / total_raw_weight) * confidence_rounded, 3) if total_raw_weight > 0 else 0.0
    for v in verdicts:
        if not v.accepted:
            v.contribution = 0.0

    breakdown = {
        "systematic_fraction": round(move.systematic_fraction, 3),
        "residual_zscore": round(move.residual_zscore, 3),
        "move_is_significant": move_is_significant,
        "directional_strength": round(directional_strength, 3),
        "corroboration_count": corroboration_count,
        "corroboration_bonus": round(corroboration_bonus, 3),
        "n_evidence_considered": len(evidence_list),
        "n_evidence_accepted": len(accepted_verdicts),
        "n_evidence_rejected": len(evidence_list) - len(accepted_verdicts),
        "has_sector_factor": move.has_sector_factor,
    }
    return AdjudicationResult(move=move, verdicts=verdicts, corroboration_count=corroboration_count,
                               confidence=round(confidence, 3), confidence_breakdown=breakdown, unexplained=unexplained)
