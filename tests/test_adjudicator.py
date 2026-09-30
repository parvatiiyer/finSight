"""
Unit tests and property-based fuzz tests for the deterministic adjudicator.
Tests:
- The three gates (temporal alignment, magnitude adequacy, directional consistency)
- Confidence formula and corroboration bonus (independent agent types vs raw count)
- Per-verdict contribution calculation
- Invariants fuzzing across random MoveWindow and Evidence combinations
"""
from datetime import date, timedelta
import numpy as np
import pytest

from src.agents.evidence_types import Direction, Evidence, MoveWindow, EvidenceVerdict
from src.agents.adjudicator import (
    temporal_alignment_score,
    magnitude_adequate,
    directional_consistency,
    adjudicate,
)


def _make_move(
    start_date=date(2025, 3, 1),
    end_date=date(2025, 3, 5),
    raw_return=0.05,
    residual_zscore=1.5,
    residual_return=0.03,
    market_beta=1.0,
    sector_beta=0.5,
):
    return MoveWindow(
        ticker="TEST.NS",
        start_date=start_date,
        end_date=end_date,
        raw_return=raw_return,
        market_beta=market_beta,
        sector_beta=sector_beta,
        market_explained_return=0.01,
        sector_explained_return=0.01,
        residual_return=residual_return,
        residual_zscore=residual_zscore,
        has_sector_factor=True,
    )


def test_temporal_alignment_gate():
    move = _make_move()
    # Exactly inside window
    ev_inside = Evidence("news", date(2025, 3, 3), Direction.POSITIVE, 0.5, "Inside window")
    assert temporal_alignment_score(ev_inside, move) == 1.0

    # 1 day before start
    ev_day_before = Evidence("news", date(2025, 2, 28), Direction.POSITIVE, 0.5, "1 day before")
    assert temporal_alignment_score(ev_day_before, move, decay_days=5) == 0.8

    # 5 days before start (at threshold)
    ev_5_days_before = Evidence("news", date(2025, 2, 24), Direction.POSITIVE, 0.5, "5 days before")
    assert temporal_alignment_score(ev_5_days_before, move, decay_days=5) == 0.0

    # Way out of window
    ev_way_off = Evidence("news", date(2025, 1, 1), Direction.POSITIVE, 0.5, "Far before")
    assert temporal_alignment_score(ev_way_off, move) == 0.0


def test_magnitude_adequate_gate():
    ev_small = Evidence("news", date(2025, 3, 3), Direction.POSITIVE, 0.25, "Small hint")
    assert not magnitude_adequate(ev_small, min_magnitude=0.30)

    ev_exact = Evidence("news", date(2025, 3, 3), Direction.POSITIVE, 0.30, "Borderline hint")
    assert magnitude_adequate(ev_exact, min_magnitude=0.30)

    ev_large = Evidence("news", date(2025, 3, 3), Direction.POSITIVE, 0.80, "Large hint")
    assert magnitude_adequate(ev_large, min_magnitude=0.30)


def test_directional_consistency_gate():
    pos_move = _make_move(residual_zscore=1.8)
    neg_move = _make_move(residual_zscore=-1.8)
    flat_move = _make_move(residual_zscore=0.2)  # inside dead-zone (|z| < 0.5)

    pos_ev = Evidence("news", date(2025, 3, 3), Direction.POSITIVE, 0.6, "Positive news")
    neg_ev = Evidence("news", date(2025, 3, 3), Direction.NEGATIVE, 0.6, "Negative news")
    neutral_ev = Evidence("technical", date(2025, 3, 3), Direction.NEUTRAL, 0.6, "Neutral technical")

    # Positive move accepts positive, rejects negative, accepts neutral
    assert directional_consistency(pos_ev, pos_move)
    assert not directional_consistency(neg_ev, pos_move)
    assert directional_consistency(neutral_ev, pos_move)

    # Negative move rejects positive, accepts negative, accepts neutral
    assert not directional_consistency(pos_ev, neg_move)
    assert directional_consistency(neg_ev, neg_move)
    assert directional_consistency(neutral_ev, neg_move)

    # Flat move in dead zone: rejects directional evidence, accepts neutral
    assert not directional_consistency(pos_ev, flat_move)
    assert not directional_consistency(neg_ev, flat_move)
    assert directional_consistency(neutral_ev, flat_move)


def test_corroboration_distinct_agents_vs_raw_count():
    move = _make_move(residual_zscore=2.0)

    # 4 items from the same agent type ("news")
    same_agent_evidence = [
        Evidence("news", date(2025, 3, 2), Direction.POSITIVE, 0.5, f"News item {i}")
        for i in range(4)
    ]
    res_same = adjudicate(move, same_agent_evidence)
    assert res_same.corroboration_count == 1
    assert res_same.confidence_breakdown["corroboration_bonus"] == 0.0

    # 2 items from 2 DIFFERENT agent types ("news", "fundamentals")
    multi_agent_evidence = [
        Evidence("news", date(2025, 3, 2), Direction.POSITIVE, 0.5, "News item"),
        Evidence("fundamentals", date(2025, 3, 2), Direction.POSITIVE, 0.5, "Quarterly ratio delta"),
    ]
    res_multi = adjudicate(move, multi_agent_evidence)
    assert res_multi.corroboration_count == 2
    assert res_multi.confidence_breakdown["corroboration_bonus"] == 0.15


def test_unexplained_flag():
    move_sig = _make_move(residual_zscore=2.5)  # significant move (|z| >= 1.0)
    # Empty evidence
    res_empty = adjudicate(move_sig, [])
    assert res_empty.unexplained is True
    assert res_empty.confidence == 0.0

    # Insignificant move (|z| < 1.0) with no evidence is NOT unexplained (noise)
    move_insig = _make_move(residual_zscore=0.4)
    res_insig = adjudicate(move_insig, [])
    assert res_insig.unexplained is False
    assert res_insig.confidence == 1.0


def test_contribution_calculation():
    move = _make_move(residual_zscore=1.5)
    ev1 = Evidence("news", date(2025, 3, 3), Direction.POSITIVE, 0.6, "Positive event")
    ev2 = Evidence("fundamentals", date(2025, 3, 3), Direction.POSITIVE, 0.4, "Positive ratio")
    ev_rej = Evidence("news", date(2024, 1, 1), Direction.NEGATIVE, 0.1, "Off-topic noise")

    res = adjudicate(move, [ev1, ev2, ev_rej])
    accepted = [v for v in res.verdicts if v.accepted]
    rejected = [v for v in res.verdicts if not v.accepted]

    assert len(accepted) == 2
    assert len(rejected) == 1

    # Rejected contribution must be 0
    assert rejected[0].contribution == 0.0
    assert rejected[0].rejection_reason is not None

    # Accepted contributions must be positive and sum roughly to confidence
    total_contrib = sum(v.contribution for v in accepted)
    assert abs(total_contrib - res.confidence) <= 0.01
    assert accepted[0].contribution > accepted[1].contribution  # higher magnitude contributes more


def test_adjudicator_invariants_fuzz():
    """Randomized property checks to verify invariants hold for arbitrary inputs."""
    rng = np.random.default_rng(42)
    agents = ["news", "fundamentals", "technical"]
    directions = [Direction.POSITIVE, Direction.NEGATIVE, Direction.NEUTRAL]

    for _ in range(200):
        start = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 100)))
        end = start + timedelta(days=int(rng.integers(1, 10)))
        z = float(rng.uniform(-3.5, 3.5))
        move = _make_move(start_date=start, end_date=end, residual_zscore=z)

        n_ev = int(rng.integers(0, 8))
        ev_list = []
        for j in range(n_ev):
            offset = int(rng.integers(-15, 15))
            ev_date = start + timedelta(days=offset)
            ev_list.append(Evidence(
                source_agent=rng.choice(agents),
                event_date=ev_date,
                direction=rng.choice(directions),
                magnitude_hint=float(rng.uniform(0.05, 1.0)),
                raw_text=f"Sample headline {j}",
                reliability_weight=float(rng.uniform(0.5, 1.0)),
            ))

        res = adjudicate(move, ev_list)

        # Invariant 1: Confidence is always in [0.0, 1.0]
        assert 0.0 <= res.confidence <= 1.0, f"Confidence {res.confidence} outside [0, 1]"

        # Invariant 2: Unexplained is never True if evidence was accepted
        accepted = [v for v in res.verdicts if v.accepted]
        if len(accepted) > 0:
            assert res.unexplained is False, "Unexplained was True even though evidence was accepted!"

        # Invariant 3: Rejected evidence always has a non-empty rejection reason
        for v in res.verdicts:
            if not v.accepted:
                assert v.rejection_reason and len(v.rejection_reason.strip()) > 0
                assert v.contribution == 0.0
            else:
                assert v.contribution >= 0.0
