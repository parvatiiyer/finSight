"""
Confidence calibration test harness and reliability metrics.

Validates that when the adjudicator outputs a confidence score of X%,
the explanation is historically/empirically correct ~X% of the time.

Metrics:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability Diagram data (predicted confidence vs observed accuracy)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence
import numpy as np
import pandas as pd

from src.agents.adjudicator import adjudicate
from src.agents.evidence_types import Direction, Evidence, MoveWindow


@dataclass
class CalibrationReport:
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    bucket_table: pd.DataFrame
    is_well_calibrated: bool


def compute_calibration_metrics(
    evaluations: Sequence[dict],
    n_buckets: int = 5,
    max_acceptable_ece: float = 0.25,
) -> CalibrationReport:
    """
    Computes calibration metrics across a collection of evaluation records.
    Each record must have:
        - "confidence": float in [0, 1]
        - "is_correct": bool (whether the attributed cause is genuine/correct)
    """
    if not evaluations:
        raise ValueError("Evaluations list cannot be empty")

    confidences = np.array([float(e["confidence"]) for e in evaluations])
    outcomes = np.array([1.0 if e["is_correct"] else 0.0 for e in evaluations])
    n_total = len(confidences)

    brier = float(np.mean((confidences - outcomes) ** 2))

    bins = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket_rows = []
    ece = 0.0
    mce = 0.0

    for i in range(n_buckets):
        low, high = bins[i], bins[i + 1]
        # Include left edge, include right edge on last bucket
        if i == n_buckets - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_conf = float(np.mean(confidences[mask]))
            empirical_acc = float(np.mean(outcomes[mask]))
            err = abs(empirical_acc - mean_conf)
            ece += (count / n_total) * err
            mce = max(mce, err)
        else:
            mean_conf = float((low + high) / 2)
            empirical_acc = float("nan")
            err = 0.0

        bucket_rows.append({
            "bucket_range": f"{low:.1f}-{high:.1f}",
            "n_samples": count,
            "mean_confidence": round(mean_conf, 3),
            "observed_accuracy": round(empirical_acc, 3) if not np.isnan(empirical_acc) else None,
            "calibration_gap": round(err, 3),
        })

    bucket_df = pd.DataFrame(bucket_rows)
    is_calibrated = ece <= max_acceptable_ece

    return CalibrationReport(
        brier_score=round(brier, 4),
        expected_calibration_error=round(ece, 4),
        maximum_calibration_error=round(mce, 4),
        bucket_table=bucket_df,
        is_well_calibrated=is_calibrated,
    )


def generate_synthetic_calibration_suite(
    n_moves: int = 120,
    seed: int = 42,
) -> list[dict]:
    """
    Generates a controlled synthetic test suite with known ground-truth events,
    evaluates them via adjudicate(), and labels correctness against ground-truth.
    """
    rng = np.random.default_rng(seed)
    evaluations = []

    for i in range(n_moves):
        start = date(2025, 1, 1) + timedelta(days=i * 2)
        end = start + timedelta(days=4)

        has_genuine_cause = rng.random() > 0.35
        true_direction = Direction.POSITIVE if rng.random() > 0.5 else Direction.NEGATIVE

        residual_z = (
            float(rng.uniform(1.2, 2.8) if true_direction == Direction.POSITIVE else rng.uniform(-2.8, -1.2))
            if has_genuine_cause
            else float(rng.uniform(-0.8, 0.8))
        )

        move = MoveWindow(
            ticker=f"STOCK_{i % 5}.NS",
            start_date=start,
            end_date=end,
            raw_return=residual_z * 0.02,
            market_beta=1.0,
            sector_beta=0.5,
            market_explained_return=0.005,
            sector_explained_return=0.005,
            residual_return=residual_z * 0.015,
            residual_zscore=residual_z,
            has_sector_factor=True,
        )

        evidence_list = []
        if has_genuine_cause:
            # Genuine aligned event
            event_date = start + timedelta(days=int(rng.integers(0, 3)))
            magnitude = float(rng.uniform(0.4, 0.9))
            evidence_list.append(Evidence(
                source_agent="news",
                event_date=event_date,
                direction=true_direction,
                magnitude_hint=magnitude,
                raw_text="Major regulatory clearance",
                reliability_weight=1.0,
            ))
            # 50% chance of second corroborating technical event
            if rng.random() > 0.5:
                evidence_list.append(Evidence(
                    source_agent="technical",
                    event_date=event_date,
                    direction=Direction.NEUTRAL,
                    magnitude_hint=0.6,
                    raw_text="Volume breakout",
                    reliability_weight=0.9,
                ))

        # Add noise headlines (off-topic, low magnitude, or mistimed)
        n_noise = int(rng.integers(0, 3))
        for _ in range(n_noise):
            noise_offset = int(rng.choice([-12, -10, 10, 14]))
            evidence_list.append(Evidence(
                source_agent="news",
                event_date=start + timedelta(days=noise_offset),
                direction=rng.choice([Direction.POSITIVE, Direction.NEGATIVE]),
                magnitude_hint=float(rng.uniform(0.05, 0.25)),
                raw_text="Random unrelated sector commentary",
                reliability_weight=0.5,
            ))

        result = adjudicate(move, evidence_list)

        # Ground truth correctness definition:
        # If there was a genuine cause and the adjudicator accepted the genuine cause -> correct
        # If there was no cause and the adjudicator declared unexplained / confidence 0 -> correct
        # If noisy cause accepted -> incorrect
        accepted = [v for v in result.verdicts if v.accepted]
        if has_genuine_cause:
            is_correct = any(v.evidence.raw_text == "Major regulatory clearance" for v in accepted)
        else:
            is_correct = (len(accepted) == 0)

        evaluations.append({
            "ticker": move.ticker,
            "window": f"{start} to {end}",
            "confidence": result.confidence,
            "has_genuine_cause": has_genuine_cause,
            "is_correct": is_correct,
            "unexplained": result.unexplained,
        })

    return evaluations
