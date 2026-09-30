"""
Unit and integration tests for the confidence calibration harness (Section 9.1).
"""
import pytest
from src.evaluation.calibration import (
    compute_calibration_metrics,
    generate_synthetic_calibration_suite,
)


def test_calibration_perfect_case():
    evals = [
        {"confidence": 0.9, "is_correct": True},
        {"confidence": 0.9, "is_correct": True},
        {"confidence": 0.1, "is_correct": False},
        {"confidence": 0.1, "is_correct": False},
    ]
    report = compute_calibration_metrics(evals, n_buckets=2)
    assert report.is_well_calibrated
    assert report.expected_calibration_error <= 0.15
    assert report.brier_score <= 0.05


def test_synthetic_calibration_harness_evaluation():
    suite = generate_synthetic_calibration_suite(n_moves=80, seed=123)
    assert len(suite) == 80

    report = compute_calibration_metrics(suite, n_buckets=5)
    assert report.brier_score >= 0.0
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert not report.bucket_table.empty

    # Assert that high-confidence predictions have higher observed accuracy than low-confidence
    buckets = report.bucket_table.dropna(subset=["observed_accuracy"])
    if len(buckets) >= 2:
        lowest_bucket_acc = buckets.iloc[0]["observed_accuracy"]
        highest_bucket_acc = buckets.iloc[-1]["observed_accuracy"]
        assert highest_bucket_acc >= lowest_bucket_acc


def test_calibration_empty_raises():
    with pytest.raises(ValueError):
        compute_calibration_metrics([])
