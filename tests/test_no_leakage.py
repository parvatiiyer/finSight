"""
Tests for walk-forward CV: boundary correctness and no train/test boundary leakage.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.walk_forward import WalkForwardCV, check_no_leakage


def test_walk_forward_basic_split_ordering():
    n_days = 200
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    df = pd.DataFrame({"Close": np.random.randn(n_days)}, index=dates)

    cv = WalkForwardCV(train_size=60, test_size=20, step_size=20, purge_size=5)
    splits = list(cv.split(df))

    assert len(splits) > 0

    for train_idx, test_idx in splits:
        assert len(train_idx) == 60
        assert len(test_idx) == 20
        # No index overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        # Causal temporal order
        assert np.max(train_idx) < np.min(test_idx)
        # Purge gap
        assert np.min(test_idx) - np.max(train_idx) == 6  # purge_size + 1 in 0-indexed spacing
        # Check date order
        assert dates[np.max(train_idx)] < dates[np.min(test_idx)]
        # Use helper check
        assert check_no_leakage(train_idx, test_idx, dates=dates, label_horizon=5)


def test_walk_forward_expanding_window():
    n_days = 150
    df = pd.DataFrame({"Close": np.random.randn(n_days)})

    cv = WalkForwardCV(train_size=50, test_size=25, step_size=25, expanding=True)
    splits = list(cv.split(df))

    assert len(splits) >= 3
    # Check that train sizes expand
    train_sizes = [len(train) for train, _ in splits]
    assert train_sizes == [50, 75, 100, 125]
    for train_idx, _ in splits:
        assert train_idx[0] == 0


def test_walk_forward_purge_prevents_label_leakage():
    n_days = 100
    df = pd.DataFrame({"Close": np.random.randn(n_days)})

    label_horizon = 7
    # Insufficient purge should raise in check_no_leakage
    cv_leaky = WalkForwardCV(train_size=50, test_size=10, purge_size=2)
    train_idx, test_idx = next(cv_leaky.split(df))
    with pytest.raises(ValueError, match="Purge window insufficient"):
        check_no_leakage(train_idx, test_idx, label_horizon=label_horizon)

    # Adequate purge should pass
    cv_safe = WalkForwardCV(train_size=50, test_size=10, purge_size=8)
    train_idx, test_idx = next(cv_safe.split(df))
    assert check_no_leakage(train_idx, test_idx, label_horizon=label_horizon)


def test_walk_forward_edge_cases():
    df = pd.DataFrame({"Close": range(10)})
    # Test size larger than remaining data yields no splits
    cv = WalkForwardCV(train_size=8, test_size=5)
    assert list(cv.split(df)) == []

    # Invalid arguments
    with pytest.raises(ValueError):
        WalkForwardCV(train_size=0, test_size=10)
    with pytest.raises(ValueError):
        WalkForwardCV(train_size=10, test_size=-1)
    with pytest.raises(ValueError):
        WalkForwardCV(train_size=10, test_size=5, purge_size=-2)
