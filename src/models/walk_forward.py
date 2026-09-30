"""
Walk-forward (rolling-origin) cross-validation harness for time series.

Prevents lookahead bias and train/test leakage.
Features:
- Rolling window (fixed train size) or Expanding window (anchored start)
- Step size (retrain frequency)
- Purge window / embargo (to eliminate label overlap leakage when labels use forward horizons)
- Strict temporal ordering verification
"""

from typing import Generator
import numpy as np
import pandas as pd


class WalkForwardCV:
    """
    Time series walk-forward cross validation.

    Parameters
    ----------
    train_size : int
        Number of observations in the training window (fixed if expanding=False).
    test_size : int
        Number of observations in the test window.
    step_size : int, optional
        Number of observations to advance the origin at each split (default = test_size).
    expanding : bool, optional
        If True, train window starts at index 0 and expands over time.
        If False, train window is a fixed rolling window of size train_size.
    purge_size : int, optional
        Number of observations between end of train set and start of test set,
        to prevent label leakage from forward-looking targets (e.g. 7-day forward return).
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        expanding: bool = False,
        purge_size: int = 0,
    ):
        if train_size <= 0:
            raise ValueError(f"train_size must be positive, got {train_size}")
        if test_size <= 0:
            raise ValueError(f"test_size must be positive, got {test_size}")
        if purge_size < 0:
            raise ValueError(f"purge_size cannot be negative, got {purge_size}")

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size if step_size is not None and step_size > 0 else test_size
        self.expanding = expanding
        self.purge_size = purge_size

    def split(
        self, X: pd.DataFrame | pd.Series | np.ndarray
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) arrays.
        Guarantees strict causal ordering: all train_idx < all test_idx.
        """
        n_samples = len(X)
        current_train_end = self.train_size

        while True:
            test_start = current_train_end + self.purge_size
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            train_start = 0 if self.expanding else (current_train_end - self.train_size)
            train_idx = np.arange(train_start, current_train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
            current_train_end += self.step_size

    def get_n_splits(self, X: pd.DataFrame | pd.Series | np.ndarray) -> int:
        return sum(1 for _ in self.split(X))


def check_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
    label_horizon: int = 0,
) -> bool:
    """
    Validates that:
    1. train and test indices have zero intersection.
    2. max(train_idx) < min(test_idx)
    3. If dates given, max(train_date) < min(test_date)
    4. If label_horizon > 0, min(test_idx) - max(train_idx) > label_horizon
       to guarantee no future information leaked into training labels.
    """
    if len(np.intersect1d(train_idx, test_idx)) > 0:
        raise ValueError("Train and test indices overlap!")
    if np.max(train_idx) >= np.min(test_idx):
        raise ValueError(
            f"Temporal ordering violation: max(train)={np.max(train_idx)} >= min(test)={np.min(test_idx)}"
        )
    if label_horizon > 0 and (np.min(test_idx) - np.max(train_idx) <= label_horizon):
        raise ValueError(
            f"Purge window insufficient for label horizon {label_horizon}: "
            f"gap is {np.min(test_idx) - np.max(train_idx)} <= {label_horizon}"
        )
    if dates is not None:
        train_max_date = dates[np.max(train_idx)]
        test_min_date = dates[np.min(test_idx)]
        if train_max_date >= test_min_date:
            raise ValueError(f"Date ordering violation: train max {train_max_date} >= test min {test_min_date}")
    return True
