"""
GARCH(1,1) volatility model via the arch package.
Complements realized rolling volatility by capturing volatility clustering.
"""

import numpy as np
import pandas as pd


def compute_garch_volatility(
    close: pd.Series,
    p: int = 1,
    q: int = 1,
    annualized: bool = True,
    fallback_window: int = 20,
) -> pd.Series:
    """
    Fits a GARCH(p, q) model to log returns of close prices.
    Returns conditional volatility aligned with the close Series index.

    Falls back cleanly to rolling std if length < 40 or optimization does not converge.
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    scale_factor = np.sqrt(252) if annualized else 1.0

    # Baseline fallback
    rolling_vol = (
        np.log(close / close.shift(1)).rolling(fallback_window).std() * scale_factor
    )

    if len(log_returns) < 40:
        return rolling_vol

    try:
        from arch import arch_model

        # Scale by 100 for numerical stability in optimizer
        scaled_returns = log_returns * 100.0
        model = arch_model(scaled_returns, vol="Garch", p=p, q=q, rescale=False)
        res = model.fit(disp="off", show_warning=False)

        cond_vol = res.conditional_volatility / 100.0 * scale_factor
        cond_series = pd.Series(cond_vol, index=log_returns.index)

        # Reindex to full close index, forward fill initial gap
        aligned = cond_series.reindex(close.index)
        aligned = aligned.fillna(rolling_vol)
        return aligned
    except Exception:
        return rolling_vol
