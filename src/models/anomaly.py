"""
Anomaly detection on price, volume, and volatility jointly via Isolation Forest.
Provides a principled, multivariate flag for unusual trading days instead of hand-tuned single thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_market_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fits an IsolationForest on joint volume, return magnitude, and intraday range.
    Returns a DataFrame with columns:
        - anomaly_score: float (lower means more anomalous)
        - is_anomaly: bool (True if flagged)
        - anomaly_reason: string description of the outlier characteristics
    """
    out = pd.DataFrame(index=df.index)
    out["is_anomaly"] = False
    out["anomaly_score"] = 0.0
    out["anomaly_reason"] = ""

    if len(df) < 30:
        return out

    close = df["Close"]
    volume = df["Volume"]

    # Feature 1: absolute 1-day return
    ret_1d = np.abs(np.log(close / close.shift(1)))
    # Feature 2: volume ratio over 20-day rolling mean
    vol_sma = volume.rolling(20).mean().replace(0, np.nan)
    vol_ratio = (volume / vol_sma).fillna(1.0)
    # Feature 3: intraday range (High - Low) / Close
    high = df["High"] if "High" in df else close
    low = df["Low"] if "Low" in df else close
    intraday_range = ((high - low) / close).fillna(0.0)

    features = pd.DataFrame({
        "abs_ret": ret_1d,
        "vol_ratio": vol_ratio,
        "range": intraday_range,
    }).dropna()

    if len(features) < 30:
        return out

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    preds = iso.fit_predict(features)  # -1 for anomalies, 1 for normal
    scores = iso.decision_function(features)

    is_anom = preds == -1
    out.loc[features.index, "is_anomaly"] = is_anom
    out.loc[features.index, "anomaly_score"] = scores

    # Generate human-readable reasons for anomalies
    for dt in features.index[is_anom]:
        row = features.loc[dt]
        reasons = []
        if row["vol_ratio"] > 1.8:
            reasons.append(f"volume {row['vol_ratio']:.1f}x normal")
        if row["abs_ret"] > 0.025:
            reasons.append(f"large price shock {row['abs_ret']:.1%}")
        if row["range"] > 0.035:
            reasons.append(f"wide intraday swing {row['range']:.1%}")
        if not reasons:
            reasons.append("unusual joint volume-volatility pattern")
        out.loc[dt, "anomaly_reason"] = ", ".join(reasons)

    return out
