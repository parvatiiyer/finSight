"""
Tests for GARCH(1,1) volatility modeling and Isolation Forest anomaly detection.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_ohlcv
from src.features.technical import add_technical_features
from src.models.volatility import compute_garch_volatility
from src.models.anomaly import detect_market_anomalies
from src.agents.technical_agent import gather_technical_evidence


def test_garch_volatility_computation():
    ohlcv = generate_ohlcv("TEST.NS", n_days=150, seed=42)
    close = ohlcv["Close"]
    vol = compute_garch_volatility(close, p=1, q=1, annualized=True)

    assert len(vol) == len(close)
    # Volatility should be positive
    valid_vol = vol.dropna()
    assert (valid_vol > 0).all()
    # Check that it's attached in add_technical_features
    feats = add_technical_features(ohlcv)
    assert "garch_vol_20" in feats.columns
    assert not feats["garch_vol_20"].isna().all()


def test_isolation_forest_anomaly_detection():
    ohlcv = generate_ohlcv("TEST.NS", n_days=150, seed=42)
    anom_df = detect_market_anomalies(ohlcv, contamination=0.05)

    assert "is_anomaly" in anom_df.columns
    assert "anomaly_score" in anom_df.columns
    assert "anomaly_reason" in anom_df.columns

    n_anomalies = anom_df["is_anomaly"].sum()
    assert n_anomalies > 0, "Isolation Forest should detect outliers with 5% contamination"

    # Verify that anomalous rows have non-empty reason strings
    flagged_reasons = anom_df[anom_df["is_anomaly"]]["anomaly_reason"]
    assert (flagged_reasons != "").all()


def test_technical_agent_anomaly_integration():
    ohlcv = generate_ohlcv("TEST.NS", n_days=150, seed=42)
    feats = add_technical_features(ohlcv)

    # Pick a window spanning multiple days
    w_start = feats.index[-15]
    w_end = feats.index[-1]

    evidence = gather_technical_evidence(feats, w_start, w_end)
    assert isinstance(evidence, list)
    # Check flexible invocation (passing ticker as first argument)
    ev_with_ticker = gather_technical_evidence("TEST.NS", feats, w_start, w_end)
    assert len(evidence) == len(ev_with_ticker)
