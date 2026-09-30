"""
Tests for XGBoost classifier trained via walk-forward CV.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_ohlcv, generate_fundamentals
from src.features.pipeline import build_feature_table
from src.models.classifier import train_walk_forward_classifier, predict_bucket, prepare_features


def test_xgboost_walk_forward_pipeline():
    ohlcv = generate_ohlcv("TEST.NS", n_days=350, seed=42)
    fundamentals = generate_fundamentals("TEST.NS")
    feature_table = build_feature_table(ohlcv, fundamentals)

    res = train_walk_forward_classifier(
        feature_table,
        train_size=100,
        test_size=20,
        step_size=20,
        purge_size=7,
        n_estimators=10,  # fast for testing
        max_depth=2,
    )

    assert len(res.fold_metrics) > 0
    assert 0.0 <= res.overall_accuracy <= 1.0
    assert len(res.oof_predictions) > 0
    assert "accuracy" in res.classification_report
    assert not res.feature_importances.empty

    # Test single-row inference
    _, _, feature_cols = prepare_features(feature_table)
    latest_row = feature_table.iloc[-1]
    pred = predict_bucket(res.final_model, latest_row, feature_cols)
    assert pred["predicted_bucket"] in ("down", "flat", "up")
    assert 0.0 <= pred["confidence"] <= 1.0
    assert sum(pred["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
