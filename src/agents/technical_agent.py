"""Technical evidence agent — descriptive market-microstructure context for a move."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agents.evidence_types import Evidence, Direction

def gather_technical_evidence(
    feature_table_or_ticker,
    window_start_or_feature_table=None,
    window_end_or_start=None,
    window_end=None,
    volume_spike_threshold: float = 1.8,
    rsi_extreme_low: float = 30,
    rsi_extreme_high: float = 70,
):
    # Support both (feature_table, window_start, window_end) and (ticker, feature_table, window_start, window_end)
    if isinstance(feature_table_or_ticker, str) and isinstance(window_start_or_feature_table, pd.DataFrame):
        feature_table = window_start_or_feature_table
        w_start = window_end_or_start
        w_end = window_end
    else:
        feature_table = feature_table_or_ticker
        w_start = window_start_or_feature_table
        w_end = window_end_or_start

    w_start = pd.Timestamp(w_start).tz_localize(None) if w_start is not None else None
    w_end = pd.Timestamp(w_end).tz_localize(None) if w_end is not None else None

    idx = feature_table.index.tz_localize(None) if getattr(feature_table.index, "tz", None) else feature_table.index
    mask = pd.Series(True, index=feature_table.index)
    if w_start is not None:
        mask &= (idx >= w_start)
    if w_end is not None:
        mask &= (idx <= w_end)

    window = feature_table.loc[mask]
    if window.empty:
        return []
    evidence = []
    max_vol_ratio = window["volume_ratio"].max()
    if max_vol_ratio >= volume_spike_threshold:
        spike_date = window["volume_ratio"].idxmax()
        evidence.append(Evidence("technical", spike_date.date(), Direction.NEUTRAL,
                                  min(1.0, (max_vol_ratio - 1) / 2),
                                  f"Volume spiked to {max_vol_ratio:.1f}x the 20-day average on {spike_date.date()}",
                                  1.0, ["volume_spike"],
                                  source_path=f"market_data/ohlcv/{spike_date.date()}#volume_spike"))
    rsi_at_start = window["rsi_14"].iloc[0]
    if pd.notna(rsi_at_start):
        if rsi_at_start <= rsi_extreme_low:
            evidence.append(Evidence("technical", window.index[0].date(), Direction.NEUTRAL,
                                      (rsi_extreme_low - rsi_at_start) / rsi_extreme_low,
                                      f"RSI(14) was already oversold at {rsi_at_start:.0f} entering the window",
                                      0.8, ["rsi_extreme", "oversold"],
                                      source_path=f"indicators/rsi_14/{window.index[0].date()}#oversold"))
        elif rsi_at_start >= rsi_extreme_high:
            evidence.append(Evidence("technical", window.index[0].date(), Direction.NEUTRAL,
                                      (rsi_at_start - rsi_extreme_high) / (100 - rsi_extreme_high),
                                      f"RSI(14) was already overbought at {rsi_at_start:.0f} entering the window",
                                      0.8, ["rsi_extreme", "overbought"],
                                      source_path=f"indicators/rsi_14/{window.index[0].date()}#overbought"))
    max_pct_b = window["bb_pct_b"].max()
    min_pct_b = window["bb_pct_b"].min()
    if pd.notna(max_pct_b) and max_pct_b >= 1.0:
        breakout_date = window["bb_pct_b"].idxmax().date()
        evidence.append(Evidence("technical", breakout_date, Direction.NEUTRAL,
                                  min(1.0, max_pct_b - 1.0 + 0.3),
                                  "Price broke above its upper Bollinger Band during the window",
                                  0.7, ["breakout", "upper_band"],
                                  source_path=f"indicators/bollinger_bands/{breakout_date}#upper_band"))
    if pd.notna(min_pct_b) and min_pct_b <= 0.0:
        breakout_date = window["bb_pct_b"].idxmin().date()
        evidence.append(Evidence("technical", breakout_date, Direction.NEUTRAL,
                                  min(1.0, abs(min_pct_b) + 0.3),
                                  "Price broke below its lower Bollinger Band during the window",
                                  0.7, ["breakout", "lower_band"],
                                  source_path=f"indicators/bollinger_bands/{breakout_date}#lower_band"))

    # Anomaly detection via Isolation Forest
    try:
        from src.models.anomaly import detect_market_anomalies
        anomalies = detect_market_anomalies(feature_table)
        anom_window = anomalies.loc[(anomalies.index >= w_start) & (anomalies.index <= w_end)]
        flagged = anom_window[anom_window["is_anomaly"]]
        if not flagged.empty:
            worst_dt = flagged["anomaly_score"].idxmin()
            worst_row = flagged.loc[worst_dt]
            reason = worst_row["anomaly_reason"] or "statistical multivariate outlier"
            evidence.append(Evidence(
                source_agent="technical",
                event_date=worst_dt.date(),
                direction=Direction.NEUTRAL,
                magnitude_hint=0.65,
                raw_text=f"Isolation Forest flagged {worst_dt.date()} as market microstructure anomaly ({reason})",
                reliability_weight=0.85,
                tags=["anomaly_detection", "isolation_forest"],
                source_path=f"models/isolation_forest/{worst_dt.date()}#multivariate_anomaly",
            ))
    except Exception as e:
        # Gracefully proceed if anomaly model fails
        pass

    return evidence
