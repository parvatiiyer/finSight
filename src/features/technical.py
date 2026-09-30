"""Technical indicator features, computed purely from OHLCV. All causal (backward-looking only)."""
import numpy as np
import pandas as pd
import ta

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]
    out["sma_10"] = close.rolling(10).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_10"] = close.ewm(span=10, adjust=False).mean()
    out["sma_10_50_ratio"] = out["sma_10"] / out["sma_50"]
    out["macd"] = ta.trend.MACD(close).macd()
    out["macd_signal"] = ta.trend.MACD(close).macd_signal()
    out["macd_diff"] = ta.trend.MACD(close).macd_diff()
    out["adx"] = ta.trend.ADXIndicator(high, low, close).adx()
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    out["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    bb = ta.volatility.BollingerBands(close)
    out["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    out["bb_pct_b"] = bb.bollinger_pband()
    out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    out["realized_vol_20"] = np.log(close / close.shift(1)).rolling(20).std() * np.sqrt(252)
    try:
        from src.models.volatility import compute_garch_volatility
        out["garch_vol_20"] = compute_garch_volatility(close)
    except Exception:
        out["garch_vol_20"] = out["realized_vol_20"]
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    out["volume_sma_20"] = volume.rolling(20).mean()
    out["volume_ratio"] = volume / out["volume_sma_20"]
    for h in [1, 5, 10, 20]:
        out[f"return_{h}d"] = close.pct_change(h)
    return out

def build_return_bucket_labels(df, horizon=7, up_threshold=0.02, down_threshold=-0.02):
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    labels = pd.Series(index=df.index, dtype="object")
    labels[future_return > up_threshold] = "up"
    labels[future_return < down_threshold] = "down"
    labels[(future_return >= down_threshold) & (future_return <= up_threshold)] = "flat"
    return labels
