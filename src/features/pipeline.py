"""End-to-end feature table builder combining technical + fundamental features + labels."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.technical import add_technical_features, build_return_bucket_labels
from src.features.fundamental import fundamentals_to_frame

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def build_feature_table(ohlcv: pd.DataFrame, fundamentals: dict, label_horizon: int = 7) -> pd.DataFrame:
    feats = add_technical_features(ohlcv)
    fund_frame = fundamentals_to_frame(fundamentals, feats.index)
    table = feats.join(fund_frame)
    table["label"] = build_return_bucket_labels(ohlcv, horizon=label_horizon)
    feature_cols = [c for c in table.columns if c != "label"]
    table["is_trainable"] = table[feature_cols].notna().all(axis=1) & table["label"].notna()
    return table
