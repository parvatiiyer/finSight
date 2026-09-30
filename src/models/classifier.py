"""
XGBoost return-bucket classifier trained and evaluated strictly via walk-forward CV.

Labels: 'down' (0), 'flat' (1), 'up' (2) as defined by build_return_bucket_labels().
Enforces:
- Strict walk-forward evaluation (no random k-fold)
- Purge window matching or exceeding forward label horizon to prevent leakage
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, log_loss

from src.models.walk_forward import WalkForwardCV, check_no_leakage

LABEL_MAP = {"down": 0, "flat": 1, "up": 2}
REV_LABEL_MAP = {0: "down", 1: "flat", 2: "up"}


@dataclass
class WalkForwardEvaluationResult:
    oof_predictions: pd.DataFrame
    fold_metrics: list[dict]
    overall_accuracy: float
    overall_loss: float
    classification_report: dict
    feature_importances: pd.Series
    final_model: xgb.XGBClassifier


def prepare_features(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Filters trainable rows and separates feature columns from targets/metadata."""
    mask = feature_table["is_trainable"] if "is_trainable" in feature_table else (
        feature_table.notna().all(axis=1)
    )
    clean = feature_table[mask].copy()

    drop_cols = {"label", "is_trainable", "ticker", "fetched_at", "sector", "longName", "shortName"}
    feature_cols = [c for c in clean.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(clean[c])]

    X = clean[feature_cols].copy()
    y = clean["label"].map(LABEL_MAP).astype(int)
    return X, y, feature_cols


def train_walk_forward_classifier(
    feature_table: pd.DataFrame,
    train_size: int = 120,
    test_size: int = 20,
    step_size: int | None = None,
    purge_size: int = 7,
    n_estimators: int = 50,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> WalkForwardEvaluationResult:
    """
    Evaluates an XGBoost return-bucket classifier across time using WalkForwardCV.
    """
    X, y, feature_cols = prepare_features(feature_table)
    if len(X) < train_size + test_size + purge_size:
        raise ValueError(
            f"Not enough trainable observations ({len(X)}) for train_size={train_size}, "
            f"test_size={test_size}, purge_size={purge_size}"
        )

    cv = WalkForwardCV(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        expanding=False,
        purge_size=purge_size,
    )

    oof_records = []
    fold_metrics = []
    importances_list = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        check_no_leakage(train_idx, test_idx, label_horizon=purge_size)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=3,
            random_state=random_state,
            eval_metric="mlogloss",
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs, labels=[0, 1, 2])
        fold_metrics.append({"fold": fold_idx, "accuracy": acc, "loss": loss, "n_test": len(test_idx)})

        for i, idx_val in enumerate(X_test.index):
            oof_records.append({
                "date": idx_val,
                "actual": REV_LABEL_MAP[y_test.iloc[i]],
                "pred": REV_LABEL_MAP[preds[i]],
                "prob_down": probs[i, 0],
                "prob_flat": probs[i, 1],
                "prob_up": probs[i, 2],
            })

        importances_list.append(clf.feature_importances_)

    oof_df = pd.DataFrame(oof_records).set_index("date")
    actual_numeric = oof_df["actual"].map(LABEL_MAP)
    pred_numeric = oof_df["pred"].map(LABEL_MAP)
    prob_matrix = oof_df[["prob_down", "prob_flat", "prob_up"]].values

    overall_acc = float(accuracy_score(actual_numeric, pred_numeric))
    overall_loss = float(log_loss(actual_numeric, prob_matrix, labels=[0, 1, 2]))
    report = classification_report(
        actual_numeric,
        pred_numeric,
        target_names=["down", "flat", "up"],
        output_dict=True,
        zero_division=0,
    )

    avg_importances = pd.Series(
        np.mean(importances_list, axis=0), index=feature_cols
    ).sort_values(ascending=False)

    # Train final model on the trailing train_size observations
    final_clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=3,
        random_state=random_state,
        eval_metric="mlogloss",
    )
    final_clf.fit(X.tail(train_size), y.tail(train_size))

    return WalkForwardEvaluationResult(
        oof_predictions=oof_df,
        fold_metrics=fold_metrics,
        overall_accuracy=overall_acc,
        overall_loss=overall_loss,
        classification_report=report,
        feature_importances=avg_importances,
        final_model=final_clf,
    )


def predict_bucket(model: xgb.XGBClassifier, feature_row: pd.Series, feature_cols: list[str]) -> dict:
    """Predicts direction bucket for a single feature observation."""
    X_input = pd.DataFrame([feature_row[feature_cols]])
    probs = model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_bucket": REV_LABEL_MAP[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "down": float(probs[0]),
            "flat": float(probs[1]),
            "up": float(probs[2]),
        },
    }
