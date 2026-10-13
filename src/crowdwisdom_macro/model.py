from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "surprise", "surprise_z", "news_sentiment", "pre_event_volatility",
    "hour_sin", "hour_cos", "weekday", "month", "active",
]
CATEGORICAL_FEATURES = ["event_name", "strategy"]


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    fold_count: int


def make_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def walk_forward_validate(
    frame: pd.DataFrame, folds: int = 5, min_train_events: int = 10
) -> WalkForwardResult:
    event_dates = np.array(sorted(frame["event_time"].unique()))
    if len(event_dates) <= min_train_events:
        raise ValueError(
            f"Need more than {min_train_events} distinct event dates; found {len(event_dates)}."
        )
    test_dates_count = min(max(1, len(event_dates) // (folds + 1)), 30)
    first_test = max(min_train_events, len(event_dates) - folds * test_dates_count)
    predictions: list[pd.DataFrame] = []
    fold_count = 0
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for start in range(first_test, len(event_dates), test_dates_count):
        test_dates = event_dates[start:start + test_dates_count]
        train = frame[frame["event_time"] < test_dates[0]]
        test = frame[frame["event_time"].isin(test_dates)]
        if train["event_time"].nunique() < min_train_events or test.empty:
            continue
        model = make_model()
        model.fit(train[feature_names], train["pnl"])
        fold = test[["event_time", "event_name", "strategy", "pnl"]].copy()
        fold["predicted_pnl"] = model.predict(test[feature_names])
        predictions.append(fold)
        fold_count += 1
    if not predictions:
        raise ValueError("Walk-forward split produced no valid out-of-sample folds.")
    return WalkForwardResult(pd.concat(predictions, ignore_index=True), fold_count)


def select_best_strategy(predictions: pd.DataFrame) -> pd.DataFrame:
    ordered = predictions.sort_values(
        ["event_time", "predicted_pnl", "strategy"],
        ascending=[True, False, True],
    )
    return ordered.groupby(["event_time", "event_name"], as_index=False).first()


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"trades": 0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    downside = values[values < 0]
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(np.insert(equity, 0, 1.0))[1:]
    drawdown = equity / peaks - 1.0
    return {
        "trades": int(len(values)),
        "sharpe": mean / std * np.sqrt(252) if std > 0 else 0.0,
        "sortino": mean / downside_std * np.sqrt(252) if downside_std > 0 else 0.0,
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
    }
