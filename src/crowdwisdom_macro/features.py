from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

POSITIVE_WORDS = {
    "beat", "booming", "growth", "improved", "increase", "strong", "surge",
    "upside", "resilient", "eased", "cooling", "cut",
}
NEGATIVE_WORDS = {
    "miss", "recession", "decline", "declined", "weak", "slowdown", "crisis",
    "risk", "downside", "inflation", "hawkish", "layoffs",
}


def lexical_sentiment(text: str) -> float:
    words = [word.strip(".,:;!?()[]{}\"'").lower() for word in text.split()]
    scored = [1.0 if word in POSITIVE_WORDS else -1.0
              for word in words if word in POSITIVE_WORDS | NEGATIVE_WORDS]
    return float(np.mean(scored)) if scored else 0.0


def load_events(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT event_time_utc, event_name, actual, forecast FROM events ORDER BY event_time_utc",
        connection,
    )
    if not frame.empty:
        frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True)
    return frame


def load_news(connection: sqlite3.Connection) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT published_at_utc, title, content FROM news WHERE published_at_utc IS NOT NULL",
        connection,
    )
    if not frame.empty:
        frame["published_at_utc"] = pd.to_datetime(frame["published_at_utc"], utc=True)
        frame["sentiment"] = (frame["title"] + " " + frame["content"]).map(lexical_sentiment)
    return frame


def _event_news_sentiment(event_time: pd.Timestamp, news: pd.DataFrame) -> float:
    if news.empty:
        return 0.0
    recent = news[
        (news["published_at_utc"] <= event_time)
        & (news["published_at_utc"] >= event_time - pd.Timedelta(days=7))
    ]
    return float(recent["sentiment"].mean()) if not recent.empty else 0.0


def _prior_surprise_zscores(events: pd.DataFrame) -> np.ndarray:
    values = events["surprise"].to_numpy(dtype=float)
    names = events["event_name"].to_numpy()
    scores = np.zeros(len(events), dtype=float)
    history: dict[str, list[float]] = {}
    for index, (name, value) in enumerate(zip(names, values)):
        previous = history.setdefault(str(name), [])
        if len(previous) >= 2:
            mean = float(np.mean(previous))
            std = float(np.std(previous, ddof=1))
            scores[index] = (value - mean) / std if std > 1e-12 else 0.0
        previous.append(value)
    return np.clip(scores, -10.0, 10.0)


def _price_return(
    prices: pd.DataFrame, event_time: pd.Timestamp, horizon: pd.Timedelta
) -> tuple[float, float] | None:
    before = prices[prices["timestamp"] >= event_time]
    after = prices[prices["timestamp"] >= event_time + horizon]
    if before.empty or after.empty:
        return None
    entry = float(before.iloc[0]["close"])
    exit_price = float(after.iloc[0]["close"])
    if entry <= 0 or exit_price <= 0:
        return None
    return exit_price / entry - 1.0, entry


def _pre_event_volatility(prices: pd.DataFrame, event_time: pd.Timestamp) -> float:
    history = prices.loc[prices["timestamp"] < event_time, "close"].tail(31)
    returns = history.pct_change().dropna()
    return float(returns.std(ddof=1)) if len(returns) > 1 else 0.0


def build_training_frame(
    events: pd.DataFrame,
    news: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_minutes: int = 1440,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    if events.empty:
        raise ValueError("No macro events are available in the database.")
    if prices.empty:
        raise ValueError("Price CSV has no usable rows.")
    required = {"timestamp", "close"}
    if not required.issubset(prices.columns):
        raise ValueError("Price CSV must contain timestamp and close columns.")

    events = events.sort_values("event_time_utc").copy().reset_index(drop=True)
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    prices = prices[prices["close"] > 0].reset_index(drop=True)
    events["actual"] = pd.to_numeric(events["actual"], errors="coerce")
    events["forecast"] = pd.to_numeric(events["forecast"], errors="coerce")
    events = events.dropna(subset=["actual", "forecast"]).reset_index(drop=True)
    if events.empty:
        raise ValueError("No events have both numeric actual and forecast values.")

    events["surprise"] = events["actual"] - events["forecast"]
    events["surprise_z"] = _prior_surprise_zscores(events)
    events["news_sentiment"] = [
        _event_news_sentiment(timestamp, news) for timestamp in events["event_time_utc"]
    ]
    horizon = pd.Timedelta(minutes=horizon_minutes)
    costs = cost_bps / 10_000.0
    samples: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        result = _price_return(prices, row.event_time_utc, horizon)
        if result is None:
            continue
        market_return, _ = result
        timestamp = row.event_time_utc
        base = {
            "event_time": timestamp,
            "event_name": str(row.event_name),
            "surprise": float(row.surprise),
            "surprise_z": float(row.surprise_z),
            "news_sentiment": float(row.news_sentiment),
            "pre_event_volatility": _pre_event_volatility(prices, timestamp),
            "hour_sin": float(np.sin(2 * np.pi * timestamp.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * timestamp.hour / 24)),
            "weekday": int(timestamp.dayofweek),
            "month": int(timestamp.month),
        }
        for threshold in (0.0, 0.5, 1.0):
            active = int(abs(row.surprise_z) >= threshold and row.surprise_z != 0)
            surprise_direction = 1 if row.surprise_z > 0 else -1
            for rule, multiplier in (("follow", 1), ("fade", -1)):
                side = surprise_direction * multiplier if active else 0
                samples.append(
                    {
                        **base,
                        "strategy": f"{rule}_z{threshold:g}",
                        "active": active,
                        "side": side,
                        "pnl": side * market_return - (costs if active else 0.0),
                    }
                )
    frame = pd.DataFrame(samples)
    if frame.empty:
        raise ValueError("No event has price observations at both entry and exit.")
    return frame
