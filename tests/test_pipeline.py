import sqlite3
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from crowdwisdom_macro.db import save_events, save_news
from crowdwisdom_macro.features import build_training_frame, lexical_sentiment
from crowdwisdom_macro.ingest import normalize_event
from crowdwisdom_macro.model import (
    performance_metrics,
    select_best_strategy,
    walk_forward_validate,
)


class IngestionTests(unittest.TestCase):
    def test_naive_event_time_converts_from_source_timezone_to_utc(self):
        event = normalize_event(
            {
                "date": "2026-01-15 08:30:00",
                "event": "CPI",
                "actual": "3.2%",
                "forecast": "3.1%",
            },
            "America/New_York",
        )
        self.assertEqual(event["event_time_utc"], "2026-01-15T13:30:00+00:00")
        self.assertAlmostEqual(event["actual"], 3.2)
        self.assertAlmostEqual(event["forecast"], 3.1)

    def test_split_date_and_time_and_suffix_values_are_normalized(self):
        event = normalize_event(
            {
                "date": "2026-01-15",
                "time": "08:30",
                "event": "Nonfarm payrolls",
                "actual": "64K",
                "forecast": "73K",
            },
            "America/New_York",
        )
        self.assertEqual(event["event_time_utc"], "2026-01-15T13:30:00+00:00")
        self.assertEqual(event["actual"], 64_000)
        self.assertEqual(event["forecast"], 73_000)

    def test_unknown_event_fields_raise_instead_of_silently_dropping(self):
        with self.assertRaises(ValueError):
            normalize_event({"label": "CPI"}, "UTC")

    def test_database_upserts_event_and_news(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY, event_time_utc TEXT, event_name TEXT,
                actual REAL, forecast REAL, source_timezone TEXT, raw_json TEXT
            );
            CREATE TABLE news (
                news_id TEXT PRIMARY KEY, published_at_utc TEXT, title TEXT,
                content TEXT, url TEXT, query TEXT, raw_json TEXT
            );
            """
        )
        event = {
            "event_time_utc": "2026-01-15T13:30:00+00:00",
            "event_name": "CPI", "actual": 3.2, "forecast": 3.1,
            "source_timezone": "America/New_York", "raw": {},
        }
        article = {
            "published_at_utc": "2026-01-15T12:00:00+00:00", "title": "Rates ease",
            "content": "", "url": "https://example.com", "query": "rates", "raw": {},
        }
        self.assertEqual(save_events(connection, [event, event]), 2)
        self.assertEqual(save_news(connection, [article, article]), 2)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM news").fetchone()[0], 1)


class FeatureAndMetricTests(unittest.TestCase):
    def test_sentiment_is_bounded_and_neutral_without_known_words(self):
        self.assertGreater(lexical_sentiment("strong growth"), 0)
        self.assertLess(lexical_sentiment("weak recession"), 0)
        self.assertEqual(lexical_sentiment("unclassified language"), 0)

    def test_features_use_surprise_and_return_and_group_strategy_candidates(self):
        timestamps = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
        events = pd.DataFrame(
            {
                "event_time_utc": timestamps[::3],
                "event_name": ["CPI"] * 30,
                "actual": np.linspace(2.0, 3.0, 30),
                "forecast": [2.0] * 30,
            }
        )
        prices = pd.DataFrame(
            {"timestamp": timestamps, "close": 100 * (1.001 ** np.arange(90))}
        )
        news = pd.DataFrame(
            columns=["published_at_utc", "title", "content", "sentiment"]
        )
        result = build_training_frame(events, news, prices, horizon_minutes=1440)
        self.assertEqual(result["event_time"].nunique(), 30)
        self.assertEqual(result["strategy"].nunique(), 6)
        self.assertTrue((result.loc[result["active"] == 0, "side"] == 0).all())
        self.assertTrue(np.isfinite(result["pre_event_volatility"]).all())

    def test_strategy_selector_picks_highest_prediction_for_each_event(self):
        predictions = pd.DataFrame(
            {
                "event_time": [datetime(2025, 1, 1, tzinfo=timezone.utc)] * 4,
                "event_name": ["CPI", "CPI", "GDP", "GDP"],
                "strategy": ["follow", "fade", "follow", "fade"],
                "pnl": [0.02, -0.01, 0.03, -0.02],
                "predicted_pnl": [-0.01, 0.03, 0.04, 0.01],
            }
        )
        selected = select_best_strategy(predictions)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.set_index("event_name").loc["CPI", "strategy"], "fade")
        self.assertEqual(selected.set_index("event_name").loc["CPI", "pnl"], -0.01)
        self.assertEqual(selected.set_index("event_name").loc["GDP", "strategy"], "follow")

    def test_performance_metrics_cover_zero_and_negative_returns(self):
        metrics = performance_metrics(pd.Series([0.0, -0.1, 0.05]))
        self.assertEqual(metrics["trades"], 3)
        self.assertLess(metrics["max_drawdown"], 0)
        self.assertTrue(np.isfinite(metrics["sortino"]))

    def test_walk_forward_keeps_each_event_in_a_single_test_fold(self):
        rows = []
        dates = pd.date_range("2025-01-01", periods=22, freq="D", tz="UTC")
        for index, timestamp in enumerate(dates):
            for strategy in ("follow_z0", "fade_z0"):
                rows.append(
                    {
                        "event_time": timestamp,
                        "event_name": "CPI",
                        "strategy": strategy,
                        "surprise": float(index % 4),
                        "surprise_z": float(index % 3),
                        "news_sentiment": 0.1,
                        "pre_event_volatility": 0.01,
                        "hour_sin": 0.0,
                        "hour_cos": 1.0,
                        "weekday": timestamp.dayofweek,
                        "month": timestamp.month,
                        "active": 1,
                        "pnl": float((index % 5) * 0.001),
                    }
                )
        result = walk_forward_validate(pd.DataFrame(rows), folds=3, min_train_events=10)
        self.assertEqual(result.fold_count, 3)
        self.assertEqual(result.predictions["event_time"].nunique(), 12)
        self.assertEqual(len(result.predictions), 24)


if __name__ == "__main__":
    unittest.main()
