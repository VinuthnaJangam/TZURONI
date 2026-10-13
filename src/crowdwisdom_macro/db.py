from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            event_time_utc TEXT NOT NULL,
            event_name TEXT NOT NULL,
            actual REAL,
            forecast REAL,
            source_timezone TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time_utc);
        CREATE TABLE IF NOT EXISTS news (
            news_id TEXT PRIMARY KEY,
            published_at_utc TEXT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            url TEXT NOT NULL,
            query TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_news_time ON news(published_at_utc);
        """
    )
    connection.commit()
    return connection


def _identifier(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def save_events(connection: sqlite3.Connection, events: list[dict[str, Any]]) -> int:
    rows = []
    for event in events:
        event_id = _identifier(
            event["event_time_utc"], event["event_name"], str(event.get("actual")),
            str(event.get("forecast")),
        )
        rows.append(
            (
                event_id, event["event_time_utc"], event["event_name"],
                event.get("actual"), event.get("forecast"),
                event["source_timezone"],
                json.dumps(event["raw"], sort_keys=True, default=str),
            )
        )
    connection.executemany(
        """INSERT OR REPLACE INTO events
        (event_id, event_time_utc, event_name, actual, forecast, source_timezone, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    connection.commit()
    return len(rows)


def save_news(connection: sqlite3.Connection, articles: list[dict[str, Any]]) -> int:
    rows = []
    for article in articles:
        news_id = _identifier(article["url"], article["title"])
        rows.append(
            (
                news_id, article.get("published_at_utc"), article["title"],
                article.get("content", ""), article["url"], article["query"],
                json.dumps(article["raw"], sort_keys=True, default=str),
            )
        )
    connection.executemany(
        """INSERT OR REPLACE INTO news
        (news_id, published_at_utc, title, content, url, query, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    connection.commit()
    return len(rows)
