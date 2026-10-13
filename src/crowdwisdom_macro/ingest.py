from __future__ import annotations

import json
import os
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .db import connect, save_events, save_news

APIFY_API = "https://api.apify.com/v2"
TAVILY_API = "https://api.tavily.com/search"


def _first(record: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "").replace("$", "")
    if not text or text.lower() in {"n/a", "na", "-", "none"}:
        return None
    multiplier = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}.get(
        text[-1:].lower(), 1.0
    )
    if multiplier != 1.0:
        text = text[:-1].strip()
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def normalize_event(record: dict[str, Any], source_timezone: str) -> dict[str, Any]:
    time_value = _first(
        record,
        (
            "datetime", "datetimeISO", "date_time", "timestamp", "release_datetime",
            "scheduled_at", "datetime_utc",
        ),
    )
    date_value = _first(record, ("event_date", "date", "release_date"))
    time_of_day = _first(
        record, ("time", "event_time_of_day", "event_time_local", "scheduled_time")
    )
    if time_value is None:
        event_time = _first(record, ("event_time", "release_time"))
        if event_time is not None:
            try:
                datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
                time_value = event_time
            except ValueError:
                if date_value is None:
                    raise ValueError(
                        f"Calendar record has an unrecognized event time: {event_time!r}"
                    )
                time_value = f"{date_value} {event_time}"
        elif date_value is not None:
            time_value = f"{date_value} {time_of_day}" if time_of_day else date_value
    if time_value is None:
        raise ValueError(f"Calendar record has no recognized date/time field: {record!r}")

    parsed = datetime.fromisoformat(str(time_value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(source_timezone))
    event_time_utc = parsed.astimezone(timezone.utc).isoformat()
    name = _first(
        record, ("event_name", "event", "name", "prefixedName", "title", "indicator")
    )
    if not name:
        raise ValueError(f"Calendar record has no recognized event-name field: {record!r}")

    return {
        "event_time_utc": event_time_utc,
        "event_name": str(name).strip(),
        "actual": _number(_first(record, ("actual", "actual_value", "value"))),
        "forecast": _number(
            _first(record, ("forecast", "forecast_value", "consensus", "expected"))
        ),
        "source_timezone": source_timezone,
        "raw": record,
    }


def fetch_apify_events(
    token: str,
    actor_id: str,
    actor_input: dict[str, Any],
    source_timezone: str,
    lookback_days: int = 180,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    client = session or requests.Session()
    actor_path = actor_id.replace("/", "~")
    response = client.post(
        f"{APIFY_API}/acts/{actor_path}/run-sync-get-dataset-items",
        params={"token": token, "format": "json"},
        json=actor_input,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Apify response was not a dataset-item list")
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    normalized = [normalize_event(item, source_timezone) for item in payload]
    return [
        item for item in normalized
        if datetime.fromisoformat(item["event_time_utc"]) >= cutoff
    ]


def fetch_tavily_news(
    api_key: str,
    queries: list[str],
    days: int = 7,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    client = session or requests.Session()
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for query in queries:
        response = client.post(
            TAVILY_API,
            json={
                "api_key": api_key,
                "query": query,
                "topic": "news",
                "days": days,
                "search_depth": "advanced",
                "max_results": 10,
                "include_answer": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            published = item.get("published_date") or item.get("published_at")
            published_utc = None
            if published:
                published_text = str(published)
                try:
                    parsed = datetime.fromisoformat(
                        published_text.replace("Z", "+00:00")
                    )
                except ValueError:
                    parsed = parsedate_to_datetime(published_text)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                published_utc = parsed.astimezone(timezone.utc).isoformat()
            article = {
                "published_at_utc": published_utc,
                "title": str(item.get("title") or "Untitled"),
                "content": str(item.get("content") or ""),
                "url": str(item.get("url") or ""),
                "query": query,
                "raw": item,
            }
            results[(article["url"], article["title"])] = article
    return list(results.values())


def ingest(
    db_path: str | Path,
    actor_input_path: str | Path,
    source_timezone: str = "America/New_York",
    lookback_days: int = 180,
    news_queries: list[str] | None = None,
) -> tuple[int, int]:
    apify_token = os.getenv("APIFY_TOKEN")
    actor_id = os.getenv("APIFY_ACTOR_ID")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not apify_token or not actor_id:
        raise RuntimeError("Set APIFY_TOKEN and APIFY_ACTOR_ID before ingesting events.")
    if not tavily_key:
        raise RuntimeError("Set TAVILY_API_KEY before ingesting news.")
    actor_input = json.loads(Path(actor_input_path).read_text(encoding="utf-8"))
    if not isinstance(actor_input, dict):
        raise ValueError("Apify actor input JSON must be an object.")

    events = fetch_apify_events(
        apify_token, actor_id, actor_input, source_timezone, lookback_days
    )
    queries = news_queries or [
        "latest US inflation CPI macroeconomic news",
        "latest Federal Reserve FOMC interest rate news",
        "latest US employment labor market economic news",
    ]
    articles = fetch_tavily_news(tavily_key, queries)
    with closing(connect(db_path)) as connection:
        event_count = save_events(connection, events)
        news_count = save_news(connection, articles)
    return event_count, news_count
