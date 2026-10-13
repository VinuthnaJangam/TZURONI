from __future__ import annotations

from contextlib import closing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .db import connect
from .features import load_news
from .model import performance_metrics


def write_macro_outlook(db_path: str | Path, output_path: str | Path) -> None:
    with closing(connect(db_path)) as connection:
        news = load_news(connection)
        events = pd.read_sql_query(
            "SELECT event_time_utc, event_name, actual, forecast FROM events",
            connection,
        )
    now = pd.Timestamp.now(tz="UTC")
    recent_news = news[
        news["published_at_utc"] >= now - pd.Timedelta(days=30)
    ] if not news.empty else news
    recent_events = events.copy()
    if not recent_events.empty:
        recent_events["event_time_utc"] = pd.to_datetime(
            recent_events["event_time_utc"], utc=True
        )
        recent_events = recent_events[
            recent_events["event_time_utc"] >= now - pd.Timedelta(days=30)
        ]

    if recent_news.empty:
        sentiment = None
        sentiment_label = "Unavailable (no dated news in the last 30 days)"
    else:
        sentiment = float(recent_news["sentiment"].mean())
        sentiment_label = f"{sentiment:+.3f} lexical score"
    event_surprises = pd.to_numeric(recent_events["actual"], errors="coerce") - pd.to_numeric(
        recent_events["forecast"], errors="coerce"
    ) if not recent_events.empty else pd.Series(dtype=float)
    event_surprises = event_surprises.dropna()
    surprise_line = (
        f"{len(event_surprises)} releases had both actual and forecast values; "
        f"mean raw surprise was {event_surprises.mean():+.4g} in source units."
        if len(event_surprises) else
        "No recent numeric actual/forecast pairs were available."
    )
    top_headlines = (
        "\n".join(f"- {row.title}" for row in recent_news.nlargest(5, "published_at_utc").itertuples())
        if not recent_news.empty else "- No dated Tavily articles were available."
    )
    body = f"""# Macro outlook (data-derived; past 30 days)

**As of:** {now.strftime('%Y-%m-%d %H:%M UTC')}

## Snapshot

- Dated news articles analyzed: {len(recent_news)}
- Headline lexical sentiment: {sentiment_label}
- Calendar releases in database during period: {len(recent_events)}
- {surprise_line}

## Recent headlines

{top_headlines}

## Interpretation and limitations

This is an automated descriptive summary of the ingested dataset, not a
forecast or investment recommendation. The sentiment score is a simple
word-list indicator, not a finance-domain language model; a positive value
does not imply positive asset returns. Raw surprises are not comparable across
indicators with different units. Confirm release timestamps, revisions,
country, and event definitions against the original sources before drawing
conclusions. The available database contains only the sources and dates
successfully ingested; missing observations are not evidence of no news or no
economic events.
"""
    Path(output_path).write_text(body, encoding="utf-8")


def write_charts(
    db_path: str | Path,
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with closing(connect(db_path)) as connection:
        news = load_news(connection)

    fig, axis = plt.subplots(figsize=(10, 4))
    if news.empty:
        axis.text(0.5, 0.5, "No dated news available", ha="center", va="center")
        axis.set_axis_off()
    else:
        daily = news.set_index("published_at_utc")["sentiment"].resample("D").mean()
        axis.plot(daily.index, daily, marker=".", linewidth=1)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_ylabel("Lexical sentiment")
        axis.set_title("Macro news sentiment through time")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "macro_sentiment.png", dpi=150)
    plt.close(fig)

    matrix = predictions.pivot_table(
        index="event_name", columns="strategy", values="pnl", aggfunc="mean"
    )
    fig, axis = plt.subplots(figsize=(11, max(4, 0.35 * len(matrix))))
    if matrix.empty:
        axis.text(0.5, 0.5, "No out-of-sample predictions available",
                  ha="center", va="center")
        axis.set_axis_off()
    else:
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn")
        axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
        axis.set_title("Mean out-of-sample P&L by event and strategy")
        fig.colorbar(image, ax=axis, label="P&L (return units)")
    fig.tight_layout()
    fig.savefig(destination / "strategy_matrix.png", dpi=150)
    plt.close(fig)

    selected = selected.sort_values("event_time").copy()
    selected["equity"] = (1.0 + selected["pnl"]).cumprod()
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(selected["event_time"], selected["equity"], marker=".")
    axis.set_title("Selected strategy: out-of-sample equity curve")
    axis.set_ylabel("Growth of 1.0")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "equity_curve.png", dpi=150)
    plt.close(fig)


def write_evaluation_report(
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    fold_count: int,
    output_path: str | Path,
) -> None:
    overall = performance_metrics(selected["pnl"])
    rows = []
    for strategy, group in predictions.groupby("strategy"):
        metrics = performance_metrics(group["pnl"])
        rows.append(
            f"| {strategy} | {metrics['trades']} | {metrics['sharpe']:.3f} | "
            f"{metrics['sortino']:.3f} | {metrics['max_drawdown']:.2%} | "
            f"{group['pnl'].mean():.5%} |"
        )
    report = f"""# Walk-forward model evaluation

## Selected strategy per held-out event

| Metric | Value |
|---|---:|
| Out-of-sample event decisions | {overall['trades']} |
| Expanding walk-forward folds | {fold_count} |
| Sharpe ratio (252 event periods/year) | {overall['sharpe']:.3f} |
| Sortino ratio (252 event periods/year) | {overall['sortino']:.3f} |
| Maximum drawdown | {overall['max_drawdown']:.2%} |

## Strategy performance on out-of-sample folds

| Strategy | Events scored | Sharpe | Sortino | Max drawdown | Mean P&L |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Method and interpretation

The target is the forward close-to-close market return multiplied by the
strategy's surprise-following or surprise-fading side, less configured costs.
At each out-of-sample event, the fitted model selects the candidate strategy
with the highest predicted P&L. All candidates for a given event remain in the
same time split. Features use the event's actual/forecast release values,
past-only surprise history, scheduled time, and news published before the
event. News sentiment is a simple lexical measure.

The Sharpe and Sortino ratios annualize event-level observations using 252
periods per year; macro events are not evenly spaced daily returns, so these
ratios are not directly comparable to daily portfolio metrics. Maximum
drawdown is calculated from compounded selected event returns. Results are
conditional on the supplied instrument, date range, calendar actor output,
price alignment, costs, and incomplete data handling. They are historical
estimates, not evidence of future profitability. No result should be
interpreted without inspecting the source data and generated plots.
"""
    Path(output_path).write_text(report, encoding="utf-8")
