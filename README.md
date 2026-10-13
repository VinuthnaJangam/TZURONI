# CrowdWisdom Macro Trading Research Pipeline

A reproducible, data-dependent research pipeline for ingesting macroeconomic
calendar events with Apify, collecting recent macro news with Tavily, joining
the results in SQLite, and evaluating event-driven strategies with expanding
walk-forward validation.

> This repository does not include credentials, actor IDs, market data, or
> investment advice. No performance results are claimed until the pipeline is
> run on the user's own data. Historical backtests do not predict future
> returns.

## Requirements

- Python 3.10+
- An Apify account and an actor that returns economic calendar events
- A Tavily API key
- Historical OHLC data for one instrument, in a CSV with `timestamp` and
  `close` columns. `timestamp` must be parseable; timestamps without a timezone
  are treated as UTC. Optional `symbol` values are filtered with `--symbol`.

Install and configure:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Set `APIFY_TOKEN`, `APIFY_ACTOR_ID`, and `TAVILY_API_KEY` in `.env`. Set
`APIFY_ACTOR_INPUT_FILE` to a JSON file containing the input expected by the
chosen actor. Actor output field names vary; the ingestion layer accepts common
aliases for event date/time, name, actual, and forecast. Verify the actor's
output and adjust the aliases in `src/crowdwisdom_macro/ingest.py` if needed.
Never commit `.env` or API tokens.

## Run

Ingest the last 180 days of events and up-to-date news:

```powershell
python -m crowdwisdom_macro ingest --db data\macro.sqlite --actor-input my_actor_input.json
```

`--news-query` can be supplied more than once. Defaults search for recent
inflation, Federal Reserve, and employment-report news. Apify receives the
actor-specific input as-is; the 180-day cutoff is also applied locally to
returned events.

Train and evaluate strategy permutations on out-of-sample event returns:

```powershell
python -m crowdwisdom_macro train --db data\macro.sqlite --prices data\prices.csv --symbol YOUR_SYMBOL --output reports
```

The price series is used to measure each event's return from the first
observation at/after the event to the first observation at/after the configured
holding horizon (default 1,440 minutes). Strategy permutations follow or fade
the actual-vs-forecast surprise, with rolling, past-only standardized surprise
thresholds of 0, 0.5, or 1.0. A trade only opens when the surprise clears its
threshold. Each event/strategy is a candidate; the model selects the strategy
with the highest predicted P&L for each held-out event.

Outputs in `reports\`:

- `evaluation_report.md`: sample counts, walk-forward setup, out-of-sample
  Sharpe, Sortino, maximum drawdown, and per-strategy results
- `macro_outlook.md`: one-page, data-derived recent macro/news summary
- `macro_sentiment.png`: news sentiment through time
- `strategy_matrix.png`: mean out-of-sample P&L by event type and strategy
- `equity_curve.png`: selected-strategy out-of-sample cumulative P&L

Optional flags: `--horizon-minutes`, `--cost-bps`, `--folds`, `--lookback-days`,
and `--source-timezone`. Naive calendar timestamps default to
`America/New_York` and are converted to UTC; naive price timestamps default to
UTC. Check source timezone conventions before relying on any alignment.

## Validation and methodology notes

Run tests:

```powershell
python -m unittest discover -s tests -v
```

The expanding-window folds split on event dates, keeping all strategy
permutations for an event together. Features include event name, surprise,
past-only surprise z-score, scheduled time, pre-event volatility, and
pre-event news sentiment. A random forest estimates strategy P&L. The selected
strategy is evaluated only on each fold's held-out dates. Transaction costs
are deducted in basis points from each traded round trip.

This is a research baseline, not a production execution system. Real-world
slippage, revisions, market microstructure, overlapping positions, and
calendar release-time conventions can materially affect results. A valid
assessment should disclose the instrument, data source, date range, costs,
actor ID/configuration, and any data-quality exclusions alongside the
generated metrics. Do not email API tokens; share them through a secure
credential-sharing method instead.
