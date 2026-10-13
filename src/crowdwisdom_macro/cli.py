from __future__ import annotations

import argparse
import os
from contextlib import closing
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .db import connect
from .features import build_training_frame, load_events, load_news
from .ingest import ingest
from .model import select_best_strategy, walk_forward_validate
from .report import write_charts, write_evaluation_report, write_macro_outlook


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crowdwisdom-macro",
        description="Ingest macro calendar/news data and evaluate event strategies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Fetch Apify events and Tavily news.")
    ingest_parser.add_argument("--db", default=None)
    ingest_parser.add_argument("--actor-input", default=None)
    ingest_parser.add_argument("--source-timezone", default="America/New_York")
    ingest_parser.add_argument("--lookback-days", type=int, default=180)
    ingest_parser.add_argument("--news-query", action="append", dest="news_queries")

    train_parser = commands.add_parser("train", help="Train, validate, and write reports.")
    train_parser.add_argument("--db", default=None)
    train_parser.add_argument("--prices", required=True)
    train_parser.add_argument("--symbol")
    train_parser.add_argument("--output", default="reports")
    train_parser.add_argument("--horizon-minutes", type=int, default=1440)
    train_parser.add_argument("--cost-bps", type=float, default=1.0)
    train_parser.add_argument("--folds", type=int, default=5)
    return parser


def _train(args: argparse.Namespace, db_path: str) -> None:
    prices = pd.read_csv(args.prices)
    if args.symbol and "symbol" in prices.columns:
        prices = prices[prices["symbol"].astype(str) == args.symbol]
    with closing(connect(db_path)) as connection:
        events = load_events(connection)
        news = load_news(connection)
    samples = build_training_frame(
        events, news, prices, args.horizon_minutes, args.cost_bps
    )
    result = walk_forward_validate(samples, args.folds)
    selected = select_best_strategy(result.predictions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output_dir / "oos_strategy_predictions.csv", index=False)
    selected.to_csv(output_dir / "oos_selected_trades.csv", index=False)
    write_evaluation_report(
        result.predictions, selected, result.fold_count,
        output_dir / "evaluation_report.md",
    )
    write_macro_outlook(db_path, output_dir / "macro_outlook.md")
    write_charts(db_path, result.predictions, selected, output_dir)
    print(f"Walk-forward folds: {result.fold_count}")
    print(f"Out-of-sample event decisions: {len(selected)}")
    print(f"Reports and charts: {output_dir.resolve()}")


def main() -> None:
    load_dotenv()
    args = _parser().parse_args()
    db_path = args.db or os.getenv("DATABASE_PATH", "data/macro.sqlite")
    if args.command == "ingest":
        actor_input = args.actor_input or os.getenv("APIFY_ACTOR_INPUT_FILE")
        if not actor_input:
            raise SystemExit("Pass --actor-input or set APIFY_ACTOR_INPUT_FILE.")
        event_count, news_count = ingest(
            db_path,
            actor_input,
            args.source_timezone,
            args.lookback_days,
            args.news_queries,
        )
        print(f"Stored {event_count} calendar events and {news_count} news articles in {db_path}.")
    elif args.command == "train":
        if args.horizon_minutes <= 0 or args.folds <= 0 or args.cost_bps < 0:
            raise SystemExit("--horizon-minutes and --folds must be positive; --cost-bps non-negative.")
        _train(args, db_path)


if __name__ == "__main__":
    main()
