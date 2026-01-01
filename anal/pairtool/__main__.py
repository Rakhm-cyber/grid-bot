from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import Config
from .loaders import load_from_binance
from .pipeline import run_analysis
from .reporting import write_html_report


def _load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.from_dict(raw)


def _handle_analyze(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    run_analysis(cfg, start=args.start, end=args.end)


def _handle_fetch(args: argparse.Namespace) -> None:
    raw = {
        "symbols": {"symbol1": args.symbol1, "symbol2": args.symbol2},
        "timeframe": args.tf,
        "data_source": "binance",
        "paths": {"input": args.output},
    }
    cfg = Config.from_dict(raw)
    load_from_binance(cfg, cfg.symbol1, start=args.start, end=args.end)
    load_from_binance(cfg, cfg.symbol2, start=args.start, end=args.end)


def _handle_report(args: argparse.Namespace) -> None:
    run_path = Path(args.output) / f"run_{args.run_id}"
    result_path = run_path / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    html_path = run_path / "report.html"
    summary = {
        "symbols": "/".join(result["metadata"]["symbols"]),
        "timeframe": result["metadata"]["timeframe"],
        "nobs": str(result["metadata"]["nobs"]),
        "score": str(result["tradeability"]["score"]),
        "decision": result["tradeability"]["decision"],
    }
    write_html_report(html_path, args.title, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pairtool")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run full analysis pipeline")
    analyze.add_argument("--config", required=True, help="Path to config.yaml")
    analyze.add_argument("--start", default=None, help="ISO start time for fetch mode")
    analyze.add_argument("--end", default=None, help="ISO end time for fetch mode")
    analyze.set_defaults(func=_handle_analyze)

    fetch = sub.add_parser("fetch", help="Fetch data and analyze")
    fetch.add_argument("--symbol1", required=True)
    fetch.add_argument("--symbol2", required=True)
    fetch.add_argument("--tf", required=True)
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=False)
    fetch.add_argument("--output", default="data/binance")
    fetch.set_defaults(func=_handle_fetch)

    report = sub.add_parser("report", help="Build report from run_id")
    report.add_argument("--run_id", required=True)
    report.add_argument("--output", default="anal/output")
    report.add_argument("--title", default="Pair Analytics Report")
    report.set_defaults(func=_handle_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
