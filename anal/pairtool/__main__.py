from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml

from .config import Config
from .loaders import load_from_binance
from .parallel import run_pair_task
from .pipeline import run_analysis
from .reporting import write_html_report_from_result


def _tqdm(iterable: Iterable, total: int):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, ncols=80)
    except Exception:
        return iterable


def _load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.from_dict(raw)


def _handle_analyze(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    if args.coins_file:
        coins = _read_coins(Path(args.coins_file))
        _run_all_pairs(cfg, coins, start=args.start, end=args.end, jobs=args.jobs, progress=not args.no_progress)
    else:
        run_analysis(cfg, start=args.start, end=args.end, progress=not args.no_progress)


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
    write_html_report_from_result(html_path, args.title, result)


def _read_coins(path: Path) -> List[str]:
    coins = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [c for c in coins if c]


def _run_all_pairs(
    cfg: Config,
    coins: Iterable[str],
    start: str | None,
    end: str | None,
    jobs: int,
    progress: bool,
) -> None:
    coins_list = list(coins)
    pairs: List[Tuple[str, str]] = []
    for i in range(len(coins_list)):
        for j in range(i + 1, len(coins_list)):
            pairs.append((coins_list[i], coins_list[j]))

    total = len(pairs)
    if total == 0:
        return

    if jobs <= 1:
        done = 0
        for s1, s2 in _tqdm(pairs, total=total):
            run_pair_task(asdict(cfg), s1, s2, start, end, progress=progress)
            done += 1
            print(f"[{done}/{total}] {s1}/{s2} done")
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_pair_task, asdict(cfg), s1, s2, start, end, False): (s1, s2)
            for s1, s2 in pairs
        }
        done = 0
        for future in _tqdm(as_completed(futures), total=total):
            s1, s2 = futures[future]
            _ = future.result()
            done += 1
            print(f"[{done}/{total}] {s1}/{s2} done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pairtool")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run full analysis pipeline")
    analyze.add_argument("--config", required=True, help="Path to config.yaml")
    analyze.add_argument("--start", default=None, help="ISO start time for fetch mode")
    analyze.add_argument("--end", default=None, help="ISO end time for fetch mode")
    analyze.add_argument("--coins-file", default=None, help="Path to coins.txt to run all pairs")
    analyze.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    analyze.add_argument("--no-progress", action="store_true", help="Disable progress bars")
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
