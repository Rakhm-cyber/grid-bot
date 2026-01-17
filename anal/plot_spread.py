import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anal.pairtool.tests import build_spread


def _load_pair_data(symbol_a: str, symbol_b: str, data_dir: Path, tf: str) -> pd.DataFrame:
    path_a = data_dir / f"{symbol_a}_{tf}.csv"
    path_b = data_dir / f"{symbol_b}_{tf}.csv"
    if not path_a.exists():
        raise FileNotFoundError(f"Missing data file: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Missing data file: {path_b}")

    df_a = pd.read_csv(path_a, usecols=["ts_ms", "close"]).dropna()
    df_b = pd.read_csv(path_b, usecols=["ts_ms", "close"]).dropna()
    df_a = df_a.sort_values("ts_ms")
    df_b = df_b.sort_values("ts_ms")
    merged = df_a.merge(df_b, on="ts_ms", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("No overlapping timestamps between symbols.")

    merged["ts_iso"] = pd.to_datetime(merged["ts_ms"], unit="ms", utc=True)
    return merged


def plot_spread(
    merged: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    tf: str,
    output: Path,
    include_intercept: bool = True,
) -> Path:
    # Спред как разница доходностей (returns_a - returns_b).
    returns_a = merged["close_a"].pct_change()
    returns_b = merged["close_b"].pct_change()
    spread = (returns_a - returns_b).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged["ts_iso"], spread, label="spread", linewidth=1.2)
    ax.set_title(f"Spread: {symbol_a} vs {symbol_b} ({tf})")
    ax.set_xlabel("time")
    ax.set_ylabel("spread")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_spread_cumulative(
    merged: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    tf: str,
    output: Path,
) -> Path:
    # Накопительный график по спред-доходности: prod(1 + spread).
    returns_a = merged["close_a"].pct_change()
    returns_b = merged["close_b"].pct_change()
    spread = (returns_a - returns_b).fillna(0.0)
    cumulative = (1.0 + spread).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged["ts_iso"], cumulative, label="cumulative", linewidth=1.2)
    ax.set_title(f"Cumulative Spread: {symbol_a} vs {symbol_b} ({tf})")
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_ratio(
    merged: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    output: Path,
) -> Path:
    ratio = merged["close_a"] / merged["close_b"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged["ts_iso"], ratio, label="ratio", linewidth=1.2)
    ax.set_title(f"Ratio: {symbol_a} / {symbol_b}")
    ax.set_xlabel("time")
    ax.set_ylabel("ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot spread from historical CSV data.")
    parser.add_argument("--symbol-a", required=True, help="First symbol, e.g. ETHUSDT")
    parser.add_argument("--symbol-b", required=True, help="Second symbol, e.g. BTCUSDT")
    parser.add_argument("--data-dir", default=REPO_ROOT / "data" / "binance", type=Path)
    parser.add_argument("--tf", default="15m", help="Timeframe suffix, e.g. 15m")
    parser.add_argument("--output", default=REPO_ROOT / "anal" / "output" / "spread.png", type=Path)
    parser.add_argument(
        "--cum-output", default=REPO_ROOT / "anal" / "output" / "spread_cum.png", type=Path
    )
    parser.add_argument(
        "--ratio-output", default=REPO_ROOT / "anal" / "output" / "ratio.png", type=Path
    )
    parser.add_argument("--no-intercept", action="store_true")
    args = parser.parse_args()

    merged = _load_pair_data(args.symbol_a, args.symbol_b, args.data_dir, args.tf)
    out_spread = plot_spread(
        merged,
        args.symbol_a,
        args.symbol_b,
        args.tf,
        args.output,
        include_intercept=not args.no_intercept,
    )
    out_ratio = plot_ratio(merged, args.symbol_a, args.symbol_b, args.ratio_output)
    out_cum = plot_spread_cumulative(
        merged,
        args.symbol_a,
        args.symbol_b,
        args.tf,
        args.cum_output,
    )
    print(f"Wrote {out_spread}")
    print(f"Wrote {out_ratio}")
    print(f"Wrote {out_cum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
