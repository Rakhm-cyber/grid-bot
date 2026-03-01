"""
Grid parameter optimizer with walk-forward validation.

Grid-searches spread_min, spread_max, levels, max_steps on a 70% train
window and validates on the remaining 30%.

Usage:
    python -m anal.optimizer --coin1 DOTUSDT --coin2 FILUSDT
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anal.backtester import (
    BacktestResult,
    GridBacktester,
    _align_prices,
    _auto_spread_range,
    _read_close,
    DEFAULT_FEE_PCT,
)


logger = logging.getLogger("optimizer")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    best_params: Dict[str, Any]
    train_sharpe: float
    train_pnl_pct: float
    val_sharpe: float
    val_pnl_pct: float
    val_max_dd_pct: float
    val_win_rate: float
    val_num_trades: int
    all_results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Search grid defaults
# ---------------------------------------------------------------------------

def _quantile_spread_candidates(
    spread: pd.Series, n_variants: int = 5
) -> List[Tuple[float, float]]:
    """Generate (spread_min, spread_max) candidates from quantile pairs."""
    lo_qs = np.linspace(0.02, 0.15, n_variants)
    hi_qs = np.linspace(0.85, 0.98, n_variants)
    candidates = []
    for lo_q, hi_q in zip(lo_qs, hi_qs):
        lo = float(spread.quantile(lo_q))
        hi = float(spread.quantile(hi_q))
        if hi > lo:
            candidates.append((lo, hi))
    return candidates


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

def optimize(
    coin1: str,
    coin2: str,
    data_dir: Path,
    *,
    # Search space
    levels_grid: Optional[List[int]] = None,
    max_steps_grid: Optional[List[int]] = None,
    n_spread_variants: int = 5,
    # Fixed params
    per_step_usdt: float = 50.0,
    leverage: int = 10,
    stop_loss_pct: float = 0.10,
    trailing_stop_pct: float = 0.0,
    target_profit_pct: float = 0.0,
    max_cycles: int = 0,
    hold_minutes: int = 0,
    beta: float = 1.0,
    spread_mode: str = "ratio",
    fee_pct: float = DEFAULT_FEE_PCT,
    # Walk-forward split
    train_pct: float = 0.70,
) -> OptResult:
    """Run grid search + walk-forward validation."""
    if levels_grid is None:
        levels_grid = [11, 15, 21, 31]
    if max_steps_grid is None:
        max_steps_grid = [3, 5, 7]

    # Load data
    df1 = _read_close(data_dir, coin1)
    df2 = _read_close(data_dir, coin2)
    ts_ms, pa, pb = _align_prices(df1, df2)
    n = len(pa)

    # Train / validation split
    split_idx = int(n * train_pct)
    if split_idx < 100 or n - split_idx < 50:
        raise ValueError(f"Not enough data for walk-forward: {n} bars, split at {split_idx}")

    pa_train = pa.iloc[:split_idx].reset_index(drop=True)
    pb_train = pb.iloc[:split_idx].reset_index(drop=True)
    ts_train = ts_ms.iloc[:split_idx].reset_index(drop=True)

    pa_val = pa.iloc[split_idx:].reset_index(drop=True)
    pb_val = pb.iloc[split_idx:].reset_index(drop=True)
    ts_val = ts_ms.iloc[split_idx:].reset_index(drop=True)

    # Spread range candidates from train data
    if spread_mode == "beta":
        train_spread = pa_train - beta * pb_train
    else:
        train_spread = pa_train / pb_train.replace(0, np.nan)
    train_spread = train_spread.dropna()
    spread_candidates = _quantile_spread_candidates(train_spread, n_spread_variants)

    if not spread_candidates:
        raise ValueError("Cannot generate spread range candidates from train data")

    # Build search space
    combos = list(itertools.product(spread_candidates, levels_grid, max_steps_grid))
    logger.info(
        "Optimizer: %d combinations (%d spread x %d levels x %d steps)",
        len(combos), len(spread_candidates), len(levels_grid), len(max_steps_grid),
    )

    all_results: List[Dict[str, Any]] = []
    best_sharpe = -np.inf
    best_combo = None
    best_train_res = None

    for (s_min, s_max), lvl, msteps in combos:
        try:
            bt = GridBacktester(
                pa_train, pb_train, ts_train,
                spread_min=s_min,
                spread_max=s_max,
                levels=lvl,
                max_steps=msteps,
                per_step_usdt=per_step_usdt,
                leverage=leverage,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
                target_profit_pct=target_profit_pct,
                max_cycles=max_cycles,
                hold_minutes=hold_minutes,
                beta=beta,
                spread_mode=spread_mode,
                fee_pct=fee_pct,
            )
            res = bt.run()
        except Exception as exc:
            logger.debug("Skip combo (%.4f,%.4f) lvl=%d steps=%d: %s", s_min, s_max, lvl, msteps, exc)
            continue

        row = {
            "spread_min": s_min,
            "spread_max": s_max,
            "levels": lvl,
            "max_steps": msteps,
            "train_sharpe": res.sharpe_ratio,
            "train_pnl_pct": res.total_pnl_pct,
            "train_num_trades": res.num_trades,
            "train_win_rate": res.win_rate,
            "train_max_dd_pct": res.max_drawdown_pct,
        }
        all_results.append(row)

        if res.sharpe_ratio > best_sharpe and res.num_trades >= 2:
            best_sharpe = res.sharpe_ratio
            best_combo = (s_min, s_max, lvl, msteps)
            best_train_res = res

    if best_combo is None:
        raise RuntimeError("No valid parameter combination found on train set")

    s_min, s_max, lvl, msteps = best_combo
    logger.info(
        "Best train: spread=[%.6f, %.6f] lvl=%d steps=%d sharpe=%.4f",
        s_min, s_max, lvl, msteps, best_sharpe,
    )

    # Validate on out-of-sample
    bt_val = GridBacktester(
        pa_val, pb_val, ts_val,
        spread_min=s_min,
        spread_max=s_max,
        levels=lvl,
        max_steps=msteps,
        per_step_usdt=per_step_usdt,
        leverage=leverage,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        target_profit_pct=target_profit_pct,
        max_cycles=max_cycles,
        hold_minutes=hold_minutes,
        beta=beta,
        spread_mode=spread_mode,
        fee_pct=fee_pct,
    )
    val_res = bt_val.run()

    logger.info(
        "Validation: sharpe=%.4f pnl=%.2f%% dd=%.2f%% trades=%d win=%.1f%%",
        val_res.sharpe_ratio,
        val_res.total_pnl_pct * 100,
        val_res.max_drawdown_pct * 100,
        val_res.num_trades,
        val_res.win_rate * 100,
    )

    best_params = {
        "spread_min": s_min,
        "spread_max": s_max,
        "levels": lvl,
        "max_steps": msteps,
        "per_step_usdt": per_step_usdt,
        "leverage": leverage,
        "stop_loss_pct": stop_loss_pct,
        "trailing_stop_pct": trailing_stop_pct,
        "target_profit_pct": target_profit_pct,
        "max_cycles": max_cycles,
        "hold_minutes": hold_minutes,
        "beta": beta,
        "spread_mode": spread_mode,
        "fee_pct": fee_pct,
    }

    return OptResult(
        best_params=best_params,
        train_sharpe=best_sharpe,
        train_pnl_pct=best_train_res.total_pnl_pct if best_train_res else 0.0,
        val_sharpe=val_res.sharpe_ratio,
        val_pnl_pct=val_res.total_pnl_pct,
        val_max_dd_pct=val_res.max_drawdown_pct,
        val_win_rate=val_res.win_rate,
        val_num_trades=val_res.num_trades,
        all_results=all_results,
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_opt_result(opt: OptResult, coin1: str, coin2: str) -> None:
    sep = "=" * 60
    print(sep)
    print(f"OPTIMIZATION RESULTS  {coin1} / {coin2}")
    print(sep)
    print("\nBest Parameters:")
    for k, v in opt.best_params.items():
        print(f"  {k:<25s}  {v}")
    print(f"\nTrain  Sharpe: {opt.train_sharpe:>10.4f}   PnL: {opt.train_pnl_pct * 100:.2f}%")
    print(f"Val    Sharpe: {opt.val_sharpe:>10.4f}   PnL: {opt.val_pnl_pct * 100:.2f}%")
    print(f"Val    MaxDD:  {opt.val_max_dd_pct * 100:>10.2f}%   WinRate: {opt.val_win_rate * 100:.1f}%")
    print(f"Val    Trades: {opt.val_num_trades:>10d}")
    print(sep)

    # Top-5 train results
    sorted_res = sorted(opt.all_results, key=lambda r: r["train_sharpe"], reverse=True)
    print("\nTop 5 Parameter Combos (by train Sharpe):")
    print(f"  {'Spread_min':>10}  {'Spread_max':>10}  {'Lvl':>4}  {'Steps':>5}  {'Sharpe':>8}  {'PnL%':>7}  {'Trades':>6}")
    for row in sorted_res[:5]:
        print(
            f"  {row['spread_min']:>10.6f}  {row['spread_max']:>10.6f}  "
            f"{row['levels']:>4d}  {row['max_steps']:>5d}  "
            f"{row['train_sharpe']:>8.4f}  {row['train_pnl_pct'] * 100:>6.2f}%  "
            f"{row['train_num_trades']:>6d}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description="Grid parameter optimizer")
    parser.add_argument("--coin1", required=True)
    parser.add_argument("--coin2", required=True)
    parser.add_argument("--data-dir", default=repo_root / "data" / "binance", type=Path)
    parser.add_argument("--levels", nargs="+", type=int, default=None, help="Levels to search, e.g. 11 15 21 31")
    parser.add_argument("--steps", nargs="+", type=int, default=None, help="Max steps to search, e.g. 3 5 7")
    parser.add_argument("--spread-variants", type=int, default=5)
    parser.add_argument("--per-step-usdt", type=float, default=50.0)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--trailing-stop", type=float, default=0.0)
    parser.add_argument("--target-profit", type=float, default=0.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--hold-minutes", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--spread-mode", choices=["ratio", "beta"], default="ratio")
    parser.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT)
    parser.add_argument("--train-pct", type=float, default=0.70)
    args = parser.parse_args()

    opt = optimize(
        args.coin1,
        args.coin2,
        args.data_dir,
        levels_grid=args.levels,
        max_steps_grid=args.steps,
        n_spread_variants=args.spread_variants,
        per_step_usdt=args.per_step_usdt,
        leverage=args.leverage,
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        target_profit_pct=args.target_profit,
        max_cycles=args.max_cycles,
        hold_minutes=args.hold_minutes,
        beta=args.beta,
        spread_mode=args.spread_mode,
        fee_pct=args.fee_pct,
        train_pct=args.train_pct,
    )
    _print_opt_result(opt, args.coin1, args.coin2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
