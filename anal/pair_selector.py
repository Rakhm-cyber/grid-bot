"""
Automatic pair selector.

Runs corr_pairs_excel scoring, filters by gate=1 and final_score > threshold,
backtests each passing pair, ranks by composite score (analytics + backtest),
and generates a config.yaml for the live bot.

Composite ranking weights:
    40% final_score  (cointegration + mean reversion analytics)
    25% sharpe       (backtest risk-adjusted return)
    20% pnl_pct      (total PnL %)
    15% win_rate     (trade success rate)

Usage:
    python -m anal.pair_selector --top 5 --output okx_pair_bot/config_auto.yaml
    python -m anal.pair_selector --top 10 --refresh-scores --fee-pct 0.0005
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anal.backtester import (
    _align_prices,
    _auto_spread_range,
    _read_close,
    run_backtest,
    DEFAULT_FEE_PCT,
)
from anal.optimizer import optimize

logger = logging.getLogger("pair_selector")


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------

def _composite_rank(result: Dict[str, Any]) -> float:
    """
    Composite ranking: analytics score + backtest metrics.

    Weights:
        40% final_score  — cointegration + mean reversion analytics
        25% sharpe       — backtest risk-adjusted return
        20% pnl_pct      — total PnL %
        15% win_rate     — trade success rate
    """
    w_score = 0.40
    w_sharpe = 0.25
    w_pnl = 0.20
    w_wr = 0.15

    # Normalize sharpe: clip to [-0.5, 1.0] range, scale to [0, 1]
    norm_sharpe = max(0.0, min(1.0, (result.get("sharpe", 0) + 0.5) / 1.5))
    # Normalize pnl_pct: clip to [-5%, +10%] range, scale to [0, 1]
    norm_pnl = max(0.0, min(1.0, (result.get("pnl_pct", 0) + 0.05) / 0.15))
    # Win rate: clip to [0, 1]
    norm_wr = max(0.0, min(1.0, float(result.get("win_rate", 0))))
    # Final score: clip to [0, 1]
    norm_score = max(0.0, min(1.0, float(result.get("final_score", 0.5))))

    return w_score * norm_score + w_sharpe * norm_sharpe + w_pnl * norm_pnl + w_wr * norm_wr


# ---------------------------------------------------------------------------
# Score loading: run corr_pairs_excel or read existing CSV
# ---------------------------------------------------------------------------

def _load_scores(
    coins_file: Path, data_dir: Path, output_csv: Path, refresh: bool
) -> pd.DataFrame:
    """
    Load pair scores from corr_pairs_excel output CSV.
    If refresh=True or file missing, regenerate.
    """
    if refresh or not output_csv.exists():
        logger.info("Running corr_pairs_excel to generate scores...")
        from anal.corr_pairs_excel import main as corr_main
        old_argv = sys.argv
        sys.argv = [
            "corr_pairs_excel",
            "--coins-file", str(coins_file),
            "--data-dir", str(data_dir),
            "--output", str(output_csv),
        ]
        try:
            corr_main()
        finally:
            sys.argv = old_argv

    df = pd.read_csv(output_csv)
    return df


def _filter_pairs(
    scores_df: pd.DataFrame,
    min_score: float = 0.4,
) -> pd.DataFrame:
    """Filter pairs where gate=1 and final_score > min_score."""
    filtered = scores_df[
        (scores_df["gate"] == 1) & (scores_df["final_score"] > min_score)
    ].copy()
    filtered = filtered.sort_values("final_score", ascending=False).reset_index(drop=True)
    return filtered


# ---------------------------------------------------------------------------
# Backtest all passing pairs
# ---------------------------------------------------------------------------

def _backtest_pair(
    coin1: str,
    coin2: str,
    data_dir: Path,
    *,
    use_optimizer: bool = False,
    levels: int = 21,
    max_steps: int = 5,
    per_step_usdt: float = 50.0,
    leverage: int = 10,
    stop_loss_pct: float = 0.10,
    trailing_stop_pct: float = 0.0,
    target_profit_pct: float = 0.0,
    max_cycles: int = 0,
    hold_minutes: int = 0,
    fee_pct: float = DEFAULT_FEE_PCT,
) -> Optional[Dict[str, Any]]:
    """Run backtest for a pair, return summary dict or None on error."""
    try:
        df1 = _read_close(data_dir, coin1)
        df2 = _read_close(data_dir, coin2)
        ts_ms, pa, pb = _align_prices(df1, df2)
        spread_min, spread_max = _auto_spread_range(pa, pb, "ratio", 1.0)
    except Exception as exc:
        logger.warning("Skip %s/%s data load error: %s", coin1, coin2, exc)
        return None

    if use_optimizer:
        try:
            opt = optimize(
                coin1, coin2, data_dir,
                levels_grid=[11, 15, 21],
                max_steps_grid=[3, 5],
                n_spread_variants=4,
                per_step_usdt=per_step_usdt,
                leverage=leverage,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
                target_profit_pct=target_profit_pct,
                max_cycles=max_cycles,
                hold_minutes=hold_minutes,
                fee_pct=fee_pct,
            )
            return {
                "coin1": coin1,
                "coin2": coin2,
                "sharpe": opt.val_sharpe,
                "pnl_pct": opt.val_pnl_pct,
                "max_dd_pct": opt.val_max_dd_pct,
                "win_rate": opt.val_win_rate,
                "num_trades": opt.val_num_trades,
                "params": opt.best_params,
            }
        except Exception as exc:
            logger.warning("Skip %s/%s optimizer error: %s", coin1, coin2, exc)
            return None
    else:
        try:
            res = run_backtest(
                coin1, coin2, data_dir,
                spread_min=spread_min,
                spread_max=spread_max,
                levels=levels,
                max_steps=max_steps,
                per_step_usdt=per_step_usdt,
                leverage=leverage,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
                target_profit_pct=target_profit_pct,
                max_cycles=max_cycles,
                hold_minutes=hold_minutes,
                fee_pct=fee_pct,
            )
            return {
                "coin1": coin1,
                "coin2": coin2,
                "sharpe": res.sharpe_ratio,
                "pnl_pct": res.total_pnl_pct,
                "max_dd_pct": res.max_drawdown_pct,
                "win_rate": res.win_rate,
                "num_trades": res.num_trades,
                "spread_min": spread_min,
                "spread_max": spread_max,
                "params": {
                    "spread_min": spread_min,
                    "spread_max": spread_max,
                    "levels": levels,
                    "max_steps": max_steps,
                },
            }
        except Exception as exc:
            logger.warning("Skip %s/%s backtest error: %s", coin1, coin2, exc)
            return None


# ---------------------------------------------------------------------------
# Config generation (matches okx_pair_bot/config.yaml format)
# ---------------------------------------------------------------------------

def _symbol_to_short(symbol: str) -> str:
    """DOTUSDT -> DOT (for config.yaml pairs format)."""
    for suffix in ("USDT", "USDC"):
        if symbol.upper().endswith(suffix):
            return symbol.upper()[: -len(suffix)]
    return symbol.upper()


def _generate_config(
    ranked: List[Dict[str, Any]],
    top_n: int,
    leverage: int = 10,
    hold_minutes: int = 0,
    stop_loss_pct: float = 0.10,
    trailing_stop_pct: float = 0.0,
    target_profit_pct: float = 0.0,
    max_cycles: int = 0,
) -> Dict[str, Any]:
    """Build a config dict matching okx_pair_bot/config.yaml schema."""
    selected = ranked[:top_n]
    if not selected:
        raise ValueError("No pairs to include in config")

    # Use the first pair's grid params as global defaults
    first_params = selected[0]["params"]

    config: Dict[str, Any] = {
        "dry_run": True,
        "use_demo": False,
        "leverage": leverage,
        "hold_minutes": hold_minutes,
        "stop_loss_pct": stop_loss_pct,
        "trailing_stop_pct": trailing_stop_pct,
        "target_profit_pct": target_profit_pct,
        "max_cycles": max_cycles,
        "check_interval_seconds": 2,
        "cooldown_seconds": 10,
        "grid": {
            "spread_min": float(first_params.get("spread_min", 0.0)),
            "spread_max": float(first_params.get("spread_max", 1.0)),
            "levels": int(first_params.get("levels", 21)),
            "max_position_steps": int(first_params.get("max_steps", 5)),
            "per_step_usdt": float(first_params.get("per_step_usdt", 50)),
            "beta": 1.0,
            "refresh_seconds": 10,
            "order_timeout_seconds": 180,
            "ratio_mode": "big_over_small",
        },
        "plot": {
            "enabled": True,
            "max_points": 2000,
            "host": "127.0.0.1",
            "port": 8099,
            "refresh_ms": 1000,
            "scale": 1,
            "auto_range": True,
            "history_days": 1,
            "history_timeframe": "1m",
            "update_seconds": 60,
        },
        "pairs": [],
    }

    for item in selected:
        config["pairs"].append({
            "long": _symbol_to_short(item["coin1"]),
            "short": _symbol_to_short(item["coin2"]),
        })

    return config


# ---------------------------------------------------------------------------
# Public API (used by pipeline.py)
# ---------------------------------------------------------------------------

def select_top_pairs(
    top_n: int = 3,
    min_score: float = 0.4,
    use_optimizer: bool = False,
) -> list[dict[str, str]]:
    """Return top pairs as [{"long": "DOT", "short": "FIL"}, ...] for pipeline."""
    repo_root = REPO_ROOT
    coins_file = repo_root / "coins.txt"
    data_dir = repo_root / "data" / "binance"
    scores_csv = repo_root / "anal" / "output" / "corr_pairs.csv"

    scores_df = _load_scores(coins_file, data_dir, scores_csv, refresh=False)
    filtered = _filter_pairs(scores_df, min_score)

    if filtered.empty:
        logger.warning("No pairs pass filter (gate=1, score>%.2f)", min_score)
        return []

    results: List[Dict[str, Any]] = []
    for _, row in filtered.iterrows():
        coin1, coin2 = row["coin1"], row["coin2"]
        logger.info("Backtesting %s / %s (score=%.3f)...", coin1, coin2, row["final_score"])
        bt = _backtest_pair(coin1, coin2, data_dir, use_optimizer=use_optimizer)
        if bt is not None:
            bt["final_score"] = float(row["final_score"])
            results.append(bt)

    if not results:
        return []

    ranked = sorted(results, key=_composite_rank, reverse=True)
    return [
        {"long": _symbol_to_short(r["coin1"]), "short": _symbol_to_short(r["coin2"])}
        for r in ranked[:top_n]
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description="Automatic pair selector")
    parser.add_argument("--coins-file", default=repo_root / "coins.txt", type=Path)
    parser.add_argument("--data-dir", default=repo_root / "data" / "binance", type=Path)
    parser.add_argument("--scores-csv", default=repo_root / "anal" / "output" / "corr_pairs.csv", type=Path)
    parser.add_argument("--output", default=repo_root / "okx_pair_bot" / "config_auto.yaml", type=Path)
    parser.add_argument("--top", type=int, default=5, help="Number of top pairs to select")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum final_score threshold")
    parser.add_argument("--refresh-scores", action="store_true", help="Re-run corr_pairs_excel")
    parser.add_argument("--use-optimizer", action="store_true", help="Use optimizer instead of default backtest")
    parser.add_argument("--per-step-usdt", type=float, default=50.0)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT)
    args = parser.parse_args()

    # Load and filter scores
    scores_df = _load_scores(args.coins_file, args.data_dir, args.scores_csv, args.refresh_scores)
    filtered = _filter_pairs(scores_df, args.min_score)
    logger.info(
        "Pairs passing filter (gate=1, score>%.2f): %d out of %d",
        args.min_score, len(filtered), len(scores_df),
    )

    if filtered.empty:
        logger.warning("No pairs pass the filter. Try lowering --min-score.")
        return 1

    # Build half_life lookup from scores CSV
    hl_lookup: Dict[str, float] = {}
    if "half_life_month" in scores_df.columns:
        for _, row in scores_df.iterrows():
            key = f"{row['coin1']}_{row['coin2']}"
            hl_lookup[key] = float(row["half_life_month"]) if not pd.isna(row["half_life_month"]) else 0.0

    # Backtest each pair
    results: List[Dict[str, Any]] = []
    for _, row in filtered.iterrows():
        coin1, coin2 = row["coin1"], row["coin2"]
        logger.info("Backtesting %s / %s (score=%.3f)...", coin1, coin2, row["final_score"])
        bt_res = _backtest_pair(
            coin1, coin2, args.data_dir,
            use_optimizer=args.use_optimizer,
            per_step_usdt=args.per_step_usdt,
            leverage=args.leverage,
            fee_pct=args.fee_pct,
        )
        if bt_res is not None:
            bt_res["final_score"] = float(row["final_score"])
            bt_res["half_life"] = hl_lookup.get(f"{coin1}_{coin2}", 0.0)
            results.append(bt_res)

    if not results:
        logger.warning("No pairs produced valid backtest results.")
        return 1

    # Rank by composite score (analytics + backtest)
    ranked = sorted(results, key=_composite_rank, reverse=True)

    # Print extended summary for manual review
    sep = "=" * 120
    print(sep)
    print(f"PAIR SELECTION RESULTS  (top {args.top} of {len(ranked)} backtested)")
    print(sep)
    print(
        f"  {'#':>3}  {'Coin1':>10}  {'Coin2':>10}  {'AScore':>6}  "
        f"{'Comp':>6}  {'Sharpe':>8}  {'PnL%':>7}  {'MaxDD%':>7}  "
        f"{'WR%':>5}  {'Trades':>6}  {'HL(bars)':>8}  {'SpreadRange':>20}"
    )
    for i, r in enumerate(ranked[: args.top]):
        comp = _composite_rank(r)
        hl_val = r.get("half_life", 0.0)
        sp_min = r.get("spread_min", 0.0)
        sp_max = r.get("spread_max", 0.0)
        spread_str = f"[{sp_min:.4f}, {sp_max:.4f}]"
        print(
            f"  {i + 1:>3}  {r['coin1']:>10}  {r['coin2']:>10}  "
            f"{r['final_score']:>6.3f}  {comp:>6.3f}  {r['sharpe']:>8.4f}  "
            f"{r['pnl_pct'] * 100:>6.2f}%  {r['max_dd_pct'] * 100:>6.2f}%  "
            f"{r['win_rate'] * 100:>4.1f}%  {r['num_trades']:>6d}  "
            f"{hl_val:>8.0f}  {spread_str:>20}"
        )
    print(sep)
    print()
    print("Legend: AScore=analytics final_score, Comp=composite rank,")
    print("       HL(bars)=half-life in 1m bars, SpreadRange=auto [5%, 95%] quantiles")
    print()

    # Generate config
    config = _generate_config(ranked, args.top, leverage=args.leverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info("Config written to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
