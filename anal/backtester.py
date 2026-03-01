"""
Grid pair backtester.

Simulates the exact same grid logic from okx_pair_bot/bot.py
on historical CSV data from data/binance/.

Usage:
    python -m anal.backtester --coin1 DOTUSDT --coin2 FILUSDT \
        --spread-min 1.335 --spread-max 1.472 --levels 21
"""
from __future__ import annotations

import argparse
import logging
import sys
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TF_SUFFIX = "1m"
TF_MINUTES = 1
DEFAULT_FEE_PCT = 0.001  # 0.1% per trade (taker)

logger = logging.getLogger("backtester")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_close(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Load close prices from CSV, return DataFrame with ts_ms and close."""
    path = data_dir / f"{symbol}_{TF_SUFFIX}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, usecols=["ts_ms", "close"])
    df = df.dropna(subset=["ts_ms", "close"]).sort_values("ts_ms").reset_index(drop=True)
    df["ts_ms"] = df["ts_ms"].astype(int)
    return df


def _align_prices(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Align two price series on common ts_ms timestamps."""
    merged = df1.merge(df2, on="ts_ms", how="inner", suffixes=("_1", "_2"))
    merged = merged.sort_values("ts_ms").reset_index(drop=True)
    return merged["ts_ms"], merged["close_1"], merged["close_2"]


# ---------------------------------------------------------------------------
# Grid builder (same as bot.py)
# ---------------------------------------------------------------------------

def build_grid(spread_min: float, spread_max: float, levels: int) -> List[float]:
    """Uniform grid from spread_min to spread_max with `levels` points."""
    if levels < 2:
        raise ValueError("Grid levels must be >= 2")
    step = (spread_max - spread_min) / (levels - 1)
    return [spread_min + i * step for i in range(levels)]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single round-trip cycle."""
    entry_bar: int
    exit_bar: int
    entry_ts: int
    exit_ts: int
    entry_spread: float
    exit_spread: float
    direction: str          # "long_spread" or "short_spread"
    max_steps: int
    pnl_usdt: float
    pnl_pct: float
    fees_usdt: float
    duration_minutes: float
    exit_reason: str        # "grid", "stop_loss", "trailing", "target", "timeout", "max_cycles"


@dataclass
class BacktestResult:
    total_pnl_usdt: float
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    num_cycles: int
    win_rate: float
    total_fees_usdt: float
    equity_curve: pd.Series
    trade_log: List[Dict[str, Any]]
    spread_series: pd.Series
    grid_levels: List[float]
    params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Position tracker (mirrors SymbolPosition from bot.py)
# ---------------------------------------------------------------------------

@dataclass
class _SimPosition:
    long_qty: float = 0.0
    long_avg: float = 0.0
    short_qty: float = 0.0
    short_avg: float = 0.0

    @property
    def net_qty(self) -> float:
        return self.long_qty - self.short_qty


def _update_avg(qty: float, avg: float, delta_qty: float, price: float) -> Tuple[float, float]:
    """Weighted-average price update (same logic as bot.py)."""
    if delta_qty > 0:
        new_qty = qty + delta_qty
        new_avg = (avg * qty + price * delta_qty) / new_qty if new_qty > 0 else 0.0
        return new_qty, new_avg
    new_qty = qty + delta_qty
    if new_qty <= 0:
        return 0.0, 0.0
    return new_qty, avg


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

class GridBacktester:
    """
    Replay historical prices through the same grid logic as run_grid_pair().

    Spread modes:
        ratio  -- spread = price_a / price_b   (default, matches bot.py)
        beta   -- spread = price_a - beta * price_b
    """

    def __init__(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        ts_ms: pd.Series,
        *,
        spread_min: float,
        spread_max: float,
        levels: int = 21,
        max_steps: int = 5,
        per_step_usdt: float = 50.0,
        leverage: int = 10,
        # risk
        stop_loss_pct: float = 0.03,
        trailing_stop_pct: float = 0.02,
        target_profit_pct: float = 0.01,
        max_cycles: int = 10,
        hold_minutes: int = 120,
        # spread model
        beta: float = 1.0,
        spread_mode: str = "ratio",
        # fees
        fee_pct: float = DEFAULT_FEE_PCT,
    ) -> None:
        self.price_a = price_a.reset_index(drop=True)
        self.price_b = price_b.reset_index(drop=True)
        self.ts_ms = ts_ms.reset_index(drop=True)
        self.n_bars = len(self.price_a)

        self.spread_min = spread_min
        self.spread_max = spread_max
        self.levels = levels
        self.max_steps = max_steps
        self.per_step_usdt = per_step_usdt
        self.leverage = leverage

        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.target_profit_pct = target_profit_pct
        self.max_cycles = max_cycles
        self.hold_minutes = hold_minutes

        self.beta = beta
        self.spread_mode = spread_mode
        self.fee_pct = fee_pct

        self.grid = build_grid(spread_min, spread_max, levels)

    # ------------------------------------------------------------------
    # Spread calculation
    # ------------------------------------------------------------------

    def _calc_spread(self, pa: float, pb: float) -> float:
        if self.spread_mode == "beta":
            return pa - self.beta * pb
        return pa / pb if pb != 0 else 0.0

    def _calc_spread_series(self) -> pd.Series:
        if self.spread_mode == "beta":
            return self.price_a - self.beta * self.price_b
        return self.price_a / self.price_b.replace(0, np.nan)

    # ------------------------------------------------------------------
    # PnL helpers
    # ------------------------------------------------------------------

    def _pair_pnl(
        self, pos_a: _SimPosition, pos_b: _SimPosition, pa: float, pb: float
    ) -> float:
        """Unrealised PnL in USDT (same formula as bot.py _pair_pnl_pct, but absolute)."""
        pnl_a = (pa - pos_a.long_avg) * pos_a.long_qty
        pnl_a += (pos_a.short_avg - pa) * pos_a.short_qty
        pnl_b = (pb - pos_b.long_avg) * pos_b.long_qty
        pnl_b += (pos_b.short_avg - pb) * pos_b.short_qty
        return pnl_a + pnl_b

    def _pair_pnl_pct(
        self,
        pos_a: _SimPosition,
        pos_b: _SimPosition,
        pa: float,
        pb: float,
        pos_steps: int,
    ) -> float:
        if pos_steps == 0:
            return 0.0
        exposure = self.per_step_usdt * 2.0 * abs(pos_steps)
        return self._pair_pnl(pos_a, pos_b, pa, pb) / exposure

    # ------------------------------------------------------------------
    # Simulated trade execution
    # ------------------------------------------------------------------

    def _sim_buy(
        self, pos: _SimPosition, price: float, notional: float
    ) -> Tuple[float, float, float]:
        """Buy on simulation: returns (amount, fee_usdt, realized_pnl)."""
        amount = notional / price
        fee = notional * self.fee_pct
        realized = 0.0
        if pos.short_qty > 0:
            close_amt = min(amount, pos.short_qty)
            # Realized PnL from closing short: sold high (short_avg), buying back low (price)
            realized = close_amt * (pos.short_avg - price)
            pos.short_qty, pos.short_avg = _update_avg(
                pos.short_qty, pos.short_avg, -close_amt, price
            )
            remaining = amount - close_amt
            if remaining > 0:
                pos.long_qty, pos.long_avg = _update_avg(
                    pos.long_qty, pos.long_avg, remaining, price
                )
        else:
            pos.long_qty, pos.long_avg = _update_avg(
                pos.long_qty, pos.long_avg, amount, price
            )
        return amount, fee, realized

    def _sim_sell(
        self, pos: _SimPosition, price: float, notional: float
    ) -> Tuple[float, float, float]:
        """Sell on simulation: returns (amount, fee_usdt, realized_pnl)."""
        amount = notional / price
        fee = notional * self.fee_pct
        realized = 0.0
        if pos.long_qty > 0:
            close_amt = min(amount, pos.long_qty)
            # Realized PnL from closing long: bought low (long_avg), selling high (price)
            realized = close_amt * (price - pos.long_avg)
            pos.long_qty, pos.long_avg = _update_avg(
                pos.long_qty, pos.long_avg, -close_amt, price
            )
            remaining = amount - close_amt
            if remaining > 0:
                pos.short_qty, pos.short_avg = _update_avg(
                    pos.short_qty, pos.short_avg, remaining, price
                )
        else:
            pos.short_qty, pos.short_avg = _update_avg(
                pos.short_qty, pos.short_avg, amount, price
            )
        return amount, fee, realized

    def _close_position(
        self, pos: _SimPosition, price: float
    ) -> Tuple[float, float]:
        """Close all open quantities, return (total_fee, realized_pnl)."""
        fee = 0.0
        realized = 0.0
        if pos.long_qty > 0:
            notional = pos.long_qty * price
            fee += notional * self.fee_pct
            realized += pos.long_qty * (price - pos.long_avg)
            pos.long_qty, pos.long_avg = 0.0, 0.0
        if pos.short_qty > 0:
            notional = pos.short_qty * price
            fee += notional * self.fee_pct
            realized += pos.short_qty * (pos.short_avg - price)
            pos.short_qty, pos.short_avg = 0.0, 0.0
        return fee, realized

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the full backtest, return BacktestResult."""
        grid = self.grid
        spread_series = self._calc_spread_series()

        # State
        pos_a = _SimPosition()
        pos_b = _SimPosition()
        pos_steps = 0
        last_idx: Optional[int] = None
        cycles = 0
        best_pnl: Optional[float] = None

        # Tracking
        total_fees = 0.0
        equity = np.zeros(self.n_bars)
        trade_log: List[Dict[str, Any]] = []

        # Current cycle tracking
        cycle_entry_bar: Optional[int] = None
        cycle_entry_spread: Optional[float] = None
        cycle_max_steps = 0
        cycle_direction: Optional[str] = None
        cycle_fees = 0.0
        cycle_realized_pnl = 0.0  # accumulated realized PnL from grid crossings
        cycle_start_ts: Optional[int] = None

        def _record_exit(bar: int, reason: str) -> None:
            nonlocal cycles, cycle_entry_bar, cycle_entry_spread
            nonlocal cycle_max_steps, cycle_direction, cycle_fees, cycle_start_ts
            nonlocal total_fees, pos_steps, best_pnl, cycle_realized_pnl

            pa = float(self.price_a.iloc[bar])
            pb = float(self.price_b.iloc[bar])
            # Close remaining positions and capture realized PnL
            fee_a, rpnl_a = self._close_position(pos_a, pa)
            fee_b, rpnl_b = self._close_position(pos_b, pb)
            close_fees = fee_a + fee_b
            cycle_fees += close_fees
            total_fees += close_fees
            cycle_realized_pnl += rpnl_a + rpnl_b
            pnl_after_fees = cycle_realized_pnl - cycle_fees
            exposure = self.per_step_usdt * 2.0 * cycle_max_steps if cycle_max_steps > 0 else 1.0

            ts_exit = int(self.ts_ms.iloc[bar])
            duration = (ts_exit - (cycle_start_ts or ts_exit)) / 60_000.0

            trade_log.append({
                "entry_bar": cycle_entry_bar,
                "exit_bar": bar,
                "entry_ts": cycle_start_ts,
                "exit_ts": ts_exit,
                "entry_spread": cycle_entry_spread,
                "exit_spread": float(spread_series.iloc[bar]),
                "direction": cycle_direction or "unknown",
                "max_steps": cycle_max_steps,
                "pnl_usdt": float(pnl_after_fees),
                "pnl_pct": float(pnl_after_fees / exposure) if exposure > 0 else 0.0,
                "fees_usdt": float(cycle_fees),
                "duration_minutes": float(duration),
                "exit_reason": reason,
            })

            # Reset state
            pos_steps = 0
            last_idx_reset = None
            best_pnl = None
            cycle_entry_bar = None
            cycle_entry_spread = None
            cycle_max_steps = 0
            cycle_direction = None
            cycle_fees = 0.0
            cycle_realized_pnl = 0.0
            cycle_start_ts = None
            cycles += 1
            return last_idx_reset

        # ----- Main loop over bars -----
        for bar in range(self.n_bars):
            pa = float(self.price_a.iloc[bar])
            pb = float(self.price_b.iloc[bar])
            spread = self._calc_spread(pa, pb)
            ts = int(self.ts_ms.iloc[bar])

            # Spread out of range -- skip (same as bot.py)
            if spread < self.spread_min or spread > self.spread_max:
                equity[bar] = equity[bar - 1] if bar > 0 else 0.0
                continue

            idx = bisect_right(grid, spread) - 1
            idx = max(0, min(idx, len(grid) - 1))

            if last_idx is None:
                last_idx = idx
                equity[bar] = equity[bar - 1] if bar > 0 else 0.0
                continue

            # ---- Grid crossing logic (exact mirror of bot.py) ----
            if idx > last_idx:
                # Spread moved UP -> sell base, buy hedge
                steps = idx - last_idx
                for _ in range(steps):
                    if pos_steps <= -self.max_steps:
                        break
                    prev_abs = abs(pos_steps)
                    _, fee_s, rpnl_s = self._sim_sell(pos_a, pa, self.per_step_usdt)
                    _, fee_b, rpnl_b = self._sim_buy(pos_b, pb, self.per_step_usdt)
                    step_fee = fee_s + fee_b
                    total_fees += step_fee
                    cycle_fees += step_fee
                    cycle_realized_pnl += rpnl_s + rpnl_b
                    pos_steps -= 1

                    # Track cycle start
                    if cycle_entry_bar is None:
                        cycle_entry_bar = bar
                        cycle_entry_spread = spread
                        cycle_start_ts = ts
                        cycle_direction = "short_spread"

                    cycle_max_steps = max(cycle_max_steps, abs(pos_steps))

                    if abs(pos_steps) < prev_abs:
                        cycles += 1

            elif idx < last_idx:
                # Spread moved DOWN -> buy base, sell hedge
                steps = last_idx - idx
                for _ in range(steps):
                    if pos_steps >= self.max_steps:
                        break
                    prev_abs = abs(pos_steps)
                    _, fee_b, rpnl_b = self._sim_buy(pos_a, pa, self.per_step_usdt)
                    _, fee_s, rpnl_s = self._sim_sell(pos_b, pb, self.per_step_usdt)
                    step_fee = fee_b + fee_s
                    total_fees += step_fee
                    cycle_fees += step_fee
                    cycle_realized_pnl += rpnl_b + rpnl_s
                    pos_steps += 1

                    if cycle_entry_bar is None:
                        cycle_entry_bar = bar
                        cycle_entry_spread = spread
                        cycle_start_ts = ts
                        cycle_direction = "long_spread"

                    cycle_max_steps = max(cycle_max_steps, abs(pos_steps))

                    if abs(pos_steps) < prev_abs:
                        cycles += 1

            last_idx = idx

            # ---- Risk management ----
            # Total cycle PnL = realized from grid crossings + unrealized from open positions - fees
            unrealized_pnl = self._pair_pnl(pos_a, pos_b, pa, pb)
            cycle_total_pnl = cycle_realized_pnl + unrealized_pnl - cycle_fees
            exposure = self.per_step_usdt * 2.0 * abs(pos_steps) if pos_steps != 0 else 1.0
            pnl_pct = cycle_total_pnl / exposure if pos_steps != 0 else 0.0
            if best_pnl is None or pnl_pct > best_pnl:
                best_pnl = pnl_pct

            need_exit = False
            exit_reason = ""

            if pos_steps != 0:
                # Stop-loss
                if self.stop_loss_pct > 0 and pnl_pct <= -self.stop_loss_pct:
                    need_exit = True
                    exit_reason = "stop_loss"
                # Target profit
                elif self.target_profit_pct > 0 and pnl_pct >= self.target_profit_pct:
                    need_exit = True
                    exit_reason = "target"
                # Trailing stop
                elif (
                    self.trailing_stop_pct > 0
                    and best_pnl is not None
                    and pnl_pct >= 0
                    and pnl_pct <= best_pnl - self.trailing_stop_pct
                ):
                    need_exit = True
                    exit_reason = "trailing"
                # Hold timeout (exit only if in profit)
                elif self.hold_minutes > 0 and cycle_start_ts is not None:
                    elapsed_min = (ts - cycle_start_ts) / 60_000.0
                    if elapsed_min >= self.hold_minutes and pnl_pct >= 0:
                        need_exit = True
                        exit_reason = "timeout"
                # Max cycles
                if self.max_cycles > 0 and cycles >= self.max_cycles:
                    need_exit = True
                    exit_reason = "max_cycles"

            if need_exit and pos_steps != 0:
                last_idx = _record_exit(bar, exit_reason)

            # Equity curve: sum of closed trades + current cycle PnL
            closed_pnl = sum(t["pnl_usdt"] for t in trade_log)
            current_cycle_pnl = (cycle_realized_pnl + unrealized_pnl - cycle_fees) if pos_steps != 0 else 0.0
            equity[bar] = closed_pnl + current_cycle_pnl

        # ---- Close any remaining position at the last bar ----
        if pos_steps != 0:
            _record_exit(self.n_bars - 1, "end_of_data")

        # ---- Compute summary metrics ----
        equity_series = pd.Series(equity, name="equity_usdt")
        initial_capital = self.per_step_usdt * self.leverage * self.max_steps * 2.0
        total_pnl = sum(t["pnl_usdt"] for t in trade_log)
        total_pnl_pct = total_pnl / initial_capital if initial_capital > 0 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_dd = float(np.min(drawdown))
        max_dd_pct = max_dd / initial_capital if initial_capital > 0 else 0.0

        # Sharpe ratio (per-bar returns, annualised)
        returns = np.diff(equity)
        if len(returns) > 1 and np.std(returns) > 0:
            bars_per_year = 365.25 * 24 * 60 / TF_MINUTES
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0

        # Win rate
        wins = sum(1 for t in trade_log if t["pnl_usdt"] > 0)
        win_rate = wins / len(trade_log) if trade_log else 0.0

        params = {
            "spread_min": self.spread_min,
            "spread_max": self.spread_max,
            "levels": self.levels,
            "max_steps": self.max_steps,
            "per_step_usdt": self.per_step_usdt,
            "leverage": self.leverage,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "target_profit_pct": self.target_profit_pct,
            "max_cycles": self.max_cycles,
            "hold_minutes": self.hold_minutes,
            "beta": self.beta,
            "spread_mode": self.spread_mode,
            "fee_pct": self.fee_pct,
        }

        return BacktestResult(
            total_pnl_usdt=float(total_pnl),
            total_pnl_pct=float(total_pnl_pct),
            max_drawdown_pct=float(max_dd_pct),
            sharpe_ratio=sharpe,
            num_trades=len(trade_log),
            num_cycles=cycles,
            win_rate=win_rate,
            total_fees_usdt=float(total_fees),
            equity_curve=equity_series,
            trade_log=trade_log,
            spread_series=spread_series,
            grid_levels=list(grid),
            params=params,
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_backtest(
    coin1: str,
    coin2: str,
    data_dir: Path,
    *,
    spread_min: float,
    spread_max: float,
    levels: int = 21,
    max_steps: int = 5,
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
) -> BacktestResult:
    """Load CSVs and run backtest, return BacktestResult."""
    df1 = _read_close(data_dir, coin1)
    df2 = _read_close(data_dir, coin2)
    ts_ms, pa, pb = _align_prices(df1, df2)

    bt = GridBacktester(
        pa, pb, ts_ms,
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
        beta=beta,
        spread_mode=spread_mode,
        fee_pct=fee_pct,
    )
    return bt.run()


def _auto_spread_range(
    price_a: pd.Series, price_b: pd.Series, spread_mode: str, beta: float
) -> Tuple[float, float]:
    """Estimate spread_min/max from historical spread quantiles."""
    if spread_mode == "beta":
        spread = price_a - beta * price_b
    else:
        spread = price_a / price_b.replace(0, np.nan)
    spread = spread.dropna()
    q_lo = float(spread.quantile(0.05))
    q_hi = float(spread.quantile(0.95))
    return q_lo, q_hi


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_result(res: BacktestResult) -> None:
    sep = "-" * 50
    print(sep)
    print("BACKTEST RESULTS")
    print(sep)
    print(f"  Total PnL (USDT):      {res.total_pnl_usdt:>12.2f}")
    print(f"  Total PnL (%):         {res.total_pnl_pct * 100:>12.2f}%")
    print(f"  Max Drawdown (%):      {res.max_drawdown_pct * 100:>12.2f}%")
    print(f"  Sharpe Ratio:          {res.sharpe_ratio:>12.4f}")
    print(f"  Num Trades:            {res.num_trades:>12d}")
    print(f"  Num Cycles:            {res.num_cycles:>12d}")
    print(f"  Win Rate:              {res.win_rate * 100:>12.2f}%")
    print(f"  Total Fees (USDT):     {res.total_fees_usdt:>12.2f}")
    print(sep)
    if res.trade_log:
        print("\nTRADE LOG:")
        print(f"  {'#':>3}  {'Dir':>14}  {'Steps':>5}  {'PnL$':>8}  {'PnL%':>8}  {'Fees$':>7}  {'Dur(min)':>9}  {'Reason'}")
        for i, t in enumerate(res.trade_log):
            print(
                f"  {i + 1:>3}  {t['direction']:>14}  {t['max_steps']:>5}  "
                f"{t['pnl_usdt']:>8.2f}  {t['pnl_pct'] * 100:>7.2f}%  "
                f"{t['fees_usdt']:>7.2f}  {t['duration_minutes']:>9.1f}  {t['exit_reason']}"
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
    parser = argparse.ArgumentParser(description="Grid pair backtester")
    parser.add_argument("--coin1", required=True, help="Base symbol, e.g. DOTUSDT")
    parser.add_argument("--coin2", required=True, help="Hedge symbol, e.g. FILUSDT")
    parser.add_argument("--data-dir", default=repo_root / "data" / "binance", type=Path)
    parser.add_argument("--spread-min", type=float, default=None, help="Grid lower bound (auto if omitted)")
    parser.add_argument("--spread-max", type=float, default=None, help="Grid upper bound (auto if omitted)")
    parser.add_argument("--levels", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=5)
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
    args = parser.parse_args()

    # Load data
    df1 = _read_close(args.data_dir, args.coin1)
    df2 = _read_close(args.data_dir, args.coin2)
    ts_ms, pa, pb = _align_prices(df1, df2)

    # Auto-detect spread range if not given
    spread_min = args.spread_min
    spread_max = args.spread_max
    if spread_min is None or spread_max is None:
        auto_lo, auto_hi = _auto_spread_range(pa, pb, args.spread_mode, args.beta)
        if spread_min is None:
            spread_min = auto_lo
        if spread_max is None:
            spread_max = auto_hi
        logger.info("Auto spread range: [%.6f, %.6f]", spread_min, spread_max)

    logger.info(
        "Running backtest %s / %s | grid [%.6f .. %.6f] %d levels",
        args.coin1, args.coin2, spread_min, spread_max, args.levels,
    )

    bt = GridBacktester(
        pa, pb, ts_ms,
        spread_min=spread_min,
        spread_max=spread_max,
        levels=args.levels,
        max_steps=args.max_steps,
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
    )
    result = bt.run()
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
