"""Tests for okx_pair_bot.bot grid helper functions."""
from __future__ import annotations

import math

import pytest

from okx_pair_bot.bot import (
    SymbolPosition,
    _apply_fill,
    _build_grid,
    _tv_to_okx_symbol,
    _update_avg,
)


# ---------------------------------------------------------------------------
# _build_grid
# ---------------------------------------------------------------------------

class TestBuildGrid:
    def test_correct_number_of_levels(self):
        grid = _build_grid(1.0, 2.0, 5)
        assert len(grid) == 5

    def test_evenly_spaced(self):
        grid = _build_grid(0.0, 1.0, 11)
        expected_step = 0.1
        for i in range(1, len(grid)):
            assert math.isclose(grid[i] - grid[i - 1], expected_step, rel_tol=1e-9)

    def test_boundary_values(self):
        grid = _build_grid(-0.05, 0.05, 7)
        assert math.isclose(grid[0], -0.05, rel_tol=1e-9)
        assert math.isclose(grid[-1], 0.05, rel_tol=1e-9)

    def test_two_levels(self):
        grid = _build_grid(10.0, 20.0, 2)
        assert len(grid) == 2
        assert math.isclose(grid[0], 10.0)
        assert math.isclose(grid[1], 20.0)

    def test_raises_on_less_than_two_levels(self):
        with pytest.raises(ValueError, match="Grid levels must be >= 2"):
            _build_grid(0.0, 1.0, 1)

    def test_single_point_range(self):
        grid = _build_grid(5.0, 5.0, 3)
        assert len(grid) == 3
        for val in grid:
            assert math.isclose(val, 5.0)


# ---------------------------------------------------------------------------
# _update_avg
# ---------------------------------------------------------------------------

class TestUpdateAvg:
    def test_add_to_empty_position(self):
        qty, avg = _update_avg(0.0, 0.0, 10.0, 100.0)
        assert qty == 10.0
        assert avg == 100.0

    def test_average_with_existing_position(self):
        # 10 @ 100, then add 10 @ 200 => 20 @ 150
        qty, avg = _update_avg(10.0, 100.0, 10.0, 200.0)
        assert qty == 20.0
        assert math.isclose(avg, 150.0)

    def test_partial_reduce(self):
        # 20 @ 150, reduce by 5 => 15 @ 150 (avg unchanged)
        qty, avg = _update_avg(20.0, 150.0, -5.0, 200.0)
        assert qty == 15.0
        assert avg == 150.0

    def test_full_reduce(self):
        qty, avg = _update_avg(10.0, 100.0, -10.0, 120.0)
        assert qty == 0.0
        assert avg == 0.0

    def test_over_reduce_clamps_to_zero(self):
        qty, avg = _update_avg(5.0, 100.0, -10.0, 90.0)
        assert qty == 0.0
        assert avg == 0.0

    def test_weighted_average_precision(self):
        # 3 @ 50, add 7 @ 80 => 10 @ 71
        qty, avg = _update_avg(3.0, 50.0, 7.0, 80.0)
        assert qty == 10.0
        assert math.isclose(avg, 71.0)


# ---------------------------------------------------------------------------
# _tv_to_okx_symbol
# ---------------------------------------------------------------------------

class TestTvToOkxSymbol:
    def test_plain_usdt_symbol(self):
        assert _tv_to_okx_symbol("BTCUSDT") == "BTC/USDT:USDT"

    def test_okx_prefix_strip(self):
        assert _tv_to_okx_symbol("OKX:ETHUSDT") == "ETH/USDT:USDT"

    def test_perp_suffix_strip(self):
        assert _tv_to_okx_symbol("OKX:SOLUSDT.P") == "SOL/USDT:USDT"

    def test_usdc_quote(self):
        assert _tv_to_okx_symbol("BTCUSDC") == "BTC/USDC:USDC"

    def test_bare_symbol_defaults_to_usdt(self):
        assert _tv_to_okx_symbol("DOGE") == "DOGE/USDT:USDT"

    def test_case_insensitive(self):
        assert _tv_to_okx_symbol("btcusdt") == "BTC/USDT:USDT"

    def test_whitespace_stripped(self):
        assert _tv_to_okx_symbol("  ETHUSDT  ") == "ETH/USDT:USDT"

    def test_unsupported_base_usdt(self):
        with pytest.raises(ValueError, match="Unsupported base symbol"):
            _tv_to_okx_symbol("USDT")

    def test_unsupported_base_usdc(self):
        with pytest.raises(ValueError, match="Unsupported base symbol"):
            _tv_to_okx_symbol("USDC")


# ---------------------------------------------------------------------------
# _apply_fill
# ---------------------------------------------------------------------------

class TestApplyFill:
    def test_buy_open_long(self):
        pos = SymbolPosition()
        _apply_fill(pos, "buy", 5.0, 100.0, reduce_only=False)
        assert pos.long_qty == 5.0
        assert pos.long_avg == 100.0
        assert pos.short_qty == 0.0

    def test_sell_open_short(self):
        pos = SymbolPosition()
        _apply_fill(pos, "sell", 3.0, 200.0, reduce_only=False)
        assert pos.short_qty == 3.0
        assert pos.short_avg == 200.0
        assert pos.long_qty == 0.0

    def test_buy_reduce_short(self):
        pos = SymbolPosition(short_qty=10.0, short_avg=150.0)
        _apply_fill(pos, "buy", 4.0, 140.0, reduce_only=True)
        assert pos.short_qty == 6.0
        assert pos.short_avg == 150.0  # avg unchanged on reduce

    def test_sell_reduce_long(self):
        pos = SymbolPosition(long_qty=8.0, long_avg=100.0)
        _apply_fill(pos, "sell", 3.0, 120.0, reduce_only=True)
        assert pos.long_qty == 5.0
        assert pos.long_avg == 100.0

    def test_sequential_fills_update_avg(self):
        pos = SymbolPosition()
        _apply_fill(pos, "buy", 10.0, 100.0, reduce_only=False)
        _apply_fill(pos, "buy", 10.0, 200.0, reduce_only=False)
        assert pos.long_qty == 20.0
        assert math.isclose(pos.long_avg, 150.0)


# ---------------------------------------------------------------------------
# _pair_pnl_pct (unit-level: formula check with mock-like approach)
# ---------------------------------------------------------------------------

class TestPairPnlPct:
    """Test PnL calculation logic without requiring exchange connectivity.

    We replicate the formula used in _pair_pnl_pct:
        pnl_a = (price - long_avg) * long_qty * contract_size
               + (short_avg - price) * short_qty * contract_size
        pnl_b = same for symbol B
        pnl_pct = (pnl_a + pnl_b) / (per_step_usdt * 2 * abs(pos_steps))
    """

    @staticmethod
    def _pnl_formula(
        price_a: float,
        pos_a: SymbolPosition,
        price_b: float,
        pos_b: SymbolPosition,
        per_step_usdt: float,
        pos_steps: int,
    ) -> float:
        if pos_steps == 0:
            return 0.0
        pnl_a = (price_a - pos_a.long_avg) * pos_a.long_qty * pos_a.contract_size
        pnl_a += (pos_a.short_avg - price_a) * pos_a.short_qty * pos_a.contract_size
        pnl_b = (price_b - pos_b.long_avg) * pos_b.long_qty * pos_b.contract_size
        pnl_b += (pos_b.short_avg - price_b) * pos_b.short_qty * pos_b.contract_size
        exposure = per_step_usdt * 2.0 * abs(pos_steps)
        return (pnl_a + pnl_b) / exposure

    def test_zero_steps_returns_zero(self):
        pos_a = SymbolPosition(long_qty=5.0, long_avg=100.0)
        pos_b = SymbolPosition(short_qty=5.0, short_avg=200.0)
        assert self._pnl_formula(110.0, pos_a, 190.0, pos_b, 50.0, 0) == 0.0

    def test_positive_pnl(self):
        pos_a = SymbolPosition(long_qty=1.0, long_avg=100.0, contract_size=1.0)
        pos_b = SymbolPosition(short_qty=1.0, short_avg=200.0, contract_size=1.0)
        # price_a goes to 110 => pnl_a = 10
        # price_b goes to 190 => pnl_b = 10
        pnl = self._pnl_formula(110.0, pos_a, 190.0, pos_b, 50.0, 1)
        expected = 20.0 / (50.0 * 2.0 * 1)  # 0.2
        assert math.isclose(pnl, expected)

    def test_negative_pnl(self):
        pos_a = SymbolPosition(long_qty=1.0, long_avg=100.0, contract_size=1.0)
        pos_b = SymbolPosition(short_qty=1.0, short_avg=200.0, contract_size=1.0)
        # price_a drops to 90 => pnl_a = -10
        # price_b rises to 210 => pnl_b = -10
        pnl = self._pnl_formula(90.0, pos_a, 210.0, pos_b, 50.0, 1)
        expected = -20.0 / 100.0
        assert math.isclose(pnl, expected)

    def test_contract_size_factor(self):
        pos_a = SymbolPosition(long_qty=2.0, long_avg=50.0, contract_size=0.01)
        pos_b = SymbolPosition()
        pnl = self._pnl_formula(60.0, pos_a, 100.0, pos_b, 50.0, 1)
        # pnl_a = (60-50) * 2 * 0.01 = 0.2
        expected = 0.2 / 100.0
        assert math.isclose(pnl, expected)
