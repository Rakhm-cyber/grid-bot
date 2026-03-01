"""Tests for anal.pair_selector composite ranking."""
from __future__ import annotations

import pytest

from anal.pair_selector import _composite_rank


# ---------------------------------------------------------------------------
# _composite_rank
# ---------------------------------------------------------------------------


class TestCompositeRank:
    """Test the composite ranking function."""

    def test_perfect_pair(self):
        """A pair with ideal metrics should score close to 1.0."""
        result = {
            "final_score": 1.0,
            "sharpe": 1.0,
            "pnl_pct": 0.10,
            "win_rate": 1.0,
        }
        score = _composite_rank(result)
        assert 0.95 <= score <= 1.0

    def test_terrible_pair(self):
        """A pair with worst-case metrics should score close to 0."""
        result = {
            "final_score": 0.0,
            "sharpe": -0.5,
            "pnl_pct": -0.05,
            "win_rate": 0.0,
        }
        score = _composite_rank(result)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_average_pair(self):
        """A pair with middle-of-the-road metrics."""
        result = {
            "final_score": 0.5,
            "sharpe": 0.0,
            "pnl_pct": 0.0,
            "win_rate": 0.5,
        }
        score = _composite_rank(result)
        # 0.40*0.5 + 0.25*(0.5/1.5) + 0.20*(0.05/0.15) + 0.15*0.5
        expected = 0.40 * 0.5 + 0.25 * (0.5 / 1.5) + 0.20 * (0.05 / 0.15) + 0.15 * 0.5
        assert score == pytest.approx(expected, abs=1e-6)

    def test_weights_sum_to_one(self):
        """Verify that internal weights sum to 1.0."""
        # When all normalized components = 1.0, score should be 1.0
        result = {
            "final_score": 1.0,
            "sharpe": 1.0,
            "pnl_pct": 0.10,
            "win_rate": 1.0,
        }
        score = _composite_rank(result)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_clipping_sharpe_upper(self):
        """Very high Sharpe should be clipped to 1.0 normalized."""
        result_high = {"final_score": 0.0, "sharpe": 5.0, "pnl_pct": 0.0, "win_rate": 0.0}
        result_cap = {"final_score": 0.0, "sharpe": 1.0, "pnl_pct": 0.0, "win_rate": 0.0}
        assert _composite_rank(result_high) == _composite_rank(result_cap)

    def test_clipping_sharpe_lower(self):
        """Very negative Sharpe should be clipped to 0.0 normalized."""
        result_low = {"final_score": 0.0, "sharpe": -10.0, "pnl_pct": 0.0, "win_rate": 0.0}
        result_floor = {"final_score": 0.0, "sharpe": -0.5, "pnl_pct": 0.0, "win_rate": 0.0}
        assert _composite_rank(result_low) == _composite_rank(result_floor)

    def test_clipping_pnl_upper(self):
        """Very high PnL should be clipped."""
        result_high = {"final_score": 0.0, "sharpe": 0.0, "pnl_pct": 1.0, "win_rate": 0.0}
        result_cap = {"final_score": 0.0, "sharpe": 0.0, "pnl_pct": 0.10, "win_rate": 0.0}
        assert _composite_rank(result_high) == _composite_rank(result_cap)

    def test_clipping_pnl_lower(self):
        """Very negative PnL should be clipped."""
        result_low = {"final_score": 0.0, "sharpe": 0.0, "pnl_pct": -1.0, "win_rate": 0.0}
        result_floor = {"final_score": 0.0, "sharpe": 0.0, "pnl_pct": -0.05, "win_rate": 0.0}
        assert _composite_rank(result_low) == _composite_rank(result_floor)

    def test_missing_final_score_uses_default(self):
        """If final_score missing, should use default 0.5."""
        result = {"sharpe": 0.0, "pnl_pct": 0.0, "win_rate": 0.0}
        score = _composite_rank(result)
        # 0.40*0.5 + 0.25*(0.5/1.5) + 0.20*(0.05/0.15) + 0.15*0.0
        expected = 0.40 * 0.5 + 0.25 * (0.5 / 1.5) + 0.20 * (0.05 / 0.15) + 0.15 * 0.0
        assert score == pytest.approx(expected, abs=1e-6)

    def test_ordering_better_pair_scores_higher(self):
        """A clearly better pair should rank higher than a worse one."""
        good = {
            "final_score": 0.8,
            "sharpe": 0.5,
            "pnl_pct": 0.05,
            "win_rate": 0.7,
        }
        bad = {
            "final_score": 0.2,
            "sharpe": -0.2,
            "pnl_pct": -0.02,
            "win_rate": 0.3,
        }
        assert _composite_rank(good) > _composite_rank(bad)

    def test_analytics_score_dominant(self):
        """Analytics score (40%) should have significant impact."""
        high_analytics = {
            "final_score": 0.9,
            "sharpe": 0.0,
            "pnl_pct": 0.0,
            "win_rate": 0.0,
        }
        high_sharpe = {
            "final_score": 0.1,
            "sharpe": 0.8,
            "pnl_pct": 0.0,
            "win_rate": 0.0,
        }
        # Analytics weight (0.40) > sharpe weight (0.25), so high analytics should win
        # high_analytics: 0.40*0.9 + 0.25*(0.5/1.5) = 0.36 + 0.083 = 0.443 + pnl/wr
        # high_sharpe: 0.40*0.1 + 0.25*(1.3/1.5) = 0.04 + 0.217 = 0.257 + pnl/wr
        assert _composite_rank(high_analytics) > _composite_rank(high_sharpe)

    def test_return_type_is_float(self):
        """Result should always be a float."""
        result = {"final_score": 0.5, "sharpe": 0.1, "pnl_pct": 0.01, "win_rate": 0.6}
        score = _composite_rank(result)
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        """Score should always be between 0 and 1."""
        import random
        random.seed(42)
        for _ in range(100):
            result = {
                "final_score": random.uniform(-1, 2),
                "sharpe": random.uniform(-5, 5),
                "pnl_pct": random.uniform(-1, 1),
                "win_rate": random.uniform(-0.5, 1.5),
            }
            score = _composite_rank(result)
            assert 0.0 <= score <= 1.0 + 1e-9, f"Score {score} out of range for {result}"
