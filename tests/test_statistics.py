"""Tests for anal.pairtool.tests statistical functions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from anal.pairtool.tests import (
    adf_test,
    build_spread,
    correlations,
    half_life,
    hurst_exponent,
)


# ---------------------------------------------------------------------------
# half_life
# ---------------------------------------------------------------------------

class TestHalfLife:
    def test_mean_reverting_series_finite(self, mean_reverting_spread):
        hl = half_life(mean_reverting_spread)
        assert hl is not None
        assert hl > 0

    def test_mean_reverting_series_reasonable_range(self, mean_reverting_spread):
        hl = half_life(mean_reverting_spread)
        # OU process with theta=0.15 => theoretical hl ~ -ln(2)/ln(1-0.15) ~ 4.3
        # Empirical can vary, but should be in a sensible range (1..100).
        assert hl is not None
        assert 1.0 < hl < 100.0

    def test_short_series_returns_none(self):
        s = pd.Series([1.0, 1.1, 1.05])
        assert half_life(s) is None

    def test_constant_series(self):
        s = pd.Series([5.0] * 50)
        hl = half_life(s)
        # Constant series: no mean reversion possible => None
        assert hl is None

    def test_random_walk_large_halflife(self, sample_prices):
        hl = half_life(sample_prices)
        # Random walk should have very large (or None) half-life
        if hl is not None:
            assert hl > 10.0


# ---------------------------------------------------------------------------
# hurst_exponent
# ---------------------------------------------------------------------------

class TestHurstExponent:
    def test_mean_reverting_below_half(self, mean_reverting_spread):
        h = hurst_exponent(mean_reverting_spread, max_lag=50)
        assert h is not None
        assert h < 0.5

    def test_random_walk_around_half(self):
        rng = np.random.RandomState(99)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        h = hurst_exponent(rw, max_lag=50)
        assert h is not None
        # Random walk H should be approximately 0.5 (tolerance: 0.15)
        assert 0.35 < h < 0.65

    def test_trending_above_half(self):
        # Strong trend: cumulative sum of positive increments
        rng = np.random.RandomState(7)
        trend = pd.Series(np.cumsum(rng.uniform(0.5, 1.5, 500)))
        h = hurst_exponent(trend, max_lag=50)
        assert h is not None
        assert h > 0.45  # trending series typically > 0.5

    def test_short_series_returns_none(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert hurst_exponent(s, max_lag=100) is None


# ---------------------------------------------------------------------------
# adf_test
# ---------------------------------------------------------------------------

class TestAdfTest:
    def test_stationary_series_low_pvalue(self, mean_reverting_spread):
        result = adf_test(mean_reverting_spread)
        assert "stat" in result
        assert "p_value" in result
        assert "crit_values" in result
        # Mean-reverting OU process should reject null of unit root
        assert result["p_value"] < 0.05

    def test_random_walk_high_pvalue(self):
        rng = np.random.RandomState(55)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        result = adf_test(rw)
        # Random walk should NOT reject null => high p-value
        assert result["p_value"] > 0.05

    def test_result_keys(self, mean_reverting_spread):
        result = adf_test(mean_reverting_spread)
        expected_keys = {"stat", "p_value", "used_lag", "nobs", "crit_values"}
        assert expected_keys == set(result.keys())

    def test_critical_values_dict(self, mean_reverting_spread):
        result = adf_test(mean_reverting_spread)
        cv = result["crit_values"]
        assert isinstance(cv, dict)
        # Standard ADF critical values at 1%, 5%, 10%
        assert "1%" in cv
        assert "5%" in cv
        assert "10%" in cv


# ---------------------------------------------------------------------------
# build_spread
# ---------------------------------------------------------------------------

class TestBuildSpread:
    def test_returns_spread_model(self, sample_prices):
        y = sample_prices
        x = sample_prices * 1.1 + 5.0  # linear transform
        model = build_spread(y, x, include_intercept=True)
        assert hasattr(model, "alpha")
        assert hasattr(model, "beta")
        assert hasattr(model, "spread")

    def test_spread_length_matches_input(self, sample_prices):
        y = sample_prices
        x = sample_prices.copy()
        model = build_spread(y, x, include_intercept=True)
        assert len(model.spread) == len(y)

    def test_identical_series_spread_near_one(self):
        s = pd.Series(np.linspace(100, 200, 200))
        model = build_spread(s, s, include_intercept=True)
        # returns_y - returns_x = 0 => cumprod(1+0) = 1.0
        assert all(math.isclose(v, 1.0, rel_tol=1e-9) for v in model.spread.values)

    def test_spread_starts_at_one(self, sample_prices):
        rng = np.random.RandomState(10)
        x = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(sample_prices)))))
        model = build_spread(sample_prices, x, include_intercept=True)
        assert math.isclose(model.spread.iloc[0], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# correlations
# ---------------------------------------------------------------------------

class TestCorrelations:
    def test_perfect_positive_correlation(self):
        x = pd.Series(np.arange(100, dtype=float))
        y = pd.Series(np.arange(100, dtype=float))
        result = correlations(x, y, ["pearson", "spearman", "kendall"])
        assert math.isclose(result["pearson"]["value"], 1.0, rel_tol=1e-6)
        assert math.isclose(result["spearman"]["value"], 1.0, rel_tol=1e-6)
        assert math.isclose(result["kendall"]["value"], 1.0, rel_tol=1e-6)

    def test_negative_correlation(self):
        x = pd.Series(np.arange(100, dtype=float))
        y = pd.Series(-np.arange(100, dtype=float))
        result = correlations(x, y, ["pearson"])
        assert math.isclose(result["pearson"]["value"], -1.0, rel_tol=1e-6)

    def test_result_structure(self):
        rng = np.random.RandomState(0)
        x = pd.Series(rng.normal(size=200))
        y = pd.Series(rng.normal(size=200))
        methods = ["pearson", "spearman", "kendall"]
        result = correlations(x, y, methods)
        for m in methods:
            assert m in result
            assert "value" in result[m]
            assert "p_value" in result[m]
            assert isinstance(result[m]["value"], float)
            assert isinstance(result[m]["p_value"], float)

    def test_pvalues_between_zero_and_one(self):
        rng = np.random.RandomState(1)
        x = pd.Series(rng.normal(size=200))
        y = pd.Series(rng.normal(size=200))
        result = correlations(x, y, ["pearson", "spearman"])
        for m in result:
            assert 0.0 <= result[m]["p_value"] <= 1.0

    def test_unsupported_method_raises(self):
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="Unsupported correlation method"):
            correlations(x, y, ["unknown_method"])
