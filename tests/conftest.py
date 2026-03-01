from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.Series:
    """Synthetic price series: geometric random walk (500 points)."""
    rng = np.random.RandomState(42)
    log_returns = rng.normal(0.0, 0.01, size=500)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, name="close")


@pytest.fixture
def mean_reverting_spread() -> pd.Series:
    """Synthetic Ornstein-Uhlenbeck process (mean-reverting spread).

    Parameters chosen so that half-life is finite and Hurst < 0.5.
    """
    rng = np.random.RandomState(123)
    n = 600
    theta = 0.15        # speed of mean reversion
    mu = 1.0            # long-run mean
    sigma = 0.02        # volatility
    dt = 1.0

    spread = np.empty(n)
    spread[0] = mu
    for i in range(1, n):
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) * dt + sigma * rng.normal()
    return pd.Series(spread, name="spread")


@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory (pytest built-in tmp_path wrapper)."""
    return tmp_path
