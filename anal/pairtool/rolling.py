from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa import stattools


@dataclass
class RollingResult:
    df: pd.DataFrame
    summary: Dict[str, float]


def _rolling_beta(y: pd.Series, x: pd.Series, include_intercept: bool) -> float:
    X = sm.add_constant(x) if include_intercept else x.to_frame("x")
    model = sm.OLS(y, X).fit()
    if include_intercept:
        return float(model.params.drop("const").iloc[0])
    return float(model.params.iloc[0])


def _rolling_adf_p(spread: pd.Series) -> float:
    res = stattools.adfuller(spread.dropna(), autolag="AIC")
    return float(res[1])


def compute_rolling(
    log_y: pd.Series,
    log_x: pd.Series,
    spread: pd.Series,
    returns_y: pd.Series,
    returns_x: pd.Series,
    window: int,
    step: int,
    min_periods: int,
    include_intercept: bool,
) -> RollingResult:
    idx = log_y.index
    rows: List[Dict[str, float]] = []
    for end in range(window, len(idx) + 1, step):
        start = end - window
        if end - start < min_periods:
            continue
        window_slice = slice(start, end)
        y_w = log_y.iloc[window_slice]
        x_w = log_x.iloc[window_slice]
        s_w = spread.iloc[window_slice]
        ry_w = returns_y.iloc[window_slice]
        rx_w = returns_x.iloc[window_slice]

        try:
            beta = _rolling_beta(y_w, x_w, include_intercept)
        except Exception:
            beta = np.nan

        corr = np.corrcoef(ry_w.dropna(), rx_w.dropna())[0, 1]
        try:
            adf_p = _rolling_adf_p(s_w)
        except Exception:
            adf_p = np.nan

        rows.append({
            "window_end": idx[end - 1],
            "beta": beta,
            "corr": float(corr),
            "adf_p": float(adf_p),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("window_end") if not df.empty else df
    summary = {
        "windows": float(len(df)),
        "adf_p_ok_pct": float((df["adf_p"] < 0.05).mean() * 100.0) if not df.empty else 0.0,
    }
    return RollingResult(df=df, summary=summary)
