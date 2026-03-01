from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller, coint, kpss

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
except Exception:  # pragma: no cover
    coint_johansen = None


WEIGHTS: Dict[str, int] = {
    "engle_granger_adj": 10,
    "adf_adj": 10,
    "kpss": 5,
    "no_structural_breaks": 10,
    "beta_adf": 8,
    "beta_rel_std": 7,
    "half_life_mean": 5,
    "half_life_p95": 8,
    "stationary_weeks": 20,
    "reliability_score": 0,
    "max_dev_sigma": 5,
    "arch_or_garch": 4,
    "unique_ratio": 4,
    "stress_test": 4,
}


STRESS_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("covid_crash_2020", "2020-03-01", "2020-03-31"),
    ("may_2021_selloff", "2021-05-01", "2021-05-31"),
    ("ftx_2022", "2022-11-01", "2022-11-30"),
)


@dataclass
class AnalyzerConfig:
    coins_file: str = "coins.txt"
    data_dir: str = "data/binance"
    output_dir: str = "anal/output"
    timeframe: str = "1h"
    history_days: int = 365
    auto_fetch_data: bool = True
    horizon_days: int = 7
    rolling_window_days: int = 30
    rolling_step_days: int = 7
    stop_sigma: float = 3.5
    monte_carlo_enabled: bool = True
    monte_carlo_paths: int = 10_000
    bootstrap_block_bars: int = 24
    max_pairs: int = 0
    no_pdf: bool = False
    seed: int = 42

    engle_granger_alpha_adj: float = 0.001
    adf_alpha_adj: float = 0.001
    kpss_alpha: float = 0.05
    beta_adf_alpha: float = 0.05
    beta_rel_std_max: float = 0.10
    half_life_mean_max_days: float = 3.0
    half_life_p95_max_days: float = 4.0
    stationary_weeks_min_pct: float = 90.0
    reliability_min_pct: float = 95.0
    reliability_sss_pct: float = 99.0
    max_dev_sigma: float = 4.0
    unique_ratio_min_pct: float = 70.0
    stress_revert_days: int = 3

    score_sss_min: float = 98.0
    score_s_min: float = 90.0
    score_conditional_min: float = 80.0
    critical_reliability_fail_pct: float = 80.0


def _optional_tqdm(iterable: Iterable, total: int):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, ncols=100)
    except Exception:
        return iterable


def _parse_timeframe(tf: str) -> pd.Timedelta:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe '{tf}'. Use forms like 1m/1h/4h/1d.")


def _parse_ts_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" in cols:
        ts = pd.to_datetime(df[cols["timestamp"]], utc=True, errors="coerce")
    elif "ts_iso" in cols:
        ts = pd.to_datetime(df[cols["ts_iso"]], utc=True, errors="coerce")
    elif "ts_ms" in cols:
        ts = pd.to_datetime(df[cols["ts_ms"]], utc=True, errors="coerce", unit="ms")
    else:
        raise ValueError("Missing timestamp column: expected one of timestamp/ts_iso/ts_ms")
    return pd.DatetimeIndex(ts)


def _read_coins(path: Path) -> List[str]:
    coins: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if not val or val.startswith("#"):
            continue
        coins.append(val.upper())
    return coins


def _load_close_series(symbol: str, timeframe: str, data_dir: Path) -> pd.Series:
    path = data_dir / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV missing: {path}")

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "close" not in cols:
        raise ValueError(f"Column 'close' not found in {path}")

    ts = _parse_ts_index(df)
    out = pd.Series(df[cols["close"]].astype(float).values, index=ts, name=symbol)
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out.dropna()


def _align_prices(
    series_a: pd.Series,
    series_b: pd.Series,
    history_days: int,
) -> pd.DataFrame:
    df = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    df.columns = ["price_a", "price_b"]
    if df.empty:
        return df
    if history_days > 0:
        end_ts = df.index.max()
        start_ts = end_ts - pd.Timedelta(days=history_days)
        df = df.loc[df.index >= start_ts]
    df["log_a"] = np.log(df["price_a"])
    df["log_b"] = np.log(df["price_b"])
    return df.dropna()


def _ols_hedge(log_a: pd.Series, log_b: pd.Series) -> Tuple[float, float]:
    x = np.column_stack([np.ones(len(log_b)), log_b.values])
    y = log_a.values
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    return alpha, beta


def _spread(log_a: pd.Series, log_b: pd.Series, alpha: float, beta: float) -> pd.Series:
    spread = log_a - (alpha + beta * log_b)
    spread.name = "spread"
    return spread


def _safe_adf(series: pd.Series) -> float:
    try:
        return float(adfuller(series.dropna(), autolag="AIC")[1])
    except Exception:
        return float("nan")


def _safe_kpss(series: pd.Series) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(series.dropna(), regression="c", nlags="auto")[1])
    except Exception:
        return float("nan")


def _safe_engle_granger(log_a: pd.Series, log_b: pd.Series) -> float:
    try:
        return float(coint(log_a, log_b, trend="c")[1])
    except Exception:
        return float("nan")


def _safe_johansen_pass(log_a: pd.Series, log_b: pd.Series) -> float:
    if coint_johansen is None:
        return float("nan")
    data = pd.concat([log_a, log_b], axis=1).dropna()
    if len(data) < 40:
        return float("nan")
    try:
        joh = coint_johansen(data, det_order=0, k_ar_diff=1)
    except Exception:
        return float("nan")
    if len(joh.lr1) == 0:
        return float("nan")
    return float(joh.lr1[0] > joh.cvt[0, 1])


def _ols_sse(y: np.ndarray, x: np.ndarray) -> Tuple[float, int]:
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(np.sum(resid**2)), X.shape[1]


def _structural_breaks_proxy(
    log_a: pd.Series,
    log_b: pd.Series,
    grid_points: int = 20,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    data = pd.concat([log_a, log_b], axis=1).dropna()
    if len(data) < 120:
        return {"break_count": 0, "min_p": float("nan"), "points": [], "checked_points": 0}

    y = data.iloc[:, 0].values
    x = data.iloc[:, 1].values
    n = len(data)
    idx = sorted(set(np.linspace(int(n * 0.2), int(n * 0.8), grid_points).astype(int).tolist()))

    sse_pool, k = _ols_sse(y, x)
    pvals: List[Tuple[int, float]] = []
    for bp in idx:
        y1, x1 = y[:bp], x[:bp]
        y2, x2 = y[bp:], x[bp:]
        if len(y1) < 40 or len(y2) < 40:
            continue
        sse1, _ = _ols_sse(y1, x1)
        sse2, _ = _ols_sse(y2, x2)
        den_df = len(y1) + len(y2) - 2 * k
        if den_df <= 0:
            continue
        num = (sse_pool - (sse1 + sse2)) / k
        den = (sse1 + sse2) / den_df
        if den <= 0:
            continue
        f_stat = num / den
        p_val = float(stats.f.sf(f_stat, k, den_df))
        pvals.append((bp, p_val))

    if not pvals:
        return {"break_count": 0, "min_p": float("nan"), "points": [], "checked_points": 0}

    alpha_adj = alpha / len(pvals)
    breaks = [int(bp) for bp, p in pvals if p < alpha_adj]
    points = [data.index[bp].strftime("%Y-%m-%dT%H:%M:%SZ") for bp in breaks]
    min_p = float(min(p for _, p in pvals))
    return {
        "break_count": int(len(breaks)),
        "min_p": min_p,
        "points": points,
        "checked_points": int(len(pvals)),
    }


def _estimate_gamma(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 20:
        return float("nan")
    delta = s.diff().dropna()
    lag = s.shift(1).loc[delta.index]
    try:
        model = sm.OLS(delta.values, sm.add_constant(lag.values)).fit()
    except Exception:
        return float("nan")
    if len(model.params) < 2:
        return float("nan")
    return float(model.params[1])


def _half_life_days(spread: pd.Series, bar_delta: pd.Timedelta) -> float:
    gamma = _estimate_gamma(spread)
    if not np.isfinite(gamma) or gamma >= 0:
        return float("nan")
    hl_bars = math.log(2) / abs(gamma)
    hl_days = hl_bars * (bar_delta / pd.Timedelta(days=1))
    return float(hl_days) if np.isfinite(hl_days) else float("nan")


def _rolling_beta_half_life(
    log_a: pd.Series,
    log_b: pd.Series,
    bar_delta: pd.Timedelta,
    window_days: int,
    step_days: int,
) -> pd.DataFrame:
    n = len(log_a)
    if n < 100:
        return pd.DataFrame(columns=["beta", "half_life_days"])
    bars_per_day = max(1, int(round(pd.Timedelta(days=1) / bar_delta)))
    window = max(40, window_days * bars_per_day)
    step = max(1, step_days * bars_per_day)

    rows: List[Dict[str, Any]] = []
    for end in range(window, n + 1, step):
        start = end - window
        a_w = log_a.iloc[start:end]
        b_w = log_b.iloc[start:end]
        alpha, beta = _ols_hedge(a_w, b_w)
        spread_w = _spread(a_w, b_w, alpha, beta)
        rows.append(
            {
                "window_end": log_a.index[end - 1],
                "beta": float(beta),
                "half_life_days": _half_life_days(spread_w, bar_delta),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["beta", "half_life_days"])
    out = pd.DataFrame(rows).set_index("window_end")
    return out


def _weekly_stationarity_pct(spread: pd.Series, bar_delta: pd.Timedelta, adf_alpha: float = 0.05) -> float:
    bars_week = max(20, int(round(pd.Timedelta(days=7) / bar_delta)))
    s = spread.dropna()
    if len(s) < bars_week:
        return float("nan")
    n_weeks = len(s) // bars_week
    if n_weeks == 0:
        return float("nan")
    pvals: List[float] = []
    for i in range(n_weeks):
        seg = s.iloc[i * bars_week : (i + 1) * bars_week]
        p = _safe_adf(seg)
        if np.isfinite(p):
            pvals.append(p)
    if not pvals:
        return float("nan")
    return float(np.mean(np.array(pvals) < adf_alpha) * 100.0)


def _stress_test(spread: pd.Series, cfg: AnalyzerConfig, bar_delta: pd.Timedelta) -> Dict[str, Any]:
    s = spread.dropna()
    if s.empty:
        return {"covered_periods": 0, "passed": False, "details": []}
    std = s.std(ddof=0)
    if std <= 0 or np.isnan(std):
        return {"covered_periods": 0, "passed": False, "details": []}
    z = (s - s.mean()) / std
    max_lookahead = max(1, int(round(pd.Timedelta(days=cfg.stress_revert_days) / bar_delta)))

    details: List[Dict[str, Any]] = []
    covered = 0
    passed_all = True
    for name, start, end in STRESS_WINDOWS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        mask = (z.index >= start_ts) & (z.index <= end_ts)
        if not mask.any():
            continue
        covered += 1
        z_period = z.loc[mask]
        period_pass = True
        max_abs = float(z_period.abs().max()) if len(z_period) else float("nan")
        if max_abs > 3.0:
            positions = np.where(mask)[0]
            breaches = [pos for pos in positions if abs(z.iloc[pos]) > 3.0]
            for pos in breaches:
                tail = z.iloc[pos : min(len(z), pos + max_lookahead + 1)]
                if not (tail.abs() <= 3.0).any():
                    period_pass = False
                    break
        if not period_pass:
            passed_all = False
        details.append({"name": name, "max_abs_z": max_abs, "pass": period_pass})

    if covered == 0:
        return {"covered_periods": 0, "passed": True, "details": []}
    return {"covered_periods": covered, "passed": passed_all, "details": details}


def _bootstrap_residuals(
    residuals: np.ndarray,
    n_paths: int,
    horizon: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(residuals)
    if n == 0:
        return np.zeros((n_paths, horizon))
    block = max(1, min(block, n))
    if block == 1:
        idx = rng.integers(0, n, size=(n_paths, horizon))
        return residuals[idx]
    n_blocks = int(math.ceil(horizon / block))
    starts = rng.integers(0, n - block + 1, size=(n_paths, n_blocks))
    out = np.empty((n_paths, n_blocks * block), dtype=float)
    for b in range(n_blocks):
        base = starts[:, b]
        for off in range(block):
            out[:, b * block + off] = residuals[base + off]
    return out[:, :horizon]


def _fit_ar1(spread: pd.Series) -> Tuple[float, float, np.ndarray]:
    s = spread.dropna().values
    if len(s) < 30:
        return float("nan"), float("nan"), np.array([])
    y = s[1:]
    x = s[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    c = float(coef[0])
    phi = float(coef[1])
    resid = y - (c + phi * x)
    return c, phi, resid


def _monte_carlo_reliability(
    spread: pd.Series,
    cfg: AnalyzerConfig,
    bar_delta: pd.Timedelta,
) -> Dict[str, Any]:
    c, phi, residuals = _fit_ar1(spread)
    if not np.isfinite(c) or not np.isfinite(phi) or len(residuals) < 30:
        return {"reliability_pct": float("nan"), "phi": float("nan"), "paths_preview": np.empty((0, 0))}

    horizon = max(1, int(round(pd.Timedelta(days=cfg.horizon_days) / bar_delta)))
    s = spread.dropna()
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    if std <= 0 or np.isnan(std):
        return {"reliability_pct": float("nan"), "phi": phi, "paths_preview": np.empty((0, 0))}
    upper = mean + cfg.stop_sigma * std
    lower = mean - cfg.stop_sigma * std
    s0 = float(s.iloc[-1])

    rng = np.random.default_rng(cfg.seed)
    eps = _bootstrap_residuals(
        residuals=residuals,
        n_paths=cfg.monte_carlo_paths,
        horizon=horizon,
        block=cfg.bootstrap_block_bars,
        rng=rng,
    )

    paths = np.empty((cfg.monte_carlo_paths, horizon), dtype=float)
    prev = np.full(cfg.monte_carlo_paths, s0, dtype=float)
    for t in range(horizon):
        cur = c + phi * prev + eps[:, t]
        paths[:, t] = cur
        prev = cur

    breached = ((paths >= upper) | (paths <= lower)).any(axis=1)
    start_dist = abs(s0 - mean)
    end_dist = np.abs(paths[:, -1] - mean)
    crosses_mean = ((paths - mean) * (s0 - mean) <= 0).any(axis=1)
    mean_rev = (end_dist < start_dist) | crosses_mean
    ar_stable = abs(phi) < 1.0
    success = (~breached) & mean_rev & ar_stable
    reliability = float(success.mean() * 100.0)
    preview = paths[: min(100, len(paths))]
    return {"reliability_pct": reliability, "phi": phi, "paths_preview": preview}


def _bool_pass(value: float, comparator: str, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    if comparator == "<":
        return bool(value < threshold)
    if comparator == "<=":
        return bool(value <= threshold)
    if comparator == ">":
        return bool(value > threshold)
    if comparator == ">=":
        return bool(value >= threshold)
    raise ValueError(f"Unsupported comparator '{comparator}'")


def _score_and_classify(
    criteria: Dict[str, bool],
    criteria_evaluated: Dict[str, bool],
    reliability: float,
    cfg: AnalyzerConfig,
) -> Tuple[float, str, float]:
    score_raw = 0.0
    active_weight_sum = 0.0
    for key, weight in WEIGHTS.items():
        if criteria_evaluated.get(key, True):
            active_weight_sum += weight
            if criteria.get(key, False):
                score_raw += weight

    score = float(score_raw / active_weight_sum * 100.0) if active_weight_sum > 0 else 0.0

    critical_fail = (
        (not criteria.get("engle_granger_adj", False))
        or (not criteria.get("no_structural_breaks", False))
        or (
            cfg.monte_carlo_enabled
            and ((not np.isfinite(reliability)) or (reliability < cfg.critical_reliability_fail_pct))
        )
    )
    if critical_fail:
        return score, "Не пригодна", active_weight_sum

    if cfg.monte_carlo_enabled and score >= cfg.score_sss_min and reliability > cfg.reliability_sss_pct:
        return score, "SSS", active_weight_sum
    if cfg.score_s_min <= score < cfg.score_sss_min:
        return score, "S", active_weight_sum
    if cfg.score_conditional_min <= score < cfg.score_s_min:
        return score, "Условно пригодна", active_weight_sum
    return score, "Не пригодна", active_weight_sum


def _analyze_pair(
    symbol_a: str,
    symbol_b: str,
    series_a: pd.Series,
    series_b: pd.Series,
    cfg: AnalyzerConfig,
) -> Dict[str, Any]:
    df = _align_prices(series_a, series_b, cfg.history_days)
    if len(df) < 300:
        raise ValueError(f"Недостаточно данных после выравнивания: {len(df)}")

    inferred_delta = df.index.to_series().diff().median()
    if pd.isna(inferred_delta) or inferred_delta <= pd.Timedelta(0):
        bar_delta = _parse_timeframe(cfg.timeframe)
    else:
        bar_delta = inferred_delta

    alpha, beta = _ols_hedge(df["log_a"], df["log_b"])
    spread = _spread(df["log_a"], df["log_b"], alpha, beta)
    df["spread"] = spread

    eg_p = _safe_engle_granger(df["log_a"], df["log_b"])
    adf_p = _safe_adf(spread)
    kpss_p = _safe_kpss(spread)
    joh_pass = _safe_johansen_pass(df["log_a"], df["log_b"])

    correction_factor = 2.0
    eg_p_adj = float(min(1.0, eg_p * correction_factor)) if np.isfinite(eg_p) else float("nan")
    adf_p_adj = float(min(1.0, adf_p * correction_factor)) if np.isfinite(adf_p) else float("nan")

    breaks = _structural_breaks_proxy(df["log_a"], df["log_b"])
    rolling = _rolling_beta_half_life(
        df["log_a"],
        df["log_b"],
        bar_delta=bar_delta,
        window_days=cfg.rolling_window_days,
        step_days=cfg.rolling_step_days,
    )
    beta_series = rolling["beta"].dropna() if "beta" in rolling else pd.Series(dtype=float)
    beta_adf_p = _safe_adf(beta_series) if len(beta_series) >= 20 else float("nan")
    beta_mean = float(beta_series.mean()) if len(beta_series) else float("nan")
    beta_std = float(beta_series.std(ddof=0)) if len(beta_series) else float("nan")
    beta_rel_std = (
        float(abs(beta_std / beta_mean))
        if np.isfinite(beta_std) and np.isfinite(beta_mean) and abs(beta_mean) > 1e-12
        else float("nan")
    )

    hl_series = (
        rolling["half_life_days"].replace([np.inf, -np.inf], np.nan).dropna()
        if "half_life_days" in rolling
        else pd.Series(dtype=float)
    )
    hl_mean = float(hl_series.mean()) if len(hl_series) else float("nan")
    hl_p95 = float(np.nanpercentile(hl_series, 95)) if len(hl_series) else float("nan")

    try:
        arch_p = float(het_arch(spread.dropna(), nlags=min(24, max(5, len(spread) // 10)))[1])
    except Exception:
        arch_p = float("nan")

    try:
        jb_p = float(stats.jarque_bera(spread.dropna()).pvalue)
    except Exception:
        jb_p = float("nan")

    spread_std = float(spread.std(ddof=0))
    if spread_std > 0 and np.isfinite(spread_std):
        z = (spread - spread.mean()) / spread_std
        max_dev = float(z.abs().max())
        outlier_pct = float((z.abs() > 6.0).mean() * 100.0)
    else:
        z = pd.Series(index=spread.index, dtype=float)
        max_dev = float("nan")
        outlier_pct = float("nan")

    unique_ratio = float(spread.round(12).nunique() / len(spread) * 100.0)
    stationary_weeks = _weekly_stationarity_pct(spread, bar_delta, adf_alpha=0.05)
    stress = _stress_test(spread, cfg, bar_delta)
    if cfg.monte_carlo_enabled:
        mc = _monte_carlo_reliability(spread, cfg, bar_delta)
    else:
        mc = {"reliability_pct": float("nan"), "phi": float("nan"), "paths_preview": np.empty((0, 0))}
    reliability = float(mc["reliability_pct"]) if np.isfinite(mc["reliability_pct"]) else float("nan")

    criteria = {
        "engle_granger_adj": _bool_pass(eg_p_adj, "<", cfg.engle_granger_alpha_adj),
        "adf_adj": _bool_pass(adf_p_adj, "<", cfg.adf_alpha_adj),
        "kpss": _bool_pass(kpss_p, ">", cfg.kpss_alpha),
        "no_structural_breaks": breaks["break_count"] == 0,
        "beta_adf": _bool_pass(beta_adf_p, "<", cfg.beta_adf_alpha),
        "beta_rel_std": _bool_pass(beta_rel_std, "<", cfg.beta_rel_std_max),
        "half_life_mean": _bool_pass(hl_mean, "<", cfg.half_life_mean_max_days),
        "half_life_p95": _bool_pass(hl_p95, "<", cfg.half_life_p95_max_days),
        "stationary_weeks": _bool_pass(stationary_weeks, ">=", cfg.stationary_weeks_min_pct),
        "reliability_score": _bool_pass(reliability, ">", cfg.reliability_min_pct),
        "max_dev_sigma": _bool_pass(max_dev, "<", cfg.max_dev_sigma),
        "arch_or_garch": _bool_pass(arch_p, ">", 0.05),
        "unique_ratio": _bool_pass(unique_ratio, ">", cfg.unique_ratio_min_pct),
        "stress_test": bool(stress["passed"]),
    }
    criteria_evaluated = {key: True for key in WEIGHTS}
    if not cfg.monte_carlo_enabled:
        criteria_evaluated["reliability_score"] = False

    score_100, klass, score_max_weight = _score_and_classify(criteria, criteria_evaluated, reliability, cfg)
    gate = int(klass in {"SSS", "S", "Условно пригодна"})

    row = {
        "coin1": symbol_a,
        "coin2": symbol_b,
        "nobs": int(len(df)),
        "score_100": float(score_100),
        "final_score": float(score_100 / 100.0),
        "gate": gate,
        "class": klass,
        "monte_carlo_enabled": int(cfg.monte_carlo_enabled),
        "reliability_score_pct": reliability,
        "eg_p": eg_p,
        "eg_p_adj": eg_p_adj,
        "adf_p": adf_p,
        "adf_p_adj": adf_p_adj,
        "kpss_p": kpss_p,
        "johansen_pass": joh_pass,
        "break_count": int(breaks["break_count"]),
        "beta": beta,
        "beta_adf_p": beta_adf_p,
        "beta_rel_std": beta_rel_std,
        "half_life_mean_days": hl_mean,
        "half_life_p95_days": hl_p95,
        "stationary_weeks_pct": stationary_weeks,
        "arch_p": arch_p,
        "jarque_bera_p": jb_p,
        "outliers_pct_abs_z_gt_6": outlier_pct,
        "max_dev_sigma": max_dev,
        "unique_ratio_pct": unique_ratio,
        "stress_windows_covered": int(stress["covered_periods"]),
        "mc_phi": float(mc["phi"]) if np.isfinite(mc["phi"]) else float("nan"),
        "score_weight_max_active": float(score_max_weight),
    }
    for key in WEIGHTS:
        row[f"pass_{key}"] = int(criteria[key]) if criteria_evaluated.get(key, True) else -1

    details = {
        "pair": [symbol_a, symbol_b],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": asdict(cfg),
        "weights": WEIGHTS,
        "criteria": criteria,
        "criteria_evaluated": criteria_evaluated,
        "row": row,
        "tests": {
            "engle_granger_p": eg_p,
            "engle_granger_p_adj": eg_p_adj,
            "adf_spread_p": adf_p,
            "adf_spread_p_adj": adf_p_adj,
            "kpss_spread_p": kpss_p,
            "johansen_pass": joh_pass,
            "structural_breaks_proxy": breaks,
            "beta_adf_p": beta_adf_p,
            "beta_rel_std": beta_rel_std,
            "half_life_mean_days": hl_mean,
            "half_life_p95_days": hl_p95,
            "stationary_weeks_pct": stationary_weeks,
            "reliability_score_pct": reliability,
            "arch_p": arch_p,
            "jarque_bera_p": jb_p,
            "outliers_pct_abs_z_gt_6": outlier_pct,
            "max_dev_sigma": max_dev,
            "unique_ratio_pct": unique_ratio,
            "stress_test": stress,
        },
        "spread_model": {"alpha": alpha, "beta": beta},
    }
    return {
        "row": row,
        "details": details,
        "df": df,
        "rolling": rolling,
        "mc_paths_preview": mc["paths_preview"],
    }


def _save_pair_pdf(
    pdf_path: Path,
    pair_name: str,
    df: pd.DataFrame,
    rolling: pd.DataFrame,
    mc_paths_preview: np.ndarray,
    row: Dict[str, Any],
) -> None:
    spread = df["spread"].dropna()
    spread_mean = float(spread.mean())
    spread_std = float(spread.std(ddof=0))
    z = (spread - spread_mean) / spread_std if spread_std > 0 else pd.Series(index=spread.index, dtype=float)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        lines = [
            f"Pair report: {pair_name}",
            "",
            f"Class: {row.get('class')}",
            f"Score 0..100: {row.get('score_100', float('nan')):.2f}",
            f"Reliability Score: {row.get('reliability_score_pct', float('nan')):.2f}%",
            f"Observations: {int(row.get('nobs', 0))}",
            "",
            f"EG p(adj): {row.get('eg_p_adj', float('nan')):.4g}",
            f"ADF p(adj): {row.get('adf_p_adj', float('nan')):.4g}",
            f"KPSS p: {row.get('kpss_p', float('nan')):.4g}",
            f"Break count: {int(row.get('break_count', 0))}",
            f"Beta ADF p: {row.get('beta_adf_p', float('nan')):.4g}",
            f"Beta rel std: {row.get('beta_rel_std', float('nan')):.4g}",
            f"Half-life mean days: {row.get('half_life_mean_days', float('nan')):.3f}",
            f"Half-life P95 days: {row.get('half_life_p95_days', float('nan')):.3f}",
            f"Stationary weeks %: {row.get('stationary_weeks_pct', float('nan')):.2f}",
            f"ARCH p: {row.get('arch_p', float('nan')):.4g}",
            f"Max |z|: {row.get('max_dev_sigma', float('nan')):.3f}",
            f"Unique spread %: {row.get('unique_ratio_pct', float('nan')):.2f}",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(11.7, 6))
        norm_a = df["price_a"] / df["price_a"].iloc[0]
        norm_b = df["price_b"] / df["price_b"].iloc[0]
        ax1.plot(df.index, norm_a, label="Price A (norm)")
        ax1.plot(df.index, norm_b, label="Price B (norm)")
        ax1.set_title(f"{pair_name} Prices (normalized)")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(df.index, z, color="black", alpha=0.4, label="Spread z-score")
        ax2.axhline(0.0, color="gray", linewidth=1)
        ax2.axhline(3.0, color="red", linestyle="--", linewidth=0.8)
        ax2.axhline(-3.0, color="red", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("z-score")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 5))
        ax.hist(spread.values, bins=50, density=True, alpha=0.7, label="spread hist")
        if spread_std > 0 and np.isfinite(spread_std):
            x = np.linspace(spread_mean - 4 * spread_std, spread_mean + 4 * spread_std, 300)
            y = stats.norm.pdf(x, loc=spread_mean, scale=spread_std)
            ax.plot(x, y, color="red", label="normal pdf")
        ax.set_title("Spread Distribution")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not rolling.empty:
            fig, axes = plt.subplots(2, 1, figsize=(11.7, 7), sharex=True)
            axes[0].plot(rolling.index, rolling["beta"], label="rolling beta")
            axes[0].axhline(float(np.nanmean(rolling["beta"])), color="black", linestyle="--", linewidth=1)
            axes[0].set_title("Rolling Beta")
            axes[0].legend()
            axes[1].plot(rolling.index, rolling["half_life_days"], label="rolling half-life (days)")
            if rolling["half_life_days"].notna().any():
                hl_mean = float(np.nanmean(rolling["half_life_days"]))
                hl_p95 = float(np.nanpercentile(rolling["half_life_days"].dropna(), 95))
                axes[1].axhline(hl_mean, color="black", linestyle="--", linewidth=1, label=f"mean={hl_mean:.2f}")
                axes[1].axhline(hl_p95, color="red", linestyle="--", linewidth=1, label=f"p95={hl_p95:.2f}")
            axes[1].set_title("Rolling Half-Life")
            axes[1].legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        residuals = spread - spread_mean
        fig = plt.figure(figsize=(11.7, 6))
        ax = fig.add_subplot(111)
        sm.qqplot(residuals.dropna(), line="s", ax=ax)
        ax.set_title("QQ-Plot of Spread Residuals")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        lags = int(min(40, max(10, len(spread) // 20)))
        fig, axes = plt.subplots(2, 1, figsize=(11.7, 7))
        try:
            plot_acf(spread.dropna(), lags=lags, ax=axes[0])
            axes[0].set_title("ACF Spread")
        except Exception:
            axes[0].set_title("ACF Spread (failed)")
        try:
            plot_pacf(spread.dropna(), lags=lags, method="ywm", ax=axes[1])
            axes[1].set_title("PACF Spread")
        except Exception:
            axes[1].set_title("PACF Spread (failed)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if mc_paths_preview.size > 0:
            fig, axes = plt.subplots(2, 1, figsize=(11.7, 7), sharex=False)
            for row_idx in range(min(100, mc_paths_preview.shape[0])):
                axes[0].plot(mc_paths_preview[row_idx], alpha=0.25, linewidth=0.7)
            axes[0].set_title("Monte Carlo Paths (first 100)")
            axes[0].set_xlabel("bar")
            axes[0].set_ylabel("spread")
            axes[1].hist(mc_paths_preview[:, -1], bins=40, alpha=0.8)
            axes[1].set_title("Distribution of Final Spread (preview paths)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _default_config() -> AnalyzerConfig:
    return AnalyzerConfig()


def _load_config_from_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config file must be a YAML mapping")
    return raw


def _build_config(args: argparse.Namespace) -> AnalyzerConfig:
    cfg = _default_config()
    if args.config:
        raw = _load_config_from_yaml(Path(args.config))
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch spread analyzer: all pairs from coins.txt with PDF reports + Excel score table.",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--coins-file", dest="coins_file", default=None)
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--timeframe", default=None, help="CSV timeframe suffix, e.g. 1m/1h/4h")
    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--no-auto-fetch-data", dest="auto_fetch_data", action="store_false", default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--rolling-window-days", type=int, default=None)
    parser.add_argument("--rolling-step-days", type=int, default=None)
    parser.add_argument("--stop-sigma", type=float, default=None)
    parser.add_argument("--no-monte-carlo", dest="monte_carlo_enabled", action="store_false", default=None)
    parser.add_argument("--monte-carlo-paths", type=int, default=None)
    parser.add_argument("--bootstrap-block-bars", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-pdf", action="store_true", default=None)
    return parser


def _refresh_data_if_enabled(cfg: AnalyzerConfig) -> None:
    if not cfg.auto_fetch_data:
        return

    # Reuse the project's downloader so CSV format stays consistent.
    from run import run_download

    days = max(1, int(cfg.history_days))
    run_download(
        days=days,
        timeframes=[cfg.timeframe],
        coins_file=cfg.coins_file,
    )


def run_batch(cfg: AnalyzerConfig) -> Dict[str, Any]:
    _refresh_data_if_enabled(cfg)

    run_ts = datetime.now()
    run_id = run_ts.strftime("%Y-%m-%d_%H-%M-%S")
    coins_file = Path(cfg.coins_file)
    data_dir = Path(cfg.data_dir)
    out_root = Path(cfg.output_dir) / run_id
    pdf_dir = out_root / "pdf_reports"
    json_dir = out_root / "json_reports"
    out_root.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    coins = _read_coins(coins_file)
    if len(coins) < 2:
        raise ValueError("Need at least 2 symbols in coins file.")

    symbol_data: Dict[str, pd.Series] = {}
    load_errors: List[Dict[str, str]] = []
    for symbol in coins:
        try:
            symbol_data[symbol] = _load_close_series(symbol, cfg.timeframe, data_dir)
        except Exception as exc:
            load_errors.append({"symbol": symbol, "error": str(exc)})

    valid_symbols = sorted(symbol_data.keys())
    if len(valid_symbols) < 2:
        raise RuntimeError("Not enough symbols with readable data.")

    pairs = list(itertools.combinations(valid_symbols, 2))
    if cfg.max_pairs and cfg.max_pairs > 0:
        pairs = pairs[: cfg.max_pairs]

    summary_rows: List[Dict[str, Any]] = []
    criteria_rows: List[Dict[str, Any]] = []
    pair_errors: List[Dict[str, Any]] = []

    iterator = _optional_tqdm(pairs, total=len(pairs))
    for symbol_a, symbol_b in iterator:
        pair_name = f"{symbol_a}_{symbol_b}"
        pair_file_stem = f"{pair_name}_{run_id}"
        try:
            result = _analyze_pair(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                series_a=symbol_data[symbol_a],
                series_b=symbol_data[symbol_b],
                cfg=cfg,
            )
            row = result["row"]
            details = result["details"]
            summary_rows.append(row)
            criteria_rows.append(
                {
                    "coin1": symbol_a,
                    "coin2": symbol_b,
                    **{
                        f"pass_{k}": (
                            int(details["criteria"][k]) if details["criteria_evaluated"].get(k, True) else -1
                        )
                        for k in details["criteria"].keys()
                    },
                    **{f"eval_{k}": int(v) for k, v in details["criteria_evaluated"].items()},
                }
            )

            (json_dir / f"{pair_file_stem}.json").write_text(
                json.dumps(details, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if not cfg.no_pdf:
                _save_pair_pdf(
                    pdf_path=pdf_dir / f"{pair_file_stem}.pdf",
                    pair_name=f"{symbol_a}/{symbol_b}",
                    df=result["df"],
                    rolling=result["rolling"],
                    mc_paths_preview=result["mc_paths_preview"],
                    row=row,
                )
        except Exception as exc:
            pair_errors.append({"coin1": symbol_a, "coin2": symbol_b, "error": str(exc)})

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["score_100", "reliability_score_pct"], ascending=[False, False]
        ).reset_index(drop=True)
    criteria_df = pd.DataFrame(criteria_rows)
    errors_df = pd.DataFrame(pair_errors)
    load_errors_df = pd.DataFrame(load_errors)

    excel_path = out_root / f"all_pairs_scoring_{run_id}.xlsx"
    excel_engine: str
    try:
        import xlsxwriter  # noqa: F401

        excel_engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401

            excel_engine = "openpyxl"
        except Exception as exc:
            raise RuntimeError("Install xlsxwriter or openpyxl to export Excel.") from exc

    with pd.ExcelWriter(excel_path, engine=excel_engine) as writer:
        summary_df.to_excel(writer, sheet_name="scores", index=False)
        criteria_df.to_excel(writer, sheet_name="criteria", index=False)
        errors_df.to_excel(writer, sheet_name="pair_errors", index=False)
        load_errors_df.to_excel(writer, sheet_name="load_errors", index=False)

    summary_df.to_csv(out_root / f"all_pairs_scoring_{run_id}.csv", index=False)
    criteria_df.to_csv(out_root / f"all_pairs_criteria_{run_id}.csv", index=False)
    errors_df.to_csv(out_root / f"pair_errors_{run_id}.csv", index=False)
    load_errors_df.to_csv(out_root / f"load_errors_{run_id}.csv", index=False)

    run_meta = {
        "run_date": run_ts.date().isoformat(),
        "run_id": run_id,
        "output_dir": str(out_root),
        "excel": str(excel_path),
        "pairs_total": int(len(pairs)),
        "pairs_ok": int(len(summary_df)),
        "pairs_failed": int(len(errors_df)),
        "symbols_failed_load": int(len(load_errors_df)),
    }
    (out_root / f"run_meta_{run_id}.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_meta


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _build_config(args)
    meta = run_batch(cfg)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
