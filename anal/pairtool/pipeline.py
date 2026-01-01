from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

from .config import Config
from .loaders import load_series
from .preprocessing import align_and_transform
from .reporting import (
    ensure_dirs,
    plot_chow,
    plot_rolling,
    plot_series,
    plot_spread,
    plot_zscore,
    save_csv,
    save_json,
    write_html_report,
)
from .rolling import compute_rolling
from .tests import (
    adf_test,
    build_spread,
    correlations,
    cross_correlation,
    engle_granger,
    granger_causality,
    hurst_exponent,
    johansen,
    kpss_test,
    normality_tests,
    pp_test,
    half_life,
)


logger = logging.getLogger("pairtool")


def _setup_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _chow_test(y: pd.Series, x: pd.Series, grid_points: int) -> pd.DataFrame:
    data = pd.concat([y, x], axis=1).dropna()
    n = len(data)
    if n < 10:
        return pd.DataFrame()
    idx = np.linspace(int(n * 0.2), int(n * 0.8), grid_points).astype(int)
    results = []
    for bp in idx:
        y1, x1 = data.iloc[:bp, 0], data.iloc[:bp, 1]
        y2, x2 = data.iloc[bp:, 0], data.iloc[bp:, 1]
        if len(y1) < 5 or len(y2) < 5:
            continue
        def _ols(yv, xv):
            X = np.column_stack([np.ones(len(xv)), xv])
            beta = np.linalg.lstsq(X, yv, rcond=None)[0]
            resid = yv - X @ beta
            return resid, len(yv), 2
        resid_pooled, n_total, k = _ols(data.iloc[:, 0].values, data.iloc[:, 1].values)
        resid1, n1, _ = _ols(y1.values, x1.values)
        resid2, n2, _ = _ols(y2.values, x2.values)
        sse_pooled = float(np.sum(resid_pooled ** 2))
        sse1 = float(np.sum(resid1 ** 2))
        sse2 = float(np.sum(resid2 ** 2))
        num = (sse_pooled - (sse1 + sse2)) / k
        den = (sse1 + sse2) / (n1 + n2 - 2 * k)
        if den <= 0:
            continue
        f_stat = num / den
        p_val = float(stats.f.sf(f_stat, k, n1 + n2 - 2 * k))
        results.append({
            "break_time": data.index[bp],
            "f_stat": float(f_stat),
            "p_value": p_val,
        })
    return pd.DataFrame(results)


def _score_tradeability(result: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    decision = "WATCH"
    try:
        static_tests = result["static_tests"]
        eg_p = static_tests["cointegration"]["engle_granger"]["resid_adf_p"]
        if eg_p < 0.05:
            score += 40
        adf_p = static_tests["stationarity"]["spread"]["adf"]["p_value"]
        kpss_p = static_tests["stationarity"]["spread"]["kpss"]["p_value"]
        if adf_p < 0.05 and kpss_p > 0.05:
            score += 30
        half_life_val = static_tests["mean_reversion"]["half_life"]
        if half_life_val and 5 <= half_life_val <= 200:
            score += 20
        rolling_ok = result["rolling"]["summary"]["adf_p_ok_pct"]
        if rolling_ok >= 60:
            score += 10
        if score >= 70:
            decision = "PASS"
        elif score < 40:
            decision = "FAIL"
    except Exception:
        pass
    return {"score": score, "decision": decision}


def run_analysis(cfg: Config, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    _setup_logging()
    logger.info("Loading data...")
    y = load_series(cfg, cfg.symbol1, start=start, end=end).df
    x = load_series(cfg, cfg.symbol2, start=start, end=end).df

    logger.info("Aligning and transforming...")
    aligned = align_and_transform(cfg, y, x)
    df = aligned.df
    if df.empty:
        raise ValueError("No overlapping data after alignment")

    logger.info("Building spread model...")
    spread_model = build_spread(df["log_y"], df["log_x"], cfg.include_intercept)
    df["spread"] = spread_model.spread

    logger.info("Running tests...")
    static_tests: Dict[str, Any] = {
        "correlations": correlations(df["returns_y"], df["returns_x"], cfg.correlations),
        "cross_correlation": cross_correlation(df["returns_y"], df["returns_x"]),
        "stationarity": {
            "log_y": {},
            "log_x": {},
            "returns_y": {},
            "returns_x": {},
            "spread": {},
        },
        "cointegration": {},
        "normality": {},
        "causality": {},
        "mean_reversion": {},
    }

    if "adf" in cfg.stationarity:
        static_tests["stationarity"]["log_y"]["adf"] = adf_test(df["log_y"])
        static_tests["stationarity"]["log_x"]["adf"] = adf_test(df["log_x"])
        static_tests["stationarity"]["returns_y"]["adf"] = adf_test(df["returns_y"])
        static_tests["stationarity"]["returns_x"]["adf"] = adf_test(df["returns_x"])
        static_tests["stationarity"]["spread"]["adf"] = adf_test(df["spread"])
    if "kpss" in cfg.stationarity:
        static_tests["stationarity"]["log_y"]["kpss"] = kpss_test(df["log_y"])
        static_tests["stationarity"]["log_x"]["kpss"] = kpss_test(df["log_x"])
        static_tests["stationarity"]["returns_y"]["kpss"] = kpss_test(df["returns_y"])
        static_tests["stationarity"]["returns_x"]["kpss"] = kpss_test(df["returns_x"])
        static_tests["stationarity"]["spread"]["kpss"] = kpss_test(df["spread"])
    if "pp" in cfg.stationarity:
        static_tests["stationarity"]["log_y"]["pp"] = pp_test(df["log_y"])
        static_tests["stationarity"]["log_x"]["pp"] = pp_test(df["log_x"])
        static_tests["stationarity"]["returns_y"]["pp"] = pp_test(df["returns_y"])
        static_tests["stationarity"]["returns_x"]["pp"] = pp_test(df["returns_x"])
        static_tests["stationarity"]["spread"]["pp"] = pp_test(df["spread"])

    if cfg.cointegration_engle_granger:
        static_tests["cointegration"]["engle_granger"] = engle_granger(
            df["log_y"], df["log_x"], cfg.include_intercept
        )
    if cfg.cointegration_johansen:
        static_tests["cointegration"]["johansen"] = johansen(df["log_y"], df["log_x"])

    static_tests["mean_reversion"] = {
        "half_life": half_life(df["spread"]),
        "hurst": hurst_exponent(df["spread"]),
    }

    static_tests["normality"] = normality_tests(
        df["spread"], df["returns_y"], df["returns_x"], cfg.normality
    )

    if cfg.granger_enabled:
        static_tests["causality"]["granger"] = granger_causality(
            df["returns_y"], df["returns_x"], cfg.granger_max_lag
        )

    logger.info("Rolling metrics...")
    rolling_res = compute_rolling(
        df["log_y"],
        df["log_x"],
        df["spread"],
        df["returns_y"],
        df["returns_x"],
        cfg.rolling_window,
        cfg.rolling_step,
        cfg.rolling_min_periods,
        cfg.include_intercept,
    )

    logger.info("Break tests...")
    breaks = _chow_test(df["log_y"], df["log_x"], cfg.chow_grid_points) if cfg.chow_enabled else pd.DataFrame()

    warnings = []
    if len(df) < cfg.rolling_min_periods:
        warnings.append("too_few_observations_for_rolling")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg.output_path) / f"run_{run_id}"
    paths = ensure_dirs(output_root)

    result = {
        "metadata": {
            "symbols": [cfg.symbol1, cfg.symbol2],
            "timeframe": cfg.timeframe,
            "nobs": int(len(df)),
            "preprocessing": {
                "price_field": cfg.price_field,
                "use_log_prices": cfg.use_log_prices,
                "winsorize": cfg.winsorize_enabled,
                "outliers_method": cfg.outliers_method,
                "missing_strategy": cfg.missing_strategy,
            },
        },
        "static_tests": static_tests,
        "spread_model": {
            "alpha": spread_model.alpha,
            "beta": spread_model.beta,
            "method": cfg.hedge_method,
            "include_intercept": cfg.include_intercept,
        },
        "mean_reversion": static_tests["mean_reversion"],
        "rolling": {
            "summary": rolling_res.summary,
        },
        "breaks": {
            "chow": {
                "enabled": cfg.chow_enabled,
                "top_p_values": [] if breaks.empty else breaks.sort_values("p_value").head(5).to_dict("records"),
            }
        },
        "warnings": warnings,
    }
    result["tradeability"] = _score_tradeability(result)

    if "json" in cfg.output_formats:
        paths.result_json = output_root / "result.json"
        save_json(result, paths.result_json)
    if "csv" in cfg.output_formats:
        paths.aligned_csv = output_root / "aligned_series.csv"
        save_csv(df[["log_y", "log_x", "returns_y", "returns_x", "spread"]], paths.aligned_csv)
        paths.rolling_csv = output_root / "rolling_metrics.csv"
        save_csv(rolling_res.df, paths.rolling_csv)
        paths.breaks_csv = output_root / "break_tests.csv"
        save_csv(breaks, paths.breaks_csv)

    if cfg.plots:
        plot_series(df, paths.plots / "log_prices.png")
        plot_spread(df, paths.plots / "spread.png")
        plot_zscore(df, paths.plots / "spread_zscore.png")
        plot_rolling(rolling_res.df, paths.plots / "rolling_beta.png", "beta", "Rolling Beta")
        plot_rolling(rolling_res.df, paths.plots / "rolling_adf_p.png", "adf_p", "Rolling ADF p-value")
        plot_rolling(rolling_res.df, paths.plots / "rolling_corr.png", "corr", "Rolling Correlation")
        if cfg.chow_enabled:
            plot_chow(breaks, paths.plots / "chow_p_values.png")

    if "html" in cfg.output_formats:
        paths.html_report = output_root / "report.html"
        write_html_report(paths.html_report, cfg.report_title, {
            "symbols": f"{cfg.symbol1}/{cfg.symbol2}",
            "timeframe": cfg.timeframe,
            "nobs": str(len(df)),
            "score": str(result["tradeability"]["score"]),
            "decision": result["tradeability"]["decision"],
        })

    logger.info("Done. Output: %s", output_root)
    return result
