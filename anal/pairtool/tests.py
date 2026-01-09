from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa import stattools


@dataclass
class SpreadModel:
    alpha: float
    beta: float
    spread: pd.Series


def _corr_with_pvalues(x: pd.Series, y: pd.Series, method: str) -> Dict[str, float]:
    # Коэффициент корреляции и p-value для линейной/монотонной связи:
    # Pearson — линейная, Spearman/Kendall — монотонная.
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    elif method == "kendall":
        r, p = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return {"value": float(r), "p_value": float(p)}


def correlations(x: pd.Series, y: pd.Series, methods: List[str]) -> Dict[str, Dict[str, float]]:
    # Пакетный расчет корреляций с p-value для нескольких методов,
    # чтобы сравнивать разные типы зависимости.
    return {m: _corr_with_pvalues(x, y, m) for m in methods}


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 20) -> Dict[str, float]:
    # Максимальная по модулю кросс-корреляция на сетке лагов.
    # Полезно видеть, есть ли сдвиг по времени между рядами.
    x = x - x.mean()
    y = y - y.mean()
    lags = range(-max_lag, max_lag + 1)
    best = {"lag": 0, "corr": 0.0}
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            continue
        if abs(corr) > abs(best["corr"]):
            best = {"lag": int(lag), "corr": float(corr)}
    return {"max_abs_corr": float(best["corr"]), "lag": int(best["lag"])}


def adf_test(series: pd.Series) -> Dict[str, float]:
    # ADF-тест на единичный корень (стационарность).
    # Низкий p-value => ряд стационарен.
    res = stattools.adfuller(series.dropna(), autolag="AIC")
    return {
        "stat": float(res[0]),
        "p_value": float(res[1]),
        "used_lag": int(res[2]),
        "nobs": int(res[3]),
        "crit_values": {k: float(v) for k, v in res[4].items()},
    }


def kpss_test(series: pd.Series) -> Dict[str, float]:
    # KPSS-тест стационарности (константа в регрессии).
    # Низкий p-value => ряд НЕстационарен (обратная логика к ADF).
    res = stattools.kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "stat": float(res[0]),
        "p_value": float(res[1]),
        "lags": int(res[2]),
        "crit_values": {k: float(v) for k, v in res[3].items()},
    }


def pp_test(series: pd.Series) -> Optional[Dict[str, float]]:
    # Phillips-Perron тест на единичный корень (если доступен).
    # Альтернатива ADF с другой устойчивостью к автокорреляции.
    try:
        from statsmodels.tsa.stattools import phillips_perron
    except Exception:
        phillips_perron = None
    if phillips_perron is None:
        return None
    res = phillips_perron(series.dropna())
    return {"stat": float(res[0]), "p_value": float(res[1])}


def engle_granger(y: pd.Series, x: pd.Series, include_intercept: bool) -> Dict[str, float]:
    # Engle-Granger тест коинтеграции (ADF по остаткам OLS).
    # Проверяет, существует ли стабильный спред между двумя рядами.
    X = sm.add_constant(x) if include_intercept else x.to_frame("x")
    model = sm.OLS(y, X).fit()
    alpha = float(model.params["const"]) if include_intercept else 0.0
    if include_intercept:
        beta = float(model.params.drop("const").iloc[0])
    else:
        beta = float(model.params.iloc[0])
    resid = model.resid
    adf = adf_test(resid)
    return {
        "alpha": alpha,
        "beta": beta,
        "resid_adf_stat": adf["stat"],
        "resid_adf_p": adf["p_value"],
    }


def johansen(y: pd.Series, x: pd.Series) -> Optional[Dict[str, List[float]]]:
    # Johansen тест коинтеграции (trace / max-eigen статистики).
    # Позволяет оценить число коинтеграционных связей.
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except Exception:
        return None
    data = pd.concat([y, x], axis=1).dropna()
    res = coint_johansen(data, det_order=0, k_ar_diff=1)
    return {
        "trace_stat": [float(v) for v in res.lr1],
        "trace_crit": [float(v) for v in res.cvt[:, 1]],
        "max_eig_stat": [float(v) for v in res.lr2],
        "max_eig_crit": [float(v) for v in res.cvm[:, 1]],
    }


def build_spread(y: pd.Series, x: pd.Series, include_intercept: bool) -> SpreadModel:
    # Линейный хедж через OLS для построения спреда:
    # это базовая модель "y ~ alpha + beta*x".
    X = sm.add_constant(x) if include_intercept else x.to_frame("x")
    model = sm.OLS(y, X).fit()
    alpha = float(model.params["const"]) if include_intercept else 0.0
    if include_intercept:
        beta = float(model.params.drop("const").iloc[0])
    else:
        beta = float(model.params.iloc[0])
    spread = y - (alpha + beta * x)
    return SpreadModel(alpha=alpha, beta=beta, spread=spread)


def half_life(spread: pd.Series) -> Optional[float]:
    # Half-life из скорости возврата к среднему (AR(1)):
    # сколько баров нужно, чтобы отклонение сократилось вдвое.
    s = spread.dropna()
    if len(s) < 5:
        return None
    delta = s.diff().dropna()
    lagged = s.shift(1).dropna().loc[delta.index]
    model = sm.OLS(delta, sm.add_constant(lagged)).fit()
    phi = float(model.params.iloc[1])
    if 1 + phi <= 0:
        return None
    return float(-np.log(2) / np.log(1 + phi))


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> Optional[float]:
    # Экспонента Херста через наклон на log-log лагах:
    # H < 0.5 — mean reversion, H ~ 0.5 — случайное блуждание, H > 0.5 — тренд.
    s = series.dropna().values
    if len(s) < max_lag + 2:
        return None
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(s[lag:] - s[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)


def normality_tests(spread: pd.Series, returns_y: pd.Series, returns_x: pd.Series, tests: List[str]) -> Dict[str, Dict[str, float]]:
    # Диагностика нормальности и сходства распределений:
    # полезно понять, насколько предположения нормальности валидны.
    out: Dict[str, Dict[str, float]] = {}
    if "jarque_bera" in tests:
        jb = stats.jarque_bera(spread.dropna())
        out["jarque_bera"] = {"stat": float(jb.statistic), "p_value": float(jb.pvalue)}
    if "anderson_darling" in tests:
        ad = stats.anderson(spread.dropna(), dist="norm")
        out["anderson_darling"] = {"stat": float(ad.statistic)}
    if "ks" in tests:
        s = spread.dropna()
        s_z = (s - s.mean()) / s.std(ddof=0)
        ks1 = stats.kstest(s_z, "norm")
        ks2 = stats.ks_2samp(returns_y.dropna(), returns_x.dropna())
        out["ks_spread_norm"] = {"stat": float(ks1.statistic), "p_value": float(ks1.pvalue)}
        out["ks_returns"] = {"stat": float(ks2.statistic), "p_value": float(ks2.pvalue)}
    return out


def granger_causality(returns_y: pd.Series, returns_x: pd.Series, max_lag: int) -> Dict[str, Dict[str, float]]:
    # F-тесты причинности Грейнджера до заданного лага:
    # показывает, улучшает ли один ряд прогноз другого.
    data = pd.concat([returns_y, returns_x], axis=1).dropna()
    res = stattools.grangercausalitytests(data, max_lag, verbose=False)
    out: Dict[str, Dict[str, float]] = {}
    for lag, tests in res.items():
        ftest = tests[0].get("ssr_ftest", None)
        if ftest:
            out[str(lag)] = {"stat": float(ftest[0]), "p_value": float(ftest[1])}
    return out
