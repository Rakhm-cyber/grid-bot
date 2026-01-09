import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from tqdm import tqdm


# Корень репозитория нужен, чтобы при запуске файла напрямую
# корректно подтягивались локальные модули проекта (anal.*).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anal.pairtool.tests import build_spread, half_life, hurst_exponent


# Константы времени и настройки данных:
# TF_SUFFIX определяет таймфрейм исходных CSV,
# ZSCORE_BAND задает коридор для доли времени в пределах |z| <= band,
# HL_MIN/HL_MAX — диапазон для нормализации half-life в скоринге.
MS_IN_DAY = 24 * 60 * 60 * 1000
TF_SUFFIX = "15m"
ZSCORE_BAND = 1.0
HL_MIN = 20.0
HL_MAX = 800.0


def _read_coins(path: Path) -> list[str]:
    # Читаем список тикеров для перебора всех пар.
    # Пустые строки и комментарии (#) игнорируем.
    coins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if not val or val.startswith("#"):
            continue
        coins.append(val)
    return coins


def _read_returns(data_dir: Path, symbol: str) -> pd.DataFrame:
    # Загружаем цены закрытия и считаем простые доходности (pct_change).
    # Далее работаем только с синхронизированными рядами по ts_ms.
    path = data_dir / f"{symbol}_{TF_SUFFIX}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, usecols=["ts_ms", "close"])
    df = df.dropna(subset=["ts_ms", "close"]).sort_values("ts_ms")
    df["returns"] = df["close"].pct_change()
    return df[["ts_ms", "close", "returns"]].dropna()


def _corr_window(
    df1: pd.DataFrame, df2: pd.DataFrame, start_ms: int, end_ms: int, col: str, method: str
) -> float:
    # Корреляция синхронизированных рядов внутри окна.
    # Сначала фильтруем по времени, затем выравниваем по общему ts_ms.
    w1 = df1[(df1["ts_ms"] >= start_ms) & (df1["ts_ms"] <= end_ms)]
    w2 = df2[(df2["ts_ms"] >= start_ms) & (df2["ts_ms"] <= end_ms)]
    if w1.empty or w2.empty:
        return np.nan
    merged = w1.merge(w2, on="ts_ms", how="inner", suffixes=("_1", "_2"))
    if len(merged) < 2:
        return np.nan
    return float(merged[f"{col}_1"].corr(merged[f"{col}_2"], method=method))


def _coint_pvalue(df1: pd.DataFrame, df2: pd.DataFrame, start_ms: int, end_ms: int) -> float:
    # P-value теста коинтеграции Engle-Granger на лог-ценах.
    # Низкий p-value указывает на совместное долгосрочное равновесие.
    w1 = df1[(df1["ts_ms"] >= start_ms) & (df1["ts_ms"] <= end_ms)]
    w2 = df2[(df2["ts_ms"] >= start_ms) & (df2["ts_ms"] <= end_ms)]
    if w1.empty or w2.empty:
        return np.nan
    merged = w1.merge(w2, on="ts_ms", how="inner", suffixes=("_1", "_2"))
    if len(merged) < 3:
        return np.nan
    try:
        _, pvalue, _ = coint(np.log(merged["close_1"]), np.log(merged["close_2"]))
    except Exception:
        return np.nan
    return float(pvalue)


def _month_spread_metrics(
    df1: pd.DataFrame, df2: pd.DataFrame, start_ms: int, end_ms: int, z_band: float
) -> tuple[float, float, float]:
    # Метрики mean reversion по спреду в месячном окне:
    # half-life — скорость возврата к среднему,
    # Hurst — склонность к mean reversion (< 0.5),
    # z-score in band — доля времени, когда спред "в пределах нормы".
    w1 = df1[(df1["ts_ms"] >= start_ms) & (df1["ts_ms"] <= end_ms)]
    w2 = df2[(df2["ts_ms"] >= start_ms) & (df2["ts_ms"] <= end_ms)]
    if w1.empty or w2.empty:
        return np.nan, np.nan, np.nan
    merged = w1.merge(w2, on="ts_ms", how="inner", suffixes=("_1", "_2"))
    if len(merged) < 5:
        return np.nan, np.nan, np.nan
    log_1 = np.log(merged["close_1"])
    log_2 = np.log(merged["close_2"])
    # Спред строим через OLS-хедж:
    # spread = log(y) - (alpha + beta * log(x)).
    spread_model = build_spread(log_1, log_2, include_intercept=True)
    spread = spread_model.spread.dropna()
    hl = half_life(spread)
    hurst = hurst_exponent(spread)
    if len(spread) < 2:
        return hl if hl is not None else np.nan, hurst if hurst is not None else np.nan, np.nan
    # Z-score: доля времени в коридоре |z| <= band, в долях (0..1).
    std = spread.std(ddof=0)
    if std == 0 or np.isnan(std):
        z_pct = np.nan
    else:
        z = (spread - spread.mean()) / std
        z_pct = float((z.abs() <= z_band).mean())
    return (
        hl if hl is not None else np.nan,
        hurst if hurst is not None else np.nan,
        z_pct,
    )


def _windows(end_ms: int) -> dict[str, tuple[int, int]]:
    # Фиксированные окна недели/месяц от последней общей точки:
    # week1 — самая свежая неделя, далее назад до week4.
    week = 7 * MS_IN_DAY
    return {
        "week1": (end_ms - week + 1, end_ms),
        "week2": (end_ms - 2 * week + 1, end_ms - week),
        "week3": (end_ms - 3 * week + 1, end_ms - 2 * week),
        "week4": (end_ms - 4 * week + 1, end_ms - 3 * week),
        "month": (end_ms - 30 * MS_IN_DAY + 1, end_ms),
    }


def main() -> int:
    start_ts = time.perf_counter()
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(
        description="Compute pairwise correlations for last month and last 4 weeks.",
    )
    parser.add_argument("--coins-file", default=repo_root / "coins.txt", type=Path)
    parser.add_argument("--data-dir", default=repo_root / "data" / "binance", type=Path)
    parser.add_argument("--output", default=repo_root / "anal" / "output" / "corr_pairs.csv", type=Path)
    parser.add_argument("--zscore-band", default=ZSCORE_BAND, type=float)
    parser.add_argument("--hl-min", default=HL_MIN, type=float)
    parser.add_argument("--hl-max", default=HL_MAX, type=float)
    parser.add_argument("--refresh-data", action="store_true", help="Run run.py to update CSV data before analysis")
    args = parser.parse_args()

    if args.refresh_data:
        subprocess.run([sys.executable, "run.py"], check=True, cwd=repo_root)

    # Загружаем символы и их ряды доходностей.
    # end_ms берем минимальный, чтобы у всех пар были данные на одном хвосте.
    coins = _read_coins(args.coins_file)
    if len(coins) < 2:
        raise ValueError("Need at least 2 coins in coins file.")

    series = {}
    end_times = []
    for coin in coins:
        df = _read_returns(args.data_dir, coin)
        if df.empty:
            raise ValueError(f"No data for {coin} in {args.data_dir}")
        series[coin] = df
        end_times.append(int(df["ts_ms"].iloc[-1]))

    end_ms = min(end_times)
    win = _windows(end_ms)

    # Метрики по всем парам монет.
    # Для каждой пары считаем недельные и месячные показатели.
    rows = []
    total_pairs = len(coins) * (len(coins) - 1) // 2
    for i in tqdm(range(len(coins)), desc="Pairs", total=len(coins)):
        for j in range(i + 1, len(coins)):
            c1, c2 = coins[i], coins[j]
            df1, df2 = series[c1], series[c2]
            hl, hurst, z_pct = _month_spread_metrics(df1, df2, *win["month"], args.zscore_band)
            row = {
                "coin1": c1,
                "coin2": c2,
                "kendall_week1": _corr_window(df1, df2, *win["week1"], col="returns", method="kendall"),
                "kendall_week2": _corr_window(df1, df2, *win["week2"], col="returns", method="kendall"),
                "kendall_week3": _corr_window(df1, df2, *win["week3"], col="returns", method="kendall"),
                "kendall_week4": _corr_window(df1, df2, *win["week4"], col="returns", method="kendall"),
                "kendall_month": _corr_window(df1, df2, *win["month"], col="returns", method="kendall"),
                "coint_p_week1": _coint_pvalue(df1, df2, *win["week1"]),
                "coint_p_week2": _coint_pvalue(df1, df2, *win["week2"]),
                "coint_p_week3": _coint_pvalue(df1, df2, *win["week3"]),
                "coint_p_week4": _coint_pvalue(df1, df2, *win["week4"]),
                "coint_p_month": _coint_pvalue(df1, df2, *win["month"]),
                "half_life_month": hl,
                "hurst_month": hurst,
                "zscore_in_band_month": z_pct,
            }
            rows.append(row)

    # Формируем таблицу в стабильном порядке колонок,
    # чтобы выходной CSV был удобен для дальнейшей обработки.
    out_df = pd.DataFrame(
        rows,
        columns=[
            "coin1",
            "coin2",
            "kendall_week1",
            "kendall_week2",
            "kendall_week3",
            "kendall_week4",
            "kendall_month",
            "coint_p_week1",
            "coint_p_week2",
            "coint_p_week3",
            "coint_p_week4",
            "coint_p_month",
            "half_life_month",
            "hurst_month",
            "zscore_in_band_month",
        ],
    )
    p_cols = [
        "coint_p_week1",
        "coint_p_week2",
        "coint_p_week3",
        "coint_p_week4",
        "coint_p_month",
    ]
    k_cols = [
        "kendall_week1",
        "kendall_week2",
        "kendall_week3",
        "kendall_week4",
        "kendall_month",
    ]

    def _clip01(val: float) -> float:
        if np.isnan(val):
            return np.nan
        return float(np.clip(val, 0.0, 1.0))

    def _score_row(row: pd.Series) -> pd.Series:
        k_vals = row[k_cols].astype(float).to_numpy()
        p_vals = row[p_cols].astype(float).to_numpy()
        kendall_med = float(np.nanmedian(k_vals)) if np.any(~np.isnan(k_vals)) else np.nan
        _ = float(np.nanstd(k_vals[:4])) if np.any(~np.isnan(k_vals[:4])) else np.nan
        if np.any(~np.isnan(p_vals)):
            pass_rate = float(np.nanmean(p_vals < 0.05))
            p_med = float(np.nanmedian(p_vals))
            logp = float(-np.log10(p_med + 1e-8))
        else:
            pass_rate = np.nan
            p_med = np.nan
            logp = np.nan

        s_k = _clip01((kendall_med - 0.30) / 0.40)
        s_c_log = _clip01((logp - 1.0) / 3.0)
        s_c = np.nan
        if not np.isnan(s_c_log) and not np.isnan(pass_rate):
            s_c = float(0.5 * s_c_log + 0.5 * pass_rate)

        hl = float(row["half_life_month"]) if not np.isnan(row["half_life_month"]) else np.nan
        hurst = float(row["hurst_month"]) if not np.isnan(row["hurst_month"]) else np.nan
        z_band = float(row["zscore_in_band_month"]) if not np.isnan(row["zscore_in_band_month"]) else np.nan
        s_hl = _clip01(1.0 - (hl - args.hl_min) / (args.hl_max - args.hl_min)) if not np.isnan(hl) else np.nan
        s_h = _clip01((0.55 - hurst) / 0.20) if not np.isnan(hurst) else np.nan
        s_z = _clip01((z_band - 0.40) / 0.40) if not np.isnan(z_band) else np.nan

        s_mr = np.nan
        if not np.isnan(s_hl) and not np.isnan(s_h) and not np.isnan(s_z):
            s_mr = float(0.4 * s_hl + 0.3 * s_h + 0.3 * s_z)

        gate = 0
        if not np.isnan(pass_rate) and not np.isnan(row["coint_p_month"]) and not np.isnan(hurst):
            gate = int(((pass_rate >= 0.4) or (row["coint_p_month"] < 0.05)) and (hurst < 0.5))

        final_score = np.nan
        if not np.isnan(s_c) and not np.isnan(s_mr) and not np.isnan(s_k):
            final_score = float(0.45 * s_c + 0.40 * s_mr + 0.15 * s_k)

        return pd.Series(
            {
                "s_c": s_c,
                "gate": gate,
                "s_mr": s_mr,
                "s_k": s_k,
                "final_score": final_score,
            }
        )

    scores = out_df.apply(_score_row, axis=1)
    out_df = pd.concat([out_df, scores], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".xlsx":
        try:
            out_df.to_excel(args.output, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to write Excel file. Install openpyxl or xlsxwriter."
            ) from exc
    else:
        out_df.to_csv(args.output, index=False)

    elapsed = time.perf_counter() - start_ts
    print(f"Wrote {len(out_df)} rows to {args.output}")
    print(f"Elapsed: {elapsed:.2f}s for {total_pairs} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
