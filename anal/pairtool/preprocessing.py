from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class AlignedData:
    df: pd.DataFrame
    y_name: str
    x_name: str


def _apply_missing_strategy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.missing_strategy == "drop":
        return df.dropna()
    if cfg.missing_strategy == "ffill":
        return df.ffill(limit=cfg.missing_ffill_limit)
    if cfg.missing_strategy == "interpolate":
        return df.interpolate(limit=cfg.missing_ffill_limit)
    raise ValueError(f"Unsupported missing strategy: {cfg.missing_strategy}")


def _winsorize(series: pd.Series, limits: Tuple[float, float]) -> pd.Series:
    lower, upper = series.quantile([limits[0], limits[1]])
    return series.clip(lower=lower, upper=upper)


def _remove_outliers_zscore(series: pd.Series, threshold: float) -> pd.Series:
    z = (series - series.mean()) / series.std(ddof=0)
    return series.where(z.abs() <= threshold)


def align_and_transform(cfg: Config, y: pd.DataFrame, x: pd.DataFrame) -> AlignedData:
    price_field = cfg.price_field
    if price_field not in y.columns or price_field not in x.columns:
        raise ValueError(f"price_field '{price_field}' missing in data")

    df = pd.DataFrame({
        "y": y[price_field],
        "x": x[price_field],
    })
    df = _apply_missing_strategy(df, cfg)
    df = df.dropna()

    if cfg.winsorize_enabled:
        df["y"] = _winsorize(df["y"], cfg.winsorize_limits)
        df["x"] = _winsorize(df["x"], cfg.winsorize_limits)

    if cfg.outliers_method == "zscore":
        df["y"] = _remove_outliers_zscore(df["y"], cfg.outliers_threshold)
        df["x"] = _remove_outliers_zscore(df["x"], cfg.outliers_threshold)
    elif cfg.outliers_method not in {"none", ""}:
        raise ValueError(f"Unsupported outliers method: {cfg.outliers_method}")

    df = df.dropna()

    if cfg.use_log_prices:
        df["log_y"] = np.log(df["y"])
        df["log_x"] = np.log(df["x"])
    else:
        df["log_y"] = df["y"]
        df["log_x"] = df["x"]

    df["returns_y"] = df["log_y"].diff()
    df["returns_x"] = df["log_x"].diff()
    df = df.dropna()

    return AlignedData(df=df, y_name="log_y", x_name="log_x")
