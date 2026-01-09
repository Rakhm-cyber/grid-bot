from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config


@dataclass
class LoadedSeries:
    symbol: str
    df: pd.DataFrame


def _parse_timestamp(df: pd.DataFrame) -> pd.DatetimeIndex:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "ts_iso" in df.columns:
        ts = pd.to_datetime(df["ts_iso"], utc=True, errors="coerce")
    elif "ts_ms" in df.columns:
        ts = pd.to_datetime(df["ts_ms"], utc=True, errors="coerce", unit="ms")
    else:
        raise ValueError("No timestamp column found. Expected one of: timestamp, ts_iso, ts_ms")
    if ts.isna().any():
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]
    return pd.DatetimeIndex(ts)


def _finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = _parse_timestamp(df)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _symbol_path(cfg: Config, symbol: str) -> Path:
    suffix = cfg.timeframe
    filename = f"{symbol}_{suffix}.csv"
    return Path(cfg.input_path) / filename


def load_from_csv(cfg: Config, symbol: str) -> LoadedSeries:
    path = _symbol_path(cfg, symbol)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = _finalize_df(df)
    return LoadedSeries(symbol=symbol, df=df)


def load_from_parquet(cfg: Config, symbol: str) -> LoadedSeries:
    path = _symbol_path(cfg, symbol).with_suffix(".parquet")
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    df = _finalize_df(df)
    return LoadedSeries(symbol=symbol, df=df)


def load_from_binance(cfg: Config, symbol: str, start: Optional[str], end: Optional[str]) -> LoadedSeries:
    from src.clients.binance_client import BinanceClient
    client = BinanceClient()
    start_dt = datetime.fromisoformat(start) if start else datetime.now(timezone.utc)
    end_dt = datetime.fromisoformat(end) if end else None
    interval = cfg.timeframe
    path = _symbol_path(cfg, symbol)
    client.get_rates_and_save(start_dt, interval, symbol, str(path), end_dt=end_dt)
    df = pd.read_csv(path)
    df = _finalize_df(df)
    return LoadedSeries(symbol=symbol, df=df)


def load_series(cfg: Config, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> LoadedSeries:
    if cfg.data_source == "csv":
        return load_from_csv(cfg, symbol)
    if cfg.data_source == "parquet":
        return load_from_parquet(cfg, symbol)
    if cfg.data_source in {"binance", "ccxt"}:
        return load_from_binance(cfg, symbol, start, end)
    raise ValueError(f"Unsupported data_source: {cfg.data_source}")
