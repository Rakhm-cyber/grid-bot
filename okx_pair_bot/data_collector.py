"""
DataCollector — модуль сбора рыночных данных с OKX и расчёта индикаторов.

Отвечает за:
- Загрузку OHLCV-свечей по парам с OKX
- Вычисление EMA логарифмического спреда
- Вычисление волатильности (MAD или STD)
- Вычисление Z-Score
- Получение текущих bid/ask цен (для аномалий и входов)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpreadSnapshot:
    """Текущее состояние спреда и индикаторов."""
    timestamp: float
    spread: float          # ln(price_base / price_quote)
    ema: float             # EMA спреда
    sigma: float           # волатильность
    zscore: float          # (spread - ema) / sigma
    price_base: float      # текущая цена base актива
    price_quote: float     # текущая цена quote актива
    bid_base: float = 0.0
    ask_base: float = 0.0
    bid_quote: float = 0.0
    ask_quote: float = 0.0
    volume_base: float = 0.0    # объём за последнюю свечу
    volume_quote: float = 0.0


@dataclass
class IndicatorState:
    """Внутреннее состояние индикаторов (для накопительного EMA)."""
    ema: float = 0.0
    sigma: float = 0.0
    spread_history: list = field(default_factory=list)
    ema_history: list = field(default_factory=list)    # серия EMA для пересчёта σ
    candle_history: list = field(default_factory=list)  # [(timestamp, spread_price)]
    initialized: bool = False


class DataCollector:
    """
    Собирает данные с OKX и вычисляет индикаторы для пары активов.

    Параметры:
        exchange: инициализированный ccxt.okx
        cfg: словарь конфигурации
    """

    # Коэффициент Фибоначчи для EMA: alpha = 2/(period+1)
    _TIMEFRAME_MAP = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }

    def __init__(self, exchange, cfg: dict):
        self.exchange = exchange
        self.cfg = cfg

        self.base_sym = cfg["symbols"]["base"]
        self.quote_sym = cfg["symbols"]["quote"]
        self.timeframe = cfg.get("base_timeframe", "1h")
        self.ema_period = int(cfg.get("ema_period", 20))
        self.vol_period = int(cfg.get("volatility_period", 24))
        self.vol_type = cfg.get("volatility_type", "mad")  # mad | std
        self.use_hl2 = cfg.get("use_hl2_ema", True)
        self.price_type = cfg.get("price_type_for_ema", "hl2")  # hl2 | ohlc4 | close

        # alpha для EMA
        self._alpha = 2.0 / (self.ema_period + 1)

        self._state = IndicatorState()

    # ──────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Загружает исторические свечи и инициализирует EMA/σ.
        Вызывается один раз при старте.
        """
        needed = max(self.ema_period, self.vol_period) + 10
        logger.info(
            "Инициализация DataCollector: загрузка %d свечей (%s) для %s/%s",
            needed, self.timeframe, self.base_sym, self.quote_sym,
        )

        candles_base = self._fetch_ohlcv(self.base_sym, self.timeframe, needed)
        candles_quote = self._fetch_ohlcv(self.quote_sym, self.timeframe, needed)

        if candles_base is None or candles_quote is None:
            logger.error("Не удалось загрузить исторические свечи")
            return False

        df = self._merge_candles(candles_base, candles_quote)
        if df is None or len(df) < self.vol_period:
            logger.error("Недостаточно данных для инициализации")
            return False

        # Рассчитываем спред по историческим данным
        spreads = np.log(df["price_base"].values / df["price_quote"].values)

        # Инициализируем EMA простым средним первых ema_period значений
        init_ema = float(np.mean(spreads[: self.ema_period]))
        ema = init_ema
        for s in spreads[self.ema_period :]:
            ema = self._alpha * s + (1 - self._alpha) * ema

        # Волатильность по последним vol_period значениям
        recent_spreads = spreads[-self.vol_period :]
        recent_ema_values = self._recompute_ema_series(spreads, self.ema_period)[-self.vol_period :]
        sigma = self._calc_volatility(recent_spreads, recent_ema_values)

        self._state.ema = ema
        self._state.sigma = max(sigma, 1e-8)  # защита от нулевой σ
        self._state.spread_history = list(spreads[-self.vol_period :])
        self._state.ema_history = list(recent_ema_values)
        self._state.initialized = True

        logger.info(
            "Инициализация завершена: EMA=%.6f, σ=%.6f, Z=%.3f",
            ema, sigma, (spreads[-1] - ema) / max(sigma, 1e-8),
        )
        return True

    def get_snapshot(self) -> Optional[SpreadSnapshot]:
        """
        Получает текущие цены с OKX и возвращает SpreadSnapshot.
        Обновляет EMA/σ на основе текущих данных.
        """
        if not self._state.initialized:
            logger.warning("DataCollector не инициализирован")
            return None

        try:
            ticker_base = self._fetch_ticker(self.base_sym)
            ticker_quote = self._fetch_ticker(self.quote_sym)
        except Exception as exc:
            logger.error("Ошибка получения тикера: %s", exc)
            return None

        if ticker_base is None or ticker_quote is None:
            return None

        price_base = ticker_base["last"]
        price_quote = ticker_quote["last"]

        if price_base <= 0 or price_quote <= 0:
            logger.warning("Некорректные цены: base=%s, quote=%s", price_base, price_quote)
            return None

        spread = math.log(price_base / price_quote)

        # Обновляем EMA
        self._state.ema = self._alpha * spread + (1 - self._alpha) * self._state.ema

        # Обновляем историю и σ
        self._state.spread_history.append(spread)
        if len(self._state.spread_history) > self.vol_period:
            self._state.spread_history = self._state.spread_history[-self.vol_period :]

        ema_series = self._recompute_ema_series(
            np.array(self._state.spread_history), self.ema_period
        )
        self._state.sigma = max(
            self._calc_volatility(np.array(self._state.spread_history), ema_series),
            1e-8,
        )

        zscore = (spread - self._state.ema) / self._state.sigma

        return SpreadSnapshot(
            timestamp=time.time(),
            spread=spread,
            ema=self._state.ema,
            sigma=self._state.sigma,
            zscore=zscore,
            price_base=price_base,
            price_quote=price_quote,
            bid_base=ticker_base.get("bid") or 0.0,
            ask_base=ticker_base.get("ask") or 0.0,
            bid_quote=ticker_quote.get("bid") or 0.0,
            ask_quote=ticker_quote.get("ask") or 0.0,
            volume_base=ticker_base.get("baseVolume") or 0.0,
            volume_quote=ticker_quote.get("baseVolume") or 0.0,
        )

    def get_candle_volume(self, symbol: str, lookback_hours: int) -> tuple[float, float]:
        """
        Возвращает (текущий_объём, средний_объём_за_lookback_hours).
        Используется детектором аномалий.
        """
        try:
            candles = self._fetch_ohlcv(symbol, self.timeframe, lookback_hours + 2)
            if candles is None or len(candles) < 2:
                return 0.0, 0.0
            volumes = [c[5] for c in candles]  # volume — 6-й элемент OHLCV
            current_vol = volumes[-1]
            avg_vol = float(np.mean(volumes[:-1]))
            return current_vol, avg_vol
        except Exception as exc:
            logger.error("Ошибка получения объёма для %s: %s", symbol, exc)
            return 0.0, 0.0

    def get_volatility_1h(self) -> float:
        """
        Возвращает σ за последний час (1 свечу timeframe).
        Используется для детектора спайков волатильности.
        """
        try:
            candles = self._fetch_ohlcv(self.base_sym, "1h", 3)
            if candles is None or len(candles) < 2:
                return 0.0
            # Используем HLC диапазон как прокси волатильности
            last = candles[-2]  # предыдущая закрытая свеча
            hi, lo = last[2], last[3]
            return (hi - lo) / lo if lo > 0 else 0.0
        except Exception:
            return 0.0

    def get_confirmation_candle(self, level: float) -> Optional[bool]:
        """
        Проверяет подтверждение входа на confirmation_timeframe.
        Возвращает True если последняя закрытая свеча пересекла уровень,
        None если данных нет.
        """
        tf = self.cfg.get("confirmation_timeframe", "5m")
        try:
            candles_b = self._fetch_ohlcv(self.base_sym, tf, 3)
            candles_q = self._fetch_ohlcv(self.quote_sym, tf, 3)
            if not candles_b or not candles_q or len(candles_b) < 2:
                return None

            # Берём предпоследнюю (уже закрытую) свечу
            cb = candles_b[-2]
            cq = candles_q[-2]

            close_b = cb[4]
            close_q = cq[4]
            if close_b <= 0 or close_q <= 0:
                return None

            close_spread = math.log(close_b / close_q)
            current_spread = self._state.spread_history[-1] if self._state.spread_history else close_spread

            # Подтверждение: свеча закрылась по ту же сторону уровня
            if level > self._state.ema:
                return close_spread >= level
            else:
                return close_spread <= level
        except Exception as exc:
            logger.debug("Ошибка проверки подтверждения: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[list]:
        """Загружает OHLCV с OKX с retry."""
        for attempt in range(5):
            try:
                data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if data:
                    return data
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Попытка %d/%d: ошибка fetch_ohlcv(%s, %s): %s. Повтор через %ds",
                    attempt + 1, 5, symbol, timeframe, exc, wait,
                )
                time.sleep(wait)
        return None

    def _fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Получает тикер с OKX с retry."""
        for attempt in range(5):
            try:
                return self.exchange.fetch_ticker(symbol)
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Попытка %d/%d: ошибка fetch_ticker(%s): %s. Повтор через %ds",
                    attempt + 1, 5, symbol, exc, wait,
                )
                time.sleep(wait)
        return None

    def _merge_candles(self, candles_base: list, candles_quote: list) -> Optional[pd.DataFrame]:
        """Объединяет свечи двух инструментов по timestamp."""
        def to_df(candles, prefix):
            df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts")
            # Вычисляем репрезентативную цену
            if self.price_type == "hl2":
                df[f"price_{prefix}"] = (df["high"] + df["low"]) / 2
            elif self.price_type == "ohlc4":
                df[f"price_{prefix}"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
            else:
                df[f"price_{prefix}"] = df["close"]
            df[f"volume_{prefix}"] = df["volume"]
            return df[[f"price_{prefix}", f"volume_{prefix}"]]

        try:
            df_b = to_df(candles_base, "base")
            df_q = to_df(candles_quote, "quote")
            merged = df_b.join(df_q, how="inner")
            merged = merged.dropna()
            # Фильтрация нулевых цен
            merged = merged[(merged["price_base"] > 0) & (merged["price_quote"] > 0)]
            return merged
        except Exception as exc:
            logger.error("Ошибка объединения свечей: %s", exc)
            return None

    def _recompute_ema_series(self, spreads: np.ndarray, period: int) -> np.ndarray:
        """Пересчитывает серию EMA для массива спредов."""
        alpha = 2.0 / (period + 1)
        ema_vals = np.zeros(len(spreads))
        ema_vals[0] = spreads[0]
        for i in range(1, len(spreads)):
            ema_vals[i] = alpha * spreads[i] + (1 - alpha) * ema_vals[i - 1]
        return ema_vals

    def _calc_volatility(self, spreads: np.ndarray, ema_vals: np.ndarray) -> float:
        """Вычисляет волатильность по выбранному методу (MAD или STD)."""
        deviations = spreads - ema_vals
        if self.vol_type == "mad":
            return float(np.mean(np.abs(deviations)))
        else:  # std
            returns = np.diff(spreads)
            return float(np.std(returns)) if len(returns) > 1 else float(np.std(deviations))

    @property
    def current_ema(self) -> float:
        return self._state.ema

    @property
    def current_sigma(self) -> float:
        return self._state.sigma
