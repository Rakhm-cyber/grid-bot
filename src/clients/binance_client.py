# binance_client.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Literal, Optional, Tuple

import requests


# Разрешённые интервалы на Binance (spot /api/v3/klines)
BinanceInterval = Literal[
    "1s","1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
]

# В миллисекундах
INTERVAL_MS: dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,  # упрощённо: 30 дней
}


@dataclass(frozen=True)
class RatePoint:
    ts: datetime   # время открытия свечи (UTC)
    close: float   # цена закрытия


class BinanceClient:
    """
    Лёгкий клиент для публичного Binance Spot API.
    Использует /api/v3/klines, сам пагинирует по startTime/endTime.

    Примечания:
      - Для «курса валюты» возвращает close-цены свечей.
      - По умолчанию QUOTE=USDT (т.е. BTC → BTCUSDT).
      - Без API-ключа (публичная часть). Следи за rate-limit.
    """

    BASE_URL = "https://api.binance.com"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        request_timeout: float = 15.0,
        max_retries: int = 3,
        pause_on_limit_sec: float = 1.0,
    ) -> None:
        self._s = session or requests.Session()
        self._timeout = request_timeout
        self._retries = max_retries
        self._pause_on_limit = pause_on_limit_sec
        self._s.headers.update({
            "User-Agent": "rates-fetcher/1.0 (+binance client)"
        })

    @staticmethod
    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            # считаем, что это UTC, если tz не задан
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _from_ms(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    def _request(self, path: str, params: dict) -> list:
        url = f"{self.BASE_URL}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._retries + 1):
            try:
                r = self._s.get(url, params=params, timeout=self._timeout)
                # 418/429 – лимиты/бан; 5xx – временные проблемы
                if r.status_code in (418, 429):
                    time.sleep(self._pause_on_limit)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                # маленькая пауза и повтор
                time.sleep(0.25 * attempt)
        raise RuntimeError(f"Binance request failed after {self._retries} attempts: {last_exc}")  # noqa: E501

    def _symbol(self, base: str, quote: str) -> str:
        return f"{base.upper()}{quote.upper()}"

    def get_rates(
        self,
        start_dt: datetime,
        interval: BinanceInterval,
        base_currency: str,
        quote_currency: str = "USDT",
        end_dt: Optional[datetime] = None,
        limit_per_call: int = 1000,
    ) -> List[RatePoint]:
        """
        Отдаёт точки (ts, close) от start_dt до сейчас (или end_dt) с заданным интервалом.

        :param start_dt: c какого момента (UTC либо naive=UTC)
        :param interval: интервал свечи (напр. "1m","5m","1h","1d","1w","1M")
        :param base_currency: например "BTC" или "ETH"
        :param quote_currency: например "USDT", "BUSD", "USDC" (по умолчанию USDT)
        :param end_dt: до какого момента (исключительно, опционально). Если None — до текущего времени
        :param limit_per_call: максимум 1000 у Binance
        :return: список RatePoint (время открытия свечи и цена закрытия)
        """
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval '{interval}'. Allowed: {', '.join(INTERVAL_MS)}")

        symbol = self._symbol(base_currency, quote_currency)
        start_ms = self._to_ms(start_dt)
        now_ms = self._to_ms(datetime.now(timezone.utc))
        end_ms = min(self._to_ms(end_dt) if end_dt else now_ms, now_ms)

        if start_ms >= end_ms:
            return []

        step_ms = INTERVAL_MS[interval]
        out: List[RatePoint] = []

        cursor = start_ms
        while cursor < end_ms:
            chunk_end = min(cursor + step_ms * limit_per_call, end_ms)

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": chunk_end - 1,
                "limit": limit_per_call,
            }
            rows = self._request("/api/v3/klines", params)

            if not rows:
                cursor = chunk_end
                continue

            for k in rows:
                open_time_ms = int(k[0])
                close_price = float(k[4])
                if open_time_ms >= end_ms:
                    break
                out.append(RatePoint(ts=self._from_ms(open_time_ms), close=close_price))

            last_open_ms = int(rows[-1][0])
            next_cursor = last_open_ms + step_ms
            cursor = max(next_cursor, chunk_end)

        return out

    def to_csv(
        self,
        points: Iterable[RatePoint],
        filepath: str,
        with_header: bool = True,
    ) -> None:
        import csv
        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            if with_header:
                w.writerow(["ts_utc", "close"])
            for p in points:
                w.writerow([p.ts.isoformat(), f"{p.close:.8f}"])
