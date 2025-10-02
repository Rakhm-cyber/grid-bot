# binance_client.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Literal, Optional

import requests

import settings
from src.common.constants_enums import BinanceInterval, INTERVAL_MS


@dataclass(frozen=True)
class RatePoint:
    ts: datetime
    close: float


class BinanceClient:

    def __init__(
        self,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._s = session or requests.Session()
        self._timeout = settings.BINANCE_REQUEST_TIMOUT
        self._retries = settings.BINANCE_REQUESTS_MAX_RETRIES
        self._pause_on_limit = settings.BINANCE_PAUSE_ON_LIMIT
        self._s.headers.update({
            'User-Agent': 'rates-fetcher/1.0 (+binance client)'
        })


    @staticmethod
    def _datetime_to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _from_ms(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    @staticmethod
    def _to_iso_utc(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _ensure_dir(filepath: str) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)

    @staticmethod
    def _load_existing_csv(filepath: str) -> dict[int, tuple[str, float]]:
        import os, csv
        data: dict[int, tuple[str, float]] = {}
        if not os.path.exists(filepath):
            return data

        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            def iter_rows():
                if header is None:
                    return reader
                if header == ["ts_ms", "ts_iso", "close"]:
                    return reader
                yield header
                yield from reader

            for row in iter_rows():
                if not row:
                    continue
                try:
                    ts_ms = int(row[0])
                    ts_iso = str(row[1])
                    close = float(row[2])
                except Exception:
                    continue
                data[ts_ms] = (ts_iso, close)
        return data

    @staticmethod
    def _atomic_write_csv(filepath: str, rows: list[tuple[int, str, float]]) -> None:
        import os, csv
        tmp_path = f"{filepath}.tmp"
        with open(tmp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_ms", "ts_iso", "close"])
            for ts_ms, ts_iso, close in rows:
                w.writerow([ts_ms, ts_iso, close])
        os.replace(tmp_path, filepath)


    def _request(self, path: str, params: dict) -> list:
        url = f"{settings.BINANCE_BASE_URL}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._retries + 1):
            try:
                r = self._s.get(url, params=params, timeout=self._timeout)
                if r.status_code in (418, 429):
                    time.sleep(self._pause_on_limit)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(0.25 * attempt)
        raise RuntimeError(f"Binance request failed after {self._retries} attempts: {last_exc}")

    def _symbol(self, base: str, quote: str) -> str:
        return f"{base.upper()}{quote.upper()}"


    def get_rates(
        self,
        start_dt: datetime,
        interval: BinanceInterval,
        symbol: str,
        end_dt: Optional[datetime] = None,
        limit_per_call: int = 1000,
    ) -> List[RatePoint]:
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval '{interval}'. Allowed: {', '.join(INTERVAL_MS)}")

        limit_per_call = min(int(limit_per_call), 1000)

        start_ms = self._datetime_to_ms(start_dt)
        now_ms = self._datetime_to_ms(datetime.now(timezone.utc))
        end_ms = min(self._datetime_to_ms(end_dt) if end_dt else now_ms, now_ms)

        if start_ms >= end_ms:
            return []

        step_ms = INTERVAL_MS[interval]
        out: List[RatePoint] = []

        cursor = start_ms
        while cursor < end_ms:
            chunk_end = min(cursor + step_ms * limit_per_call, end_ms)
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': cursor,
                'endTime': chunk_end - 1,
                'limit': limit_per_call,
            }
            rows = self._request('/api/v3/klines', params)

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

    def get_rates_and_save(
        self,
        start_dt: datetime,
        interval: BinanceInterval,
        symbol: str,
        filepath: str,
        end_dt: Optional[datetime] = None,
    ) -> None:

        points = self.get_rates(start_dt, interval, symbol, end_dt=end_dt)
        if not points:
            return

        self._ensure_dir(filepath)

        existing = self._load_existing_csv(filepath)

        for p in points:
            ts_ms = self._datetime_to_ms(p.ts)
            ts_iso = self._to_iso_utc(p.ts)
            existing[ts_ms] = (ts_iso, float(p.close))

        ordered = sorted(existing.items(), key=lambda kv: kv[0])
        rows: list[tuple[int, str, float]] = [(ts_ms, ts_iso, close) for ts_ms, (ts_iso, close) in ordered]

        self._atomic_write_csv(filepath, rows)

