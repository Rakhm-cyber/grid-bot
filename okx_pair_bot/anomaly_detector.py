"""
AnomalyDetector — детектор аномальных рыночных условий.

Триггеры:
1. Аномальный объём (spike)
2. Падение ликвидности (широкий bid-ask)
3. Рассинхронизация контрактов (only-reduce mode)
4. Резкий рост волатильности

При срабатывании блокирует новые входы.
Автоматически снимает блокировку через auto_resume_minutes после повторной проверки.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from data_collector import DataCollector, SpreadSnapshot

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Событие аномалии."""
    timestamp: float
    trigger: str       # название триггера
    details: str       # описание


class AnomalyDetector:
    """
    Отслеживает аномальные рыночные условия и блокирует новые входы.
    """

    def __init__(self, exchange, collector: DataCollector, cfg: dict):
        self.exchange = exchange
        self.collector = collector

        anom_cfg = cfg.get("anomaly_detector", {})
        self.volume_lookback_h = int(anom_cfg.get("volume_lookback_hours", 24))
        self.volume_spike_mul = float(anom_cfg.get("volume_spike_multiplier", 10.0))
        self.max_spread_pct = float(anom_cfg.get("max_allowed_spread_pct", 0.5))
        self.vol_spike_thresh = float(anom_cfg.get("volatility_spike_threshold", 3.0))
        self.auto_resume_min = float(anom_cfg.get("auto_resume_minutes", 60))

        self.base_sym = cfg["symbols"]["base"]
        self.quote_sym = cfg["symbols"]["quote"]

        self._blocked: bool = False
        self._block_time: float = 0.0
        self._block_reason: str = ""
        self._prev_sigma: float = 0.0
        self._events: list[AnomalyEvent] = []

    # ──────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ──────────────────────────────────────────────────────────────────────

    def check(self, snap: SpreadSnapshot) -> bool:
        """
        Выполняет все проверки аномалий.
        Возвращает True если торговля заблокирована.
        """
        # Проверяем автоматическое снятие блокировки
        if self._blocked:
            elapsed = (time.time() - self._block_time) / 60.0
            if elapsed >= self.auto_resume_min:
                logger.info(
                    "[AnomalyDetector] Прошло %.0f мин после блокировки — повторная проверка",
                    elapsed,
                )
                self._blocked = False  # сбрасываем для повторной проверки

        # Проверяем все триггеры
        triggers = []

        t = self._check_volume(snap)
        if t:
            triggers.append(t)

        t = self._check_liquidity(snap)
        if t:
            triggers.append(t)

        t = self._check_contract_status()
        if t:
            triggers.append(t)

        t = self._check_volatility_spike(snap)
        if t:
            triggers.append(t)

        if triggers:
            reason = "; ".join(triggers)
            if not self._blocked:
                logger.warning("[AnomalyDetector] БЛОКИРОВКА: %s", reason)
                self._events.append(AnomalyEvent(
                    timestamp=time.time(),
                    trigger="multi" if len(triggers) > 1 else triggers[0],
                    details=reason,
                ))
            self._blocked = True
            self._block_time = time.time()
            self._block_reason = reason
        else:
            if self._blocked:
                logger.info("[AnomalyDetector] Аномалии устранены, блокировка снята")
            self._blocked = False

        # Обновляем предыдущую σ
        self._prev_sigma = snap.sigma

        return self._blocked

    @property
    def is_blocked(self) -> bool:
        return self._blocked

    @property
    def block_reason(self) -> str:
        return self._block_reason

    @property
    def events(self) -> list[AnomalyEvent]:
        return list(self._events)

    # ──────────────────────────────────────────────────────────────────────
    # Приватные проверки
    # ──────────────────────────────────────────────────────────────────────

    def _check_volume(self, snap: SpreadSnapshot) -> Optional[str]:
        """Проверяет аномальный объём по обоим инструментам."""
        for sym, cur_vol in [
            (self.base_sym, snap.volume_base),
            (self.quote_sym, snap.volume_quote),
        ]:
            if cur_vol <= 0:
                continue
            _, avg_vol = self.collector.get_candle_volume(sym, self.volume_lookback_h)
            if avg_vol > 0 and cur_vol > self.volume_spike_mul * avg_vol:
                msg = (
                    f"Объём {sym}: {cur_vol:.0f} > {self.volume_spike_mul}x "
                    f"среднего ({avg_vol:.0f})"
                )
                logger.warning("[AnomalyDetector] %s", msg)
                return f"volume_spike:{sym}"
        return None

    def _check_liquidity(self, snap: SpreadSnapshot) -> Optional[str]:
        """Проверяет ширину bid-ask спреда."""
        for sym, bid, ask in [
            (self.base_sym, snap.bid_base, snap.ask_base),
            (self.quote_sym, snap.bid_quote, snap.ask_quote),
        ]:
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100
            if spread_pct > self.max_spread_pct:
                msg = f"Низкая ликвидность {sym}: bid-ask={spread_pct:.3f}% > {self.max_spread_pct}%"
                logger.warning("[AnomalyDetector] %s", msg)
                return f"low_liquidity:{sym}"
        return None

    def _check_contract_status(self) -> Optional[str]:
        """Проверяет, не переведён ли контракт в режим 'только закрытие'."""
        for sym in [self.base_sym, self.quote_sym]:
            try:
                market = self.exchange.market(sym)
                if not market:
                    continue
                # OKX может выставить поле 'active' = False или специальный статус
                if not market.get("active", True):
                    msg = f"Контракт {sym} неактивен"
                    logger.warning("[AnomalyDetector] %s", msg)
                    return f"contract_inactive:{sym}"
            except Exception as exc:
                logger.debug("Ошибка проверки статуса контракта %s: %s", sym, exc)
        return None

    def _check_volatility_spike(self, snap: SpreadSnapshot) -> Optional[str]:
        """Проверяет резкий рост волатильности за последний час."""
        if self._prev_sigma <= 0 or snap.sigma <= 0:
            return None
        ratio = snap.sigma / self._prev_sigma
        if ratio > self.vol_spike_thresh:
            msg = f"Всплеск волатильности: σ выросла в {ratio:.1f}x за последний период"
            logger.warning("[AnomalyDetector] %s", msg)
            return f"volatility_spike:{ratio:.1f}x"
        return None
