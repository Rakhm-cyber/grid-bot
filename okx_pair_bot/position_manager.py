"""
PositionManager — управление позициями через OKX.

Отвечает за:
- Открытие/закрытие рыночных ордеров (исключительно market orders)
- Кросс-маржинальные позиции
- Синхронизацию с OKX: считывание реальных цен входа, объёмов, PnL
- Отслеживание открытых позиций
- Трейлинг-стоп (опционально)
- Graceful shutdown / восстановление после перезапуска
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from strategy_engine import GridLevel, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Открытая позиция пары (один «слот» уровня)."""
    id: str                        # уникальный ID (генерируется локально)
    symbol_long: str               # инструмент в лонг
    symbol_short: str              # инструмент в шорт
    direction: SignalDirection
    level: Optional[GridLevel]     # уровень сетки (может быть None при восстановлении)

    # Фактические данные с OKX
    entry_spread: float            # спред в момент входа
    entry_price_long: float        # фактическая цена входа лонга
    entry_price_short: float       # фактическая цена входа шорта
    qty_long: float                # кол-во контрактов лонга
    qty_short: float               # кол-во контрактов шорта
    notional_usdt: float           # номинал в USDT

    # Трейлинг-стоп
    max_favorable_spread: float = 0.0  # для long: макс. спред; для short: мин. спред

    entry_time: float = field(default_factory=time.time)
    partial_closed_fractions: list = field(default_factory=list)  # уже закрытые части

    @property
    def age_hours(self) -> float:
        return (time.time() - self.entry_time) / 3600.0

    def update_trailing(self, current_spread: float):
        """Обновляет трейлинг-уровень."""
        if self.direction == SignalDirection.LONG_SPREAD:
            self.max_favorable_spread = max(self.max_favorable_spread, current_spread)
        else:
            if self.max_favorable_spread == 0.0:
                self.max_favorable_spread = current_spread
            self.max_favorable_spread = min(self.max_favorable_spread, current_spread)


class PositionManager:
    """
    Управляет открытием/закрытием позиций через OKX market orders.
    Все позиции — кросс-маржинальные.
    """

    def __init__(self, exchange, cfg: dict, dry_run: bool = True):
        self.exchange = exchange
        self.cfg = cfg
        self.dry_run = dry_run

        self.base_sym = cfg["symbols"]["base"]   # например ARBUSDT
        self.quote_sym = cfg["symbols"]["quote"]  # например ATOMUSDT

        self.trailing_stop = cfg.get("trailing_stop", False)
        self.trailing_pct = float(cfg.get("trailing_stop_pct", 0.02))
        self.time_stop = cfg.get("time_stop_enabled", False)
        self.max_hold_hours = float(cfg.get("max_hold_hours", 48))

        self._positions: dict[str, OpenPosition] = {}  # id -> OpenPosition
        self._pos_counter = 0

    # ──────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ──────────────────────────────────────────────────────────────────────

    def open_position(
        self,
        direction: SignalDirection,
        level: GridLevel,
        notional_usdt: float,
        current_spread: float,
    ) -> Optional[OpenPosition]:
        """
        Открывает парную позицию рыночными ордерами.
        LONG spread: long base + short quote.
        SHORT spread: short base + long quote.

        Возвращает объект позиции с реальными данными с OKX.
        """
        logger.info(
            "[PositionManager] Открытие %s на уровне %.6f, номинал=%.2f USDT",
            direction.value, level.spread_value, notional_usdt,
        )

        if direction == SignalDirection.LONG_SPREAD:
            long_sym = self.base_sym
            short_sym = self.quote_sym
        else:
            long_sym = self.quote_sym
            short_sym = self.base_sym

        # Получаем текущие цены для расчёта объёма
        price_long = self._get_market_price(long_sym, side="buy")
        price_short = self._get_market_price(short_sym, side="sell")

        if price_long is None or price_short is None:
            logger.error("Не удалось получить цены для открытия позиции")
            return None

        qty_long = notional_usdt / price_long
        qty_short = notional_usdt / price_short

        # Минимальные объёмы (получаем из маркет-инфо)
        qty_long = self._round_qty(long_sym, qty_long)
        qty_short = self._round_qty(short_sym, qty_short)

        if qty_long <= 0 or qty_short <= 0:
            logger.error(
                "Объём слишком мал после округления: long=%.6f short=%.6f", qty_long, qty_short
            )
            return None

        # Отправляем рыночные ордера
        long_order = self._place_market_order(long_sym, "buy", qty_long)
        short_order = self._place_market_order(short_sym, "sell", qty_short)

        if long_order is None or short_order is None:
            logger.error("Ошибка размещения одного из ордеров при открытии позиции")
            # Попытка откатить уже исполненный ордер
            if long_order is not None:
                self._place_market_order(long_sym, "sell", qty_long)
            if short_order is not None:
                self._place_market_order(short_sym, "buy", qty_short)
            return None

        # Считываем фактические данные с OKX
        actual_long = self._sync_position(long_sym)
        actual_short = self._sync_position(short_sym)

        entry_price_long = (actual_long.get("entryPrice") or price_long) if actual_long else price_long
        entry_price_short = (actual_short.get("entryPrice") or price_short) if actual_short else price_short

        self._pos_counter += 1
        pos_id = f"pos_{self._pos_counter}_{int(time.time())}"

        pos = OpenPosition(
            id=pos_id,
            symbol_long=long_sym,
            symbol_short=short_sym,
            direction=direction,
            level=level,
            entry_spread=current_spread,
            entry_price_long=float(entry_price_long),
            entry_price_short=float(entry_price_short),
            qty_long=qty_long,
            qty_short=qty_short,
            notional_usdt=notional_usdt,
            max_favorable_spread=current_spread,
        )

        self._positions[pos_id] = pos
        logger.info(
            "[PositionManager] Позиция %s открыта: long %s @ %.4f, short %s @ %.4f",
            pos_id, long_sym, entry_price_long, short_sym, entry_price_short,
        )
        return pos

    def close_position(
        self,
        pos: OpenPosition,
        reason: str = "tp",
        fraction: float = 1.0,
    ) -> bool:
        """
        Закрывает позицию (полностью или частично) рыночными ордерами.
        fraction: доля для закрытия (0..1).
        """
        qty_long = round(pos.qty_long * fraction, 8)
        qty_short = round(pos.qty_short * fraction, 8)

        qty_long = self._round_qty(pos.symbol_long, qty_long)
        qty_short = self._round_qty(pos.symbol_short, qty_short)

        logger.info(
            "[PositionManager] Закрытие %s позиции %s (причина: %s, доля: %.0f%%)",
            pos.direction.value, pos.id, reason, fraction * 100,
        )

        # Закрытие лонга: продаём
        sell_long = self._place_market_order(pos.symbol_long, "sell", qty_long)
        # Закрытие шорта: покупаем
        buy_short = self._place_market_order(pos.symbol_short, "buy", qty_short)

        if sell_long is None or buy_short is None:
            logger.error("Ошибка закрытия позиции %s", pos.id)
            return False

        if fraction >= 1.0:
            del self._positions[pos.id]
            logger.info("[PositionManager] Позиция %s полностью закрыта (%s)", pos.id, reason)
        else:
            # Частичное закрытие: уменьшаем объём
            pos.qty_long -= qty_long
            pos.qty_short -= qty_short
            pos.notional_usdt *= (1 - fraction)
            pos.partial_closed_fractions.append(fraction)
            logger.info(
                "[PositionManager] Позиция %s частично закрыта (%.0f%%). Остаток long=%.6f short=%.6f",
                pos.id, fraction * 100, pos.qty_long, pos.qty_short,
            )

        return True

    def close_all_positions(self, reason: str = "force") -> int:
        """Закрывает все открытые позиции. Возвращает количество закрытых."""
        closed = 0
        for pos_id in list(self._positions.keys()):
            pos = self._positions.get(pos_id)
            if pos:
                if self.close_position(pos, reason=reason):
                    closed += 1
        return closed

    def close_positions_by_direction(
        self, direction: SignalDirection, reason: str = "stop"
    ) -> int:
        """Закрывает все позиции заданного направления."""
        closed = 0
        for pos_id in list(self._positions.keys()):
            pos = self._positions.get(pos_id)
            if pos and pos.direction == direction:
                if self.close_position(pos, reason=reason):
                    closed += 1
        return closed

    def close_at_breakeven(self, max_loss_usdt: float) -> tuple[int, float]:
        """
        Закрывает позиции стремясь к безубытку.
        Используется на 7-й день.
        Возвращает (закрыто_позиций, итоговый_PnL_USDT).
        """
        total_pnl = 0.0
        closed = 0

        for pos_id in list(self._positions.keys()):
            pos = self._positions.get(pos_id)
            if not pos:
                continue

            pnl = self.get_position_pnl_usdt(pos)
            logger.info(
                "[PositionManager] Безубыток: позиция %s PnL=%.2f USDT",
                pos.id, pnl,
            )

            # Закрываем по рынку независимо от PnL
            if self.close_position(pos, reason="breakeven"):
                total_pnl += pnl
                closed += 1

        return closed, total_pnl

    def check_trailing_stops(self, current_spread: float) -> list[OpenPosition]:
        """
        Проверяет трейлинг-стопы для всех позиций.
        Возвращает список позиций для закрытия.
        """
        if not self.trailing_stop:
            return []

        to_close = []
        for pos in self._positions.values():
            pos.update_trailing(current_spread)
            if self._trailing_triggered(pos, current_spread):
                to_close.append(pos)
        return to_close

    def check_time_stops(self) -> list[OpenPosition]:
        """
        Проверяет тайм-стоп для всех позиций.
        Возвращает список позиций для принудительного закрытия.
        """
        if not self.time_stop:
            return []

        to_close = []
        for pos in self._positions.values():
            if pos.age_hours >= self.max_hold_hours:
                logger.info(
                    "[PositionManager] Тайм-стоп: позиция %s держится %.1f ч (лимит %.1f ч)",
                    pos.id, pos.age_hours, self.max_hold_hours,
                )
                to_close.append(pos)
        return to_close

    def recover_positions_from_exchange(self) -> list[OpenPosition]:
        """
        При перезапуске считывает открытые позиции с OKX.
        Восстанавливает внутреннее состояние.
        """
        logger.info("[PositionManager] Восстановление позиций с OKX...")
        recovered = []

        for sym in [self.base_sym, self.quote_sym]:
            pos_data = self._sync_position(sym)
            if pos_data and abs(float(pos_data.get("contracts", 0) or 0)) > 0:
                side_str = pos_data.get("side", "")
                contracts = float(pos_data.get("contracts") or 0)
                entry_price = float(pos_data.get("entryPrice") or 0)

                logger.info(
                    "  Найдена позиция: %s %s %.4f контр. @ %.4f",
                    sym, side_str, contracts, entry_price,
                )
                # Создаём «фантомную» позицию для мониторинга
                # Направление будет уточнено по обоим символам вместе
                # Здесь храним как отдельные записи (упрощение)
                recovered.append(pos_data)

        # Упрощение: если есть open позиции на обоих символах — создаём OpenPosition
        if len(recovered) == 2:
            pos_base = recovered[0] if recovered[0].get("symbol") == self.base_sym else recovered[1]
            pos_quote = recovered[1] if pos_base == recovered[0] else recovered[0]

            base_side = pos_base.get("side", "long")
            direction = (
                SignalDirection.LONG_SPREAD if base_side == "long"
                else SignalDirection.SHORT_SPREAD
            )

            sym_long = self.base_sym if base_side == "long" else self.quote_sym
            sym_short = self.quote_sym if base_side == "long" else self.base_sym

            self._pos_counter += 1
            pos_id = f"recovered_{self._pos_counter}"
            pos = OpenPosition(
                id=pos_id,
                symbol_long=sym_long,
                symbol_short=sym_short,
                direction=direction,
                level=None,
                entry_spread=0.0,  # неизвестен
                entry_price_long=float(pos_base.get("entryPrice") or 0),
                entry_price_short=float(pos_quote.get("entryPrice") or 0),
                qty_long=float(pos_base.get("contracts") or 0),
                qty_short=float(pos_quote.get("contracts") or 0),
                notional_usdt=0.0,
            )
            self._positions[pos_id] = pos
            logger.info("[PositionManager] Восстановлена позиция %s", pos_id)
            return [pos]

        return []

    def get_position_pnl_usdt(self, pos: OpenPosition) -> float:
        """Считывает PnL позиции с OKX (unrealized PnL обоих legs)."""
        pnl = 0.0
        for sym in [pos.symbol_long, pos.symbol_short]:
            data = self._sync_position(sym)
            if data:
                pnl += float(data.get("unrealizedPnl") or 0.0)
        return pnl

    def get_total_pnl_usdt(self) -> float:
        """Суммарный unrealized PnL всех открытых позиций."""
        total = 0.0
        for pos in self._positions.values():
            total += self.get_position_pnl_usdt(pos)
        return total

    def get_total_exposure_usdt(self) -> float:
        """Суммарная экспозиция (сумма номиналов) всех позиций."""
        return sum(p.notional_usdt for p in self._positions.values())

    @property
    def open_positions(self) -> dict[str, OpenPosition]:
        return self._positions

    def count_by_direction(self, direction: SignalDirection) -> int:
        return sum(1 for p in self._positions.values() if p.direction == direction)

    # ──────────────────────────────────────────────────────────────────────
    # Приватные методы
    # ──────────────────────────────────────────────────────────────────────

    def _place_market_order(
        self, symbol: str, side: str, qty: float
    ) -> Optional[dict]:
        """
        Размещает рыночный ордер. В dry_run режиме имитирует исполнение.
        """
        if self.dry_run:
            logger.info("[DRY RUN] %s %s %s qty=%.6f", "Ордер", side.upper(), symbol, qty)
            return {"id": f"dry_{int(time.time())}", "status": "closed", "average": 0}

        for attempt in range(5):
            try:
                # Устанавливаем кросс-маржу перед ордером
                self._ensure_cross_margin(symbol)

                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=qty,
                    params={"tdMode": "cross"},  # кросс-маржа OKX
                )
                logger.info(
                    "[PositionManager] Ордер исполнен: %s %s %s qty=%.6f цена=%.4f",
                    order.get("id", "?"),
                    side.upper(),
                    symbol,
                    qty,
                    order.get("average") or order.get("price") or 0,
                )
                return order

            except Exception as exc:
                err_str = str(exc)
                if "rate limit" in err_str.lower():
                    wait = 60
                    logger.warning("Rate limit — ждём %ds", wait)
                    time.sleep(wait)
                elif attempt < 4:
                    wait = 2 ** attempt
                    logger.warning(
                        "Попытка %d/5 ошибка ордера %s %s: %s. Повтор через %ds",
                        attempt + 1, side, symbol, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Ошибка размещения ордера %s %s: %s", side, symbol, exc)
                    return None
        return None

    def _sync_position(self, symbol: str) -> Optional[dict]:
        """Получает состояние позиции с OKX."""
        if self.dry_run:
            return None
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos.get("symbol") == symbol and abs(float(pos.get("contracts") or 0)) > 0:
                    return pos
        except Exception as exc:
            logger.warning("Ошибка fetch_positions(%s): %s", symbol, exc)
        return None

    def _get_market_price(self, symbol: str, side: str) -> Optional[float]:
        """Получает текущую рыночную цену (bid для sell, ask для buy)."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            if side == "buy":
                return float(ticker.get("ask") or ticker.get("last") or 0)
            else:
                return float(ticker.get("bid") or ticker.get("last") or 0)
        except Exception as exc:
            logger.warning("Ошибка получения цены %s: %s", symbol, exc)
            return None

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Округляет объём до минимального лота инструмента."""
        if self.dry_run:
            return round(qty, 4)
        try:
            market = self.exchange.market(symbol)
            precision = market.get("precision", {}).get("amount", 4)
            step = market.get("limits", {}).get("amount", {}).get("min", 0)
            if step and step > 0:
                qty = math.floor(qty / step) * step
            return round(qty, int(precision) if isinstance(precision, int) else 4)
        except Exception:
            return round(qty, 4)

    def _ensure_cross_margin(self, symbol: str):
        """Устанавливает кросс-маржу для инструмента."""
        try:
            self.exchange.set_margin_mode("cross", symbol)
        except Exception as exc:
            # Может вернуть ошибку если маржа уже установлена
            logger.debug("set_margin_mode(%s): %s", symbol, exc)

    def _trailing_triggered(self, pos: OpenPosition, current_spread: float) -> bool:
        """Проверяет, сработал ли трейлинг-стоп."""
        if pos.direction == SignalDirection.LONG_SPREAD:
            # Откат от максимума
            if pos.max_favorable_spread > pos.entry_spread:
                drawback = pos.max_favorable_spread - current_spread
                threshold = pos.max_favorable_spread * self.trailing_pct
                return drawback >= threshold
        else:
            # Откат от минимума (для short spread)
            if pos.max_favorable_spread < pos.entry_spread and pos.max_favorable_spread > 0:
                drawback = current_spread - pos.max_favorable_spread
                threshold = abs(pos.max_favorable_spread) * self.trailing_pct
                return drawback >= threshold
        return False
