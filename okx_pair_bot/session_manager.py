"""
SessionManager — управление торговой сессией.

Отвечает за:
- Отслеживание времени сессии (макс. 7 дней)
- Проверку целевой прибыли на 5-й день
- Закрытие по безубытку на 7-й день
- Досрочное завершение по сигналу пользователя
- Формирование итогового отчёта
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

DAY_SECONDS = 86400.0


class SessionStatus(Enum):
    RUNNING = "running"
    TARGET_PROFIT_REACHED = "target_profit_reached"
    TIMEOUT_7_DAYS = "timeout_7_days"
    MANUAL_STOP = "manual_stop"
    ERROR = "error"


@dataclass
class SessionStats:
    """Статистика сессии."""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    initial_capital: float = 0.0
    final_pnl_usdt: float = 0.0
    realized_pnl_usdt: float = 0.0
    unrealized_pnl_usdt: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    max_drawdown_usdt: float = 0.0
    peak_equity: float = 0.0
    status: SessionStatus = SessionStatus.RUNNING

    @property
    def duration_hours(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return (end - self.start_time) / 3600.0

    @property
    def pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return self.final_pnl_usdt / self.initial_capital * 100.0

    @property
    def win_rate(self) -> float:
        total = self.num_wins + self.num_losses
        if total == 0:
            return 0.0
        return self.num_wins / total * 100.0


class SessionManager:
    """
    Управляет жизненным циклом торговой сессии.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.capital = float(cfg.get("capital", 1000.0))
        self.target_pnl_pct = float(cfg.get("target_profit_pct", 5.0))
        self.max_exit_loss_pct = float(cfg.get("max_exit_loss_pct", 1.0))

        self.stats = SessionStats(
            initial_capital=self.capital,
            peak_equity=self.capital,
        )
        self._equity_curve: list[tuple[float, float]] = []  # (timestamp, equity)
        self._day5_checked = False
        self._stop_flag = False

    # ──────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ──────────────────────────────────────────────────────────────────────

    def check_session_state(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> tuple[bool, str]:
        """
        Проверяет, нужно ли завершить сессию.
        Возвращает (should_stop: bool, reason: str).

        Вызывать каждый цикл (или раз в N минут).
        """
        if self._stop_flag:
            return True, "manual_stop"

        elapsed_days = self._elapsed_days()
        total_pnl = realized_pnl + unrealized_pnl

        # Обновляем статистику
        self.stats.realized_pnl_usdt = realized_pnl
        self.stats.unrealized_pnl_usdt = unrealized_pnl
        self.stats.final_pnl_usdt = total_pnl

        current_equity = self.capital + total_pnl
        self._equity_curve.append((time.time(), current_equity))

        # Просадка
        if current_equity > self.stats.peak_equity:
            self.stats.peak_equity = current_equity
        drawdown = self.stats.peak_equity - current_equity
        if drawdown > self.stats.max_drawdown_usdt:
            self.stats.max_drawdown_usdt = drawdown

        # Проверка на 5-й день
        if elapsed_days >= 5 and not self._day5_checked:
            self._day5_checked = True
            target_pnl = self.capital * (self.target_pnl_pct / 100.0)
            if total_pnl >= target_pnl:
                logger.info(
                    "[SessionManager] День 5: целевая прибыль достигнута! "
                    "PnL=%.2f USDT (цель=%.2f USDT)",
                    total_pnl, target_pnl,
                )
                self.stats.status = SessionStatus.TARGET_PROFIT_REACHED
                return True, "target_profit_reached"
            else:
                logger.info(
                    "[SessionManager] День 5: прибыль %.2f USDT не достигла цели %.2f USDT — продолжаем",
                    total_pnl, target_pnl,
                )

        # Проверка на 7-й день
        if elapsed_days >= 7:
            logger.info(
                "[SessionManager] День 7: истёк срок сессии. "
                "PnL=%.2f USDT (%.2f%%)",
                total_pnl, self._pct(total_pnl),
            )
            self.stats.status = SessionStatus.TIMEOUT_7_DAYS
            return True, "timeout_7_days"

        return False, ""

    def handle_7day_close(self, pm) -> tuple[int, float]:
        """
        Закрывает все позиции по безубытку на 7-й день.
        Возвращает (закрыто_позиций, итоговый_PnL).
        """
        max_loss = self.capital * (self.max_exit_loss_pct / 100.0)
        logger.info(
            "[SessionManager] Закрытие по безубытку. Допустимый убыток: %.2f USDT",
            max_loss,
        )
        closed, total_pnl = pm.close_at_breakeven(max_loss)

        if total_pnl < -max_loss:
            logger.error(
                "[SessionManager] ВНИМАНИЕ: убыток при закрытии %.2f USDT превышает лимит %.2f USDT",
                abs(total_pnl), max_loss,
            )
        else:
            logger.info(
                "[SessionManager] Закрытие завершено: %d позиций, PnL=%.2f USDT",
                closed, total_pnl,
            )
        return closed, total_pnl

    def record_trade(self, pnl_usdt: float):
        """Записывает результат закрытой сделки."""
        self.stats.num_trades += 1
        if pnl_usdt >= 0:
            self.stats.num_wins += 1
        else:
            self.stats.num_losses += 1

    def request_stop(self):
        """Запрашивает досрочную остановку бота (из обработчика сигнала)."""
        logger.info("[SessionManager] Запрошена досрочная остановка")
        self._stop_flag = True
        self.stats.status = SessionStatus.MANUAL_STOP

    def finalize(self):
        """Фиксирует время завершения сессии."""
        self.stats.end_time = time.time()

    def generate_report(self) -> str:
        """Генерирует текстовый отчёт по сессии."""
        s = self.stats
        lines = [
            "=" * 60,
            "       ОТЧЁТ ПО ТОРГОВОЙ СЕССИИ",
            "=" * 60,
            f"  Статус:          {s.status.value}",
            f"  Длительность:    {s.duration_hours:.1f} ч ({s.duration_hours/24:.1f} дн.)",
            f"  Начальный кап.:  {s.initial_capital:.2f} USDT",
            f"  Итоговый PnL:    {s.final_pnl_usdt:+.2f} USDT ({s.pnl_pct:+.2f}%)",
            f"    Реализованный: {s.realized_pnl_usdt:+.2f} USDT",
            f"    Нереализованный:{s.unrealized_pnl_usdt:+.2f} USDT",
            f"  Макс. просадка:  {s.max_drawdown_usdt:.2f} USDT",
            f"  Сделок всего:    {s.num_trades}",
            f"  Прибыльных:      {s.num_wins}",
            f"  Убыточных:       {s.num_losses}",
            f"  Win rate:        {s.win_rate:.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)

    def get_equity_curve(self) -> list[tuple[float, float]]:
        """Возвращает кривую капитала [(timestamp, equity), ...]."""
        return list(self._equity_curve)

    def elapsed_days(self) -> float:
        return self._elapsed_days()

    # ──────────────────────────────────────────────────────────────────────
    # Приватные методы
    # ──────────────────────────────────────────────────────────────────────

    def _elapsed_days(self) -> float:
        return (time.time() - self.stats.start_time) / DAY_SECONDS

    def _pct(self, pnl: float) -> float:
        if self.capital <= 0:
            return 0.0
        return pnl / self.capital * 100.0
