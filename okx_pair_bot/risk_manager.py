"""
RiskManager — модуль управления рисками.

Отвечает за:
- Контроль максимальной суммарной экспозиции (max_leverage)
- VaR-лимит (опционально)
- Совокупный риск (потенциальный убыток по всем позициям)
- Ограничение количества позиций
- Проверку возможности открытия новой позиции
"""

from __future__ import annotations

import logging
import math

from position_manager import OpenPosition, PositionManager
from strategy_engine import EntrySignal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Проверяет допустимость новых входов с точки зрения риска.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.capital = float(cfg.get("capital", 1000.0))
        self.max_leverage = float(cfg.get("max_leverage", 2.0))
        self.max_loss_fraction = float(cfg.get("max_loss_fraction", 0.05))
        self.stop_multiplier = float(cfg.get("stop_loss_multiplier", 3.5))
        self.var_based = cfg.get("var_based_leverage", False)
        self.max_positions_per_side = int(cfg.get("max_positions_per_side", 3))
        self.num_levels = (
            int(cfg.get("num_levels_up", 5)) + int(cfg.get("num_levels_down", 5))
        )

    def check_entry_allowed(
        self,
        signal: EntrySignal,
        pm: PositionManager,
    ) -> tuple[bool, str]:
        """
        Проверяет, можно ли открыть новую позицию.
        Возвращает (allowed: bool, reason: str).
        """
        # 1. Лимит на общее количество позиций
        total_positions = len(pm.open_positions)
        if total_positions >= self.num_levels:
            return False, f"Лимит позиций ({total_positions}/{self.num_levels})"

        # 2. Лимит по стороне
        side_count = pm.count_by_direction(signal.direction)
        if side_count >= self.max_positions_per_side:
            return False, (
                f"Лимит позиций по стороне {signal.direction.value}: "
                f"{side_count}/{self.max_positions_per_side}"
            )

        # 3. Максимальная суммарная экспозиция
        current_exposure = pm.get_total_exposure_usdt()
        new_exposure = current_exposure + signal.position_size_usdt
        max_exposure = self.capital * self.max_leverage

        if self.var_based:
            max_exposure = self._var_leverage() * self.capital

        if new_exposure > max_exposure:
            return False, (
                f"Превышение макс. экспозиции: {new_exposure:.0f} > {max_exposure:.0f} USDT"
            )

        # 4. Совокупный риск (потенциальный убыток по стопу)
        current_risk = self._calc_current_risk(pm, signal.sigma)
        new_risk_per_pos = signal.position_size_usdt * self.stop_multiplier * signal.sigma
        total_risk = current_risk + new_risk_per_pos
        max_risk = self.capital * self.max_loss_fraction

        if total_risk > max_risk:
            return False, (
                f"Превышение лимита риска: {total_risk:.2f} > {max_risk:.2f} USDT"
            )

        return True, "OK"

    def get_effective_max_leverage(self, sigma: float) -> float:
        """
        Возвращает эффективное максимальное плечо.
        При var_based — рассчитывается из допустимого убытка.
        """
        if self.var_based:
            return self._var_leverage(sigma)
        return self.max_leverage

    def log_risk_state(self, pm: PositionManager, sigma: float):
        """Логирует текущее состояние рисков."""
        exposure = pm.get_total_exposure_usdt()
        risk = self._calc_current_risk(pm, sigma)
        max_exp = self.capital * self.get_effective_max_leverage(sigma)
        max_risk = self.capital * self.max_loss_fraction

        logger.info(
            "[RiskManager] Экспозиция: %.0f/%.0f USDT (%.1f%%) | "
            "Риск: %.2f/%.2f USDT (%.1f%%) | Позиций: %d",
            exposure, max_exp, exposure / max(max_exp, 1) * 100,
            risk, max_risk, risk / max(max_risk, 1) * 100,
            len(pm.open_positions),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Приватные методы
    # ──────────────────────────────────────────────────────────────────────

    def _var_leverage(self, sigma: float = None) -> float:
        """
        Рассчитывает максимальное плечо через VaR:
        max_leverage = max_loss_fraction / (stop_multiplier * σ)
        """
        if sigma is None or sigma <= 0:
            return self.max_leverage
        var_lev = self.max_loss_fraction / (self.stop_multiplier * sigma)
        return min(var_lev, self.max_leverage)

    def _calc_current_risk(self, pm: PositionManager, sigma: float) -> float:
        """
        Суммарный потенциальный убыток при движении всех позиций до стопа.
        Приблизительный расчёт: notional * stop_sigma_moves * sigma.
        """
        total_risk = 0.0
        for pos in pm.open_positions.values():
            # Расстояние до стопа в единицах спреда
            # Упрощение: предполагаем, что позиция открыта на уровне 1σ
            risk_per_pos = pos.notional_usdt * self.stop_multiplier * sigma
            total_risk += risk_per_pos
        return total_risk
