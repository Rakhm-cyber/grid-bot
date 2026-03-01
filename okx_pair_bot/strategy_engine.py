"""
StrategyEngine — модуль построения и управления сеткой уровней.

Отвечает за:
- Построение уровней сетки вокруг EMA (uniform / fibonacci)
- Адаптивный множитель шага (volatility-of-volatility)
- Определение сигналов входа (касание уровня, Z-Score)
- Расчёт размера позиции
- Пересчёт сетки при изменении EMA
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from data_collector import SpreadSnapshot

logger = logging.getLogger(__name__)


# Коэффициенты Фибоначчи для распределения уровней
FIBONACCI_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618]


class SignalDirection(Enum):
    LONG_SPREAD = "long_spread"    # long base / short quote (спред ниже EMA)
    SHORT_SPREAD = "short_spread"  # short base / long quote (спред выше EMA)
    NONE = "none"


@dataclass
class GridLevel:
    """Один уровень сетки."""
    index: int            # номер уровня (отрицательный = ниже EMA, положительный = выше)
    spread_value: float   # значение спреда на уровне
    direction: SignalDirection
    position_size_usdt: float    # номинал позиции на этом уровне
    is_active: bool = True       # уровень активен (на нём ещё можно открыть позицию)


@dataclass
class GridState:
    """Текущее состояние сетки."""
    ema: float
    sigma: float
    grid_step: float
    levels_up: list[GridLevel] = field(default_factory=list)
    levels_down: list[GridLevel] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def all_levels(self) -> list[GridLevel]:
        return self.levels_down + self.levels_up


@dataclass
class EntrySignal:
    """Сигнал на вход в позицию."""
    direction: SignalDirection
    level: GridLevel
    zscore: float
    position_size_usdt: float
    spread: float
    ema: float
    sigma: float


class StrategyEngine:
    """
    Строит и обновляет сетку уровней, генерирует сигналы входа/выхода.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.num_up = int(cfg.get("num_levels_up", 5))
        self.num_down = int(cfg.get("num_levels_down", 5))
        self.base_multiplier = float(cfg.get("grid_multiplier", 0.5))
        self.distribution = cfg.get("level_distribution", "fibonacci")
        self.adaptive_levels = cfg.get("adaptive_levels_count", True)
        self.stop_multiplier = float(cfg.get("stop_loss_multiplier", 3.5))

        # Позиционирование
        self.sizing_mode = cfg.get("position_sizing", "fixed")
        self.size_fixed = float(cfg.get("position_size_fixed", 200.0))
        self.size_pct = float(cfg.get("position_size_percent", 2.0))
        self.capital = float(cfg.get("capital", 1000.0))
        self.max_per_side = int(cfg.get("max_positions_per_side", 3))

        # Z-Score sizing
        self.zscore_sizing = cfg.get("zscore_sizing", False)
        self.zscore_scale = float(cfg.get("zscore_scale_factor", 1.5))
        self.zscore_max = float(cfg.get("zscore_max_factor", 2.0))

        # Адаптивный шаг
        self._adaptive_cfg = cfg.get("adaptive_step", {})

        self._grid: Optional[GridState] = None
        # Отслеживание позиций по направлениям
        self._long_count = 0
        self._short_count = 0

    # ──────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ──────────────────────────────────────────────────────────────────────

    def build_grid(self, snap: SpreadSnapshot) -> GridState:
        """
        Строит новую сетку на основе текущего снапшота индикаторов.
        """
        import time

        ema = snap.ema
        sigma = snap.sigma

        # Множитель шага
        multiplier = self._get_multiplier(snap)
        grid_step = sigma * multiplier

        if grid_step < 1e-10:
            logger.warning("Шаг сетки слишком мал (%.2e), используем σ", grid_step)
            grid_step = sigma

        levels_up = self._build_levels(ema, grid_step, sigma, direction="up", snap=snap)
        levels_down = self._build_levels(ema, grid_step, sigma, direction="down", snap=snap)

        self._grid = GridState(
            ema=ema,
            sigma=sigma,
            grid_step=grid_step,
            levels_up=levels_up,
            levels_down=levels_down,
            timestamp=time.time(),
        )

        logger.info(
            "Сетка построена: EMA=%.6f σ=%.6f шаг=%.6f уровни_вверх=%d уровни_вниз=%d",
            ema, sigma, grid_step, len(levels_up), len(levels_down),
        )
        return self._grid

    def check_entry_signal(
        self,
        snap: SpreadSnapshot,
        open_long_count: int,
        open_short_count: int,
    ) -> Optional[EntrySignal]:
        """
        Проверяет, достиг ли текущий спред одного из уровней сетки.
        Возвращает EntrySignal или None.
        """
        if self._grid is None:
            return None

        spread = snap.spread
        zscore = snap.zscore
        stop_lim = self.stop_multiplier

        # Не входим вблизи стопа
        if abs(zscore) >= stop_lim:
            logger.debug("Z=%.2f >= стоп=%.2f — вход заблокирован", zscore, stop_lim)
            return None

        # Проверяем уровни сверху (сигнал SHORT spread)
        if zscore > 0:
            for lvl in sorted(self._grid.levels_up, key=lambda l: l.spread_value):
                if not lvl.is_active:
                    continue
                if open_short_count >= self.max_per_side:
                    break
                if spread >= lvl.spread_value:
                    size = self._calc_position_size(zscore, lvl)
                    return EntrySignal(
                        direction=SignalDirection.SHORT_SPREAD,
                        level=lvl,
                        zscore=zscore,
                        position_size_usdt=size,
                        spread=spread,
                        ema=snap.ema,
                        sigma=snap.sigma,
                    )

        # Проверяем уровни снизу (сигнал LONG spread)
        elif zscore < 0:
            for lvl in sorted(self._grid.levels_down, key=lambda l: l.spread_value, reverse=True):
                if not lvl.is_active:
                    continue
                if open_long_count >= self.max_per_side:
                    break
                if spread <= lvl.spread_value:
                    size = self._calc_position_size(zscore, lvl)
                    return EntrySignal(
                        direction=SignalDirection.LONG_SPREAD,
                        level=lvl,
                        zscore=zscore,
                        position_size_usdt=size,
                        spread=spread,
                        ema=snap.ema,
                        sigma=snap.sigma,
                    )

        return None

    def check_tp_signal(
        self,
        entry_spread: float,
        entry_direction: SignalDirection,
        current_spread: float,
        current_ema: float,
    ) -> Optional[float]:
        """
        Проверяет достижение тейк-профита.
        Возвращает долю позиции для закрытия (0..1) или None.

        Поддерживает partial_tp из конфига.
        """
        partial_cfg = self.cfg.get("partial_tp", {})
        if not partial_cfg.get("enabled", False):
            # Стандартный TP: возврат к EMA
            if entry_direction == SignalDirection.LONG_SPREAD:
                if current_spread >= current_ema:
                    return 1.0
            else:
                if current_spread <= current_ema:
                    return 1.0
            return None

        # Partial TP
        dist_total = abs(entry_spread - current_ema)
        if dist_total < 1e-10:
            return None

        dist_covered = abs(current_spread - entry_spread)
        ratio_covered = dist_covered / dist_total

        levels_cfg = partial_cfg.get("levels", [])
        for lvl_cfg in levels_cfg:
            at_ratio = float(lvl_cfg.get("at_ratio", 0.5))
            fraction = float(lvl_cfg.get("fraction", 0.3))
            if ratio_covered >= at_ratio:
                return fraction

        # Остаток — на EMA
        if entry_direction == SignalDirection.LONG_SPREAD:
            if current_spread >= current_ema:
                return 1.0
        else:
            if current_spread <= current_ema:
                return 1.0

        return None

    def check_stop_signal(self, snap: SpreadSnapshot) -> bool:
        """
        Возвращает True если спред достиг уровня стоп-лосса.
        """
        if self._grid is None:
            return False
        stop_level = self._grid.sigma * self.stop_multiplier
        return abs(snap.zscore) >= self.stop_multiplier

    def mark_level_used(self, level: GridLevel):
        """Помечает уровень как использованный (позиция открыта)."""
        level.is_active = False

    def mark_level_free(self, level: GridLevel):
        """Освобождает уровень (позиция закрыта)."""
        level.is_active = True

    def get_tp_price_for_position(
        self,
        entry_spread: float,
        direction: SignalDirection,
        current_ema: float,
    ) -> float:
        """Возвращает значение спреда для тейк-профита (EMA)."""
        return current_ema

    @property
    def grid(self) -> Optional[GridState]:
        return self._grid

    # ──────────────────────────────────────────────────────────────────────
    # Внутренние методы
    # ──────────────────────────────────────────────────────────────────────

    def _get_multiplier(self, snap: SpreadSnapshot) -> float:
        """Возвращает множитель шага (базовый или адаптивный)."""
        if not self._adaptive_cfg.get("enabled", False):
            return self.base_multiplier

        # Адаптивный: нужны short/long σ
        # Здесь используем cached sigma как long и snap.sigma как short-proxy
        # Полноценная реализация требует двух DataCollector с разными периодами
        # Упрощение: используем базовый множитель (адаптивность реализована в DataCollector)
        sigma_short = snap.sigma  # краткосрочная (уже обновлена)
        sigma_long = snap.sigma   # в идеале — другой период

        ratio = sigma_short / max(sigma_long, 1e-10)
        hi_thresh = float(self._adaptive_cfg.get("ratio_threshold_high", 1.2))
        lo_thresh = float(self._adaptive_cfg.get("ratio_threshold_low", 0.8))

        if ratio > hi_thresh:
            return float(self._adaptive_cfg.get("multiplier_high", 1.5))
        elif ratio < lo_thresh:
            return float(self._adaptive_cfg.get("multiplier_low", 0.7))
        return self.base_multiplier

    def _build_levels(
        self,
        ema: float,
        grid_step: float,
        sigma: float,
        direction: str,
        snap: SpreadSnapshot,
    ) -> list[GridLevel]:
        """Строит уровни в одном направлении от EMA."""
        num = self.num_up if direction == "up" else self.num_down
        sign = 1 if direction == "up" else -1
        max_deviation = self.stop_multiplier * sigma

        if self.distribution == "fibonacci":
            ratios = FIBONACCI_RATIOS[:num]
        else:
            ratios = [float(i + 1) for i in range(num)]

        levels = []
        for i, ratio in enumerate(ratios):
            dist = grid_step * ratio
            # Ограничиваем максимальным отклонением
            if dist > max_deviation:
                if self.adaptive_levels:
                    logger.debug(
                        "Уровень %d (дистанция %.4f) выходит за 3σ=%.4f, пропускаем",
                        i + 1, dist, max_deviation,
                    )
                    break
                else:
                    dist = max_deviation

            spread_val = ema + sign * dist
            dir_enum = SignalDirection.SHORT_SPREAD if direction == "up" else SignalDirection.LONG_SPREAD
            size = self._calc_base_size(abs(snap.zscore))

            levels.append(GridLevel(
                index=sign * (i + 1),
                spread_value=spread_val,
                direction=dir_enum,
                position_size_usdt=size,
                is_active=True,
            ))

        return levels

    def _calc_base_size(self, abs_zscore: float = 0.0) -> float:
        """Базовый размер позиции в USDT."""
        if self.sizing_mode == "fixed":
            base = self.size_fixed
        else:
            base = self.capital * (self.size_pct / 100.0)

        if not self.zscore_sizing:
            return base

        # Масштабирование по Z-Score
        if abs_zscore < 1.5:
            return base
        elif abs_zscore < 2.5:
            return base * self.zscore_scale
        else:
            return base * self.zscore_max

    def _calc_position_size(self, zscore: float, level: GridLevel) -> float:
        """Размер позиции с учётом Z-Score."""
        return self._calc_base_size(abs(zscore))
