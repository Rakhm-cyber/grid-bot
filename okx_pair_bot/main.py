"""
main.py — точка входа адаптивного парного грид-бота (OKX, USDT Perpetual).

Запуск:
    python main.py                                   # использует config.yaml (dry_run)
    python main.py --base ARBUSDT --quote ATOMUSDT   # задать пару через CLI
    python main.py --config my_config.yaml --no-dry-run  # боевой режим

Архитектура:
    DataCollector  → SpreadSnapshot (EMA, σ, Z-Score)
    StrategyEngine → уровни сетки, EntrySignal, TP/SL
    PositionManager → market orders (OKX, cross-margin)
    RiskManager    → контроль экспозиции и риска
    AnomalyDetector → блокировка при аномалиях
    SessionManager → контроль сессии 5/7 дней
    BotDatabase    → SQLite логирование
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import time
import uuid
from typing import Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Загрузка конфигурации
# ──────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """
    Загружает YAML-конфиг, подставляет переменные окружения вида ${VAR}.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_env(m):
        var = m.group(1)
        val = os.environ.get(var, "")
        if not val:
            logger.warning("Переменная окружения '%s' не задана (используется пустая строка)", var)
        return val

    content = re.sub(r"\$\{([^}]+)\}", replace_env, content)
    return yaml.safe_load(content)


def create_exchange(cfg: dict):
    """Создаёт и настраивает ccxt.okx."""
    import ccxt

    exchange = ccxt.okx({
        "apiKey": cfg.get("api_key", ""),
        "secret": cfg.get("api_secret", ""),
        "password": cfg.get("passphrase", ""),
        "options": {
            "defaultType": "swap",   # perpetual futures
        },
        "enableRateLimit": True,
    })

    if cfg.get("dry_run", True):
        logger.info("Биржа OKX инициализирована (DRY RUN — реальные ордера отключены)")
    else:
        logger.info("Биржа OKX инициализирована (БОЕВОЙ режим)")

    try:
        exchange.load_markets()
        logger.info("Загружено %d инструментов с OKX", len(exchange.markets))
    except Exception as exc:
        logger.warning("Не удалось загрузить маркеты: %s", exc)
        if not cfg.get("dry_run", True):
            raise

    return exchange


# ──────────────────────────────────────────────────────────────────────────
# Главный класс бота
# ──────────────────────────────────────────────────────────────────────────

class GridBot:
    """
    Парный адаптивный грид-бот.
    Торгует спредом двух коррелированных USDT-perpetual активов на OKX.
    """

    CHECK_INTERVAL_SEC = 5      # пауза между проверками спреда
    EQUITY_LOG_INTERVAL_SEC = 300  # логировать equity раз в 5 мин

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dry_run = cfg.get("dry_run", True)
        self.recalc_interval_sec = int(cfg.get("recalc_interval_minutes", 60)) * 60

        self.session_id = str(uuid.uuid4())[:8]

        # Инициализация биржи
        self.exchange = create_exchange(cfg)

        # Импорт модулей (отложенный, чтобы логирование было настроено до импорта)
        from data_collector import DataCollector
        from strategy_engine import StrategyEngine, SignalDirection
        from position_manager import PositionManager
        from risk_manager import RiskManager
        from anomaly_detector import AnomalyDetector
        from session_manager import SessionManager
        from db_logger import BotDatabase

        self.SignalDirection = SignalDirection

        self.collector = DataCollector(self.exchange, cfg)
        self.strategy = StrategyEngine(cfg)
        self.pm = PositionManager(self.exchange, cfg, dry_run=self.dry_run)
        self.risk = RiskManager(cfg)
        self.anomaly = AnomalyDetector(self.exchange, self.collector, cfg)
        self.session = SessionManager(cfg)

        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               cfg.get("database", "trades.db"))
        self.db = BotDatabase(db_path)

        # Внутреннее состояние
        self._last_grid_time = 0.0
        self._last_equity_log_time = 0.0
        self._realized_pnl = 0.0
        self._running = True

        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)

    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        """Главный торговый цикл."""
        logger.info(
            "╔══════════════════════════════════════════╗\n"
            "║  Грид-бот запущен  [сессия: %s]  ║\n"
            "║  %s / %s\n"
            "╚══════════════════════════════════════════╝",
            self.session_id,
            self.cfg["symbols"]["base"],
            self.cfg["symbols"]["quote"],
        )

        self.db.create_session(
            self.session_id,
            float(self.cfg.get("capital", 1000.0)),
            self.cfg["symbols"]["base"],
            self.cfg["symbols"]["quote"],
        )

        # Инициализация (загрузка истории, расчёт начальных EMA/σ)
        if not self.collector.initialize():
            logger.error("Инициализация DataCollector не удалась. Бот не запущен.")
            return

        # Восстановление позиций после перезапуска
        recovered = self.pm.recover_positions_from_exchange()
        if recovered:
            logger.info("Восстановлено %d позиций с OKX после перезапуска", len(recovered))

        # Первичное построение сетки
        self._rebuild_grid()

        logger.info("Бот работает. Нажмите Ctrl+C для остановки.")

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.exception("Ошибка в главном цикле: %s", exc)
                time.sleep(10)

        self._shutdown()

    # ──────────────────────────────────────────────────────────────────────

    def _tick(self):
        """Один шаг мониторинга."""
        snap = self.collector.get_snapshot()
        if snap is None:
            logger.warning("Нет данных от биржи, пропускаем цикл")
            time.sleep(self.CHECK_INTERVAL_SEC)
            return

        # ── 1. Проверка аномалий ──────────────────────────────────────
        anomaly_enabled = self.cfg.get("anomaly_detector", {}).get("enabled", True)
        is_anomaly = False
        if anomaly_enabled:
            is_anomaly = self.anomaly.check(snap)
            if is_anomaly:
                logger.info(
                    "[Bot] Аномалия: %s. Новые входы заблокированы.",
                    self.anomaly.block_reason,
                )

        # ── 2. Контроль сессии ────────────────────────────────────────
        unrealized = self.pm.get_total_pnl_usdt()
        should_stop, stop_reason = self.session.check_session_state(
            self._realized_pnl, unrealized
        )

        if should_stop:
            self._handle_session_stop(stop_reason, snap)
            return

        # ── 3. Пересчёт сетки ─────────────────────────────────────────
        if time.time() - self._last_grid_time >= self.recalc_interval_sec:
            self._rebuild_grid(snap)

        if self.strategy.grid is None:
            time.sleep(self.CHECK_INTERVAL_SEC)
            return

        # ── 4. Трейлинг-стопы ─────────────────────────────────────────
        for pos in self.pm.check_trailing_stops(snap.spread):
            self._close_position_and_record(pos, snap, "trailing_stop")

        # ── 5. Тайм-стопы ────────────────────────────────────────────
        for pos in self.pm.check_time_stops():
            self._close_position_and_record(pos, snap, "time_stop")

        # ── 6. Стоп-лосс ─────────────────────────────────────────────
        if self.strategy.check_stop_signal(snap):
            losing_dir = (
                self.SignalDirection.SHORT_SPREAD if snap.zscore > 0
                else self.SignalDirection.LONG_SPREAD
            )
            n = self.pm.count_by_direction(losing_dir)
            if n > 0:
                logger.warning(
                    "[Bot] СТОП-ЛОСС: Z=%.2f, закрываем %d позиций %s",
                    snap.zscore, n, losing_dir.value,
                )
                self.pm.close_positions_by_direction(losing_dir, reason="stop_loss")
                self._rebuild_grid(snap)  # перестраиваем сетку после стопа

        # ── 7. Тейк-профиты ──────────────────────────────────────────
        for pos_id, pos in list(self.pm.open_positions.items()):
            fraction = self.strategy.check_tp_signal(
                pos.entry_spread,
                pos.direction,
                snap.spread,
                snap.ema,
            )
            if fraction is not None:
                pnl_full = self.pm.get_position_pnl_usdt(pos)
                closed = self.pm.close_position(pos, reason="tp", fraction=fraction)
                if closed:
                    realized = pnl_full * fraction
                    self._realized_pnl += realized
                    self.session.record_trade(realized)
                    if fraction >= 1.0 and pos.level:
                        self.strategy.mark_level_free(pos.level)
                    logger.info(
                        "[Bot] ТП: %s закрыт на %.0f%% | PnL=%.2f USDT",
                        pos_id, fraction * 100, realized,
                    )
                    self.db.log_trade_close(
                        self.session_id, pos,
                        snap.price_base, snap.price_quote,
                        realized, "tp", snap,
                    )

        # ── 8. Проверка входа ─────────────────────────────────────────
        if not is_anomaly:
            self._try_enter(snap)

        # ── 9. Периодическое логирование состояния ────────────────────
        if time.time() - self._last_equity_log_time >= self.EQUITY_LOG_INTERVAL_SEC:
            self._log_state(snap, unrealized)

        time.sleep(self.CHECK_INTERVAL_SEC)

    def _try_enter(self, snap):
        """Проверяет и выполняет вход в позицию."""
        from strategy_engine import SignalDirection

        signal_obj = self.strategy.check_entry_signal(
            snap,
            open_long_count=self.pm.count_by_direction(SignalDirection.LONG_SPREAD),
            open_short_count=self.pm.count_by_direction(SignalDirection.SHORT_SPREAD),
        )
        if signal_obj is None:
            return

        # Подтверждение входа (если включено)
        if self.cfg.get("entry_confirmation", False):
            confirmed = self.collector.get_confirmation_candle(signal_obj.level.spread_value)
            if not confirmed:
                logger.debug("[Bot] Подтверждение входа не получено, пропускаем уровень")
                return

        # Риск-проверка
        allowed, reason = self.risk.check_entry_allowed(signal_obj, self.pm)
        if not allowed:
            logger.debug("[Bot] Вход отклонён риск-менеджером: %s", reason)
            return

        # Открытие позиции
        pos = self.pm.open_position(
            signal_obj.direction,
            signal_obj.level,
            signal_obj.position_size_usdt,
            snap.spread,
        )
        if pos:
            self.strategy.mark_level_used(signal_obj.level)
            self.db.log_trade_open(self.session_id, pos, snap)
            logger.info(
                "[Bot] ВХОД: %s | уровень=%.6f Z=%.2f | размер=%.0f USDT",
                signal_obj.direction.value,
                signal_obj.level.spread_value,
                signal_obj.zscore,
                signal_obj.position_size_usdt,
            )

    def _close_position_and_record(self, pos, snap, reason: str):
        """Закрывает позицию и записывает в БД/статистику."""
        pnl = self.pm.get_position_pnl_usdt(pos)
        closed = self.pm.close_position(pos, reason=reason)
        if closed:
            self._realized_pnl += pnl
            self.session.record_trade(pnl)
            if pos.level:
                self.strategy.mark_level_free(pos.level)
            self.db.log_trade_close(
                self.session_id, pos,
                snap.price_base, snap.price_quote,
                pnl, reason, snap,
            )
            logger.info("[Bot] Закрыта позиция %s (причина: %s) PnL=%.2f USDT", pos.id, reason, pnl)

    def _rebuild_grid(self, snap=None):
        """Получает снапшот (если нужно) и строит сетку."""
        if snap is None:
            snap = self.collector.get_snapshot()
        if snap is None:
            logger.warning("[Bot] Нет данных для построения сетки")
            return
        self.strategy.build_grid(snap)
        self._last_grid_time = time.time()

    def _handle_session_stop(self, reason: str, snap):
        """Обрабатывает завершение сессии."""
        logger.info("[Bot] Завершение сессии: %s", reason)
        if reason == "timeout_7_days":
            closed, pnl = self.session.handle_7day_close(self.pm)
            self._realized_pnl += pnl
            logger.info("[Bot] Безубыток: закрыто %d поз., итог=%.2f USDT", closed, pnl)
        else:
            # Закрытие по целевой прибыли или вручную
            closed_n = self.pm.close_all_positions(reason=reason)
            logger.info("[Bot] Закрыто %d позиций (%s)", closed_n, reason)
        self._running = False

    def _log_state(self, snap, unrealized: float):
        """Периодическое логирование состояния бота."""
        equity = self.cfg.get("capital", 1000.0) + self._realized_pnl + unrealized
        self.db.log_equity(
            self.session_id, equity, self._realized_pnl, unrealized
        )
        self.risk.log_risk_state(self.pm, snap.sigma)
        logger.info(
            "[Bot] ── Состояние ─────────────────────────────────\n"
            "       Позиций: %d | Equity: %.2f USDT\n"
            "       PnL реализован: %.2f | нереализован: %.2f USDT\n"
            "       Спред: %.6f | EMA: %.6f | Z: %.2f | σ: %.6f\n"
            "       День сессии: %.1f / 7",
            len(self.pm.open_positions),
            equity,
            self._realized_pnl,
            unrealized,
            snap.spread, snap.ema, snap.zscore, snap.sigma,
            self.session.elapsed_days(),
        )
        self._last_equity_log_time = time.time()

    def _shutdown(self):
        """Корректное завершение: отчёт, закрытие позиций."""
        logger.info("=== Завершение бота [сессия %s] ===", self.session_id)

        # Закрытие позиций
        action = self.cfg.get("manual_stop_action", "market_close")
        n = len(self.pm.open_positions)
        if n > 0 and action == "market_close":
            logger.info("Закрытие %d открытых позиций по рынку...", n)
            self.pm.close_all_positions(reason="shutdown")

        self.session.finalize()
        report = self.session.generate_report()
        print("\n" + report)

        self.db.finalize_session(self.session_id, self.session.stats)

        # Отчёт в файл
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"session_report_{self.session_id}.txt",
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Отчёт: %s", report_path)

        # График equity
        self._save_equity_plot()

    def _save_equity_plot(self):
        curve = self.session.get_equity_curve()
        if not curve:
            return
        try:
            import datetime
            import matplotlib.pyplot as plt
            timestamps, equities = zip(*curve)
            dates = [datetime.datetime.fromtimestamp(t) for t in timestamps]
            plt.figure(figsize=(12, 5))
            plt.plot(dates, equities, linewidth=1.2)
            plt.title(f"Equity Curve | Сессия {self.session_id} | "
                      f"{self.cfg['symbols']['base']}/{self.cfg['symbols']['quote']}")
            plt.xlabel("Время")
            plt.ylabel("Капитал (USDT)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"equity_{self.session_id}.png",
            )
            plt.savefig(plot_path, dpi=100)
            plt.close()
            logger.info("График сохранён: %s", plot_path)
        except ImportError:
            logger.info("matplotlib не установлен — график пропущен")
        except Exception as exc:
            logger.warning("Ошибка сохранения графика: %s", exc)

    def _handle_stop_signal(self, signum, frame):
        logger.info("Сигнал %s — запрашиваем остановку...", signum)
        self.session.request_stop()
        self._running = False


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Адаптивный парный грид-бот (OKX USDT Perpetual)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py                                           # dry_run, config.yaml
  python main.py --base ETHUSDT --quote CRVUSDT           # другая пара
  python main.py --config custom.yaml --no-dry-run        # боевой режим
  python -m anal.backtester --plot                         # бэктест с графиком
        """,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Путь к конфигу (default: config.yaml)")
    parser.add_argument("--base", help="Base символ (напр. ARBUSDT)")
    parser.add_argument("--quote", help="Quote символ (напр. ATOMUSDT)")
    parser.add_argument("--capital", type=float, help="Рабочий капитал в USDT")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Боевой режим (реальные ордера на OKX)")
    args = parser.parse_args()

    # .env файл (API ключи)
    load_dotenv()

    # Путь к конфигу
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)

    if not os.path.exists(config_path):
        print(f"Ошибка: конфиг не найден: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    # Переопределения из CLI
    if args.base:
        cfg["symbols"]["base"] = args.base
    if args.quote:
        cfg["symbols"]["quote"] = args.quote
    if args.capital is not None:
        cfg["capital"] = args.capital
    if args.no_dry_run:
        cfg["dry_run"] = False

    # Настройка логирования (до создания бота)
    from db_logger import setup_logging
    setup_logging(cfg, log_dir=script_dir)

    logger.info("Конфиг: %s", config_path)
    logger.info(
        "Пара: %s / %s | Капитал: %.0f USDT | Dry run: %s",
        cfg["symbols"]["base"],
        cfg["symbols"]["quote"],
        cfg.get("capital", 1000),
        cfg.get("dry_run", True),
    )

    # Предупреждение при боевом режиме
    if not cfg.get("dry_run", True):
        print("\n" + "!" * 60)
        print("!  ВНИМАНИЕ: БОЕВОЙ РЕЖИМ — реальные ордера на OKX!")
        print("!  Убедитесь, что конфиг и API-ключи верны.")
        print("!" * 60)
        confirm = input("  Введите 'YES' для подтверждения запуска: ").strip()
        if confirm != "YES":
            print("Запуск отменён.")
            sys.exit(0)
        print()

    bot = GridBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
