"""
db_logger — модуль логирования и хранения данных в SQLite.

Схема:
- trades       — отдельные сделки (открытие/закрытие)
- sessions     — итоги сессий
- equity_snapshots — кривая капитала

Также настраивает Python logging с выводом в файл и консоль.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    symbol_long TEXT    NOT NULL,
    symbol_short TEXT   NOT NULL,
    direction   TEXT    NOT NULL,    -- long_spread | short_spread
    action      TEXT    NOT NULL,    -- open | close | partial_close
    entry_price_long  REAL,
    entry_price_short REAL,
    exit_price_long   REAL,
    exit_price_short  REAL,
    qty_long    REAL,
    qty_short   REAL,
    notional_usdt REAL,
    pnl_usdt    REAL,
    spread_at_entry REAL,
    spread_at_exit  REAL,
    ema_at_entry    REAL,
    sigma_at_entry  REAL,
    zscore_at_entry REAL,
    reason      TEXT,                -- tp | sl | time_stop | trailing | breakeven | manual
    hold_hours  REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL UNIQUE,
    start_time      REAL    NOT NULL,
    end_time        REAL,
    status          TEXT,
    initial_capital REAL,
    final_pnl_usdt  REAL,
    final_pnl_pct   REAL,
    num_trades      INTEGER,
    num_wins        INTEGER,
    num_losses      INTEGER,
    max_drawdown_usdt REAL,
    base_symbol     TEXT,
    quote_symbol    TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    equity_usdt REAL    NOT NULL,
    realized_pnl REAL,
    unrealized_pnl REAL
);

CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_equity_session ON equity_snapshots(session_id);
"""


class BotDatabase:
    """
    Тонкая обёртка над SQLite для хранения данных бота.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_session(
        self,
        session_id: str,
        initial_capital: float,
        base_sym: str,
        quote_sym: str,
    ):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, start_time, status, initial_capital, base_symbol, quote_symbol)
                VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (session_id, time.time(), initial_capital, base_sym, quote_sym),
            )

    def finalize_session(self, session_id: str, stats):
        """Записывает итоги сессии."""
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE sessions SET
                    end_time = ?,
                    status = ?,
                    final_pnl_usdt = ?,
                    final_pnl_pct = ?,
                    num_trades = ?,
                    num_wins = ?,
                    num_losses = ?,
                    max_drawdown_usdt = ?
                WHERE session_id = ?
                """,
                (
                    stats.end_time or time.time(),
                    stats.status.value,
                    stats.final_pnl_usdt,
                    stats.pnl_pct,
                    stats.num_trades,
                    stats.num_wins,
                    stats.num_losses,
                    stats.max_drawdown_usdt,
                    session_id,
                ),
            )

    def log_trade_open(
        self,
        session_id: str,
        pos,
        snap,
    ):
        """Записывает открытие позиции."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO trades
                    (session_id, timestamp, symbol_long, symbol_short, direction,
                     action, entry_price_long, entry_price_short,
                     qty_long, qty_short, notional_usdt,
                     spread_at_entry, ema_at_entry, sigma_at_entry, zscore_at_entry)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    pos.entry_time,
                    pos.symbol_long,
                    pos.symbol_short,
                    pos.direction.value,
                    pos.entry_price_long,
                    pos.entry_price_short,
                    pos.qty_long,
                    pos.qty_short,
                    pos.notional_usdt,
                    pos.entry_spread if hasattr(pos, "entry_spread") else 0.0,
                    snap.ema if snap else 0.0,
                    snap.sigma if snap else 0.0,
                    snap.zscore if snap else 0.0,
                ),
            )

    def log_trade_close(
        self,
        session_id: str,
        pos,
        exit_price_long: float,
        exit_price_short: float,
        pnl_usdt: float,
        reason: str,
        snap=None,
    ):
        """Записывает закрытие позиции."""
        hold_h = pos.age_hours if hasattr(pos, "age_hours") else 0.0
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO trades
                    (session_id, timestamp, symbol_long, symbol_short, direction,
                     action, entry_price_long, entry_price_short,
                     exit_price_long, exit_price_short,
                     qty_long, qty_short, notional_usdt, pnl_usdt,
                     spread_at_entry, spread_at_exit, ema_at_entry,
                     sigma_at_entry, zscore_at_entry, reason, hold_hours)
                VALUES (?, ?, ?, ?, ?, 'close', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    time.time(),
                    pos.symbol_long,
                    pos.symbol_short,
                    pos.direction.value,
                    pos.entry_price_long,
                    pos.entry_price_short,
                    exit_price_long,
                    exit_price_short,
                    pos.qty_long,
                    pos.qty_short,
                    pos.notional_usdt,
                    pnl_usdt,
                    pos.entry_spread if hasattr(pos, "entry_spread") else 0.0,
                    snap.spread if snap else 0.0,
                    snap.ema if snap else 0.0,
                    snap.sigma if snap else 0.0,
                    snap.zscore if snap else 0.0,
                    reason,
                    hold_h,
                ),
            )

    def log_equity(
        self,
        session_id: str,
        equity_usdt: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
    ):
        """Записывает снапшот капитала."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots (session_id, timestamp, equity_usdt, realized_pnl, unrealized_pnl)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, time.time(), equity_usdt, realized_pnl, unrealized_pnl),
            )

    def get_session_trades(self, session_id: str) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_equity_curve(self, session_id: str) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM equity_snapshots WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def setup_logging(cfg: dict, log_dir: str = "."):
    """
    Настраивает Python logging: файл + консоль.
    """
    log_cfg = cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_file = log_cfg.get("file", "bot.log")
    log_path = os.path.join(log_dir, log_file)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Приглушаем шумные библиотеки
    for noisy in ("ccxt", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Логирование настроено: уровень=%s файл=%s", level_str, log_path)
