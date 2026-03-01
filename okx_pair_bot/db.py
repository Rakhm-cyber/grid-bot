"""
SQLite-хранилище для трейдов, сессий и снэпшотов equity.
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "trades.db"


class TradeDB:
    """Обёртка над SQLite для логирования сделок и сессий."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       REAL    NOT NULL,
                pair            TEXT    NOT NULL,
                side            TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                amount          REAL    NOT NULL,
                price           REAL    NOT NULL,
                pnl_usdt        REAL    NOT NULL DEFAULT 0.0,
                spread_at_entry REAL    NOT NULL DEFAULT 0.0,
                spread_at_exit  REAL    NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time      REAL    NOT NULL,
                end_time        REAL,
                pair            TEXT    NOT NULL,
                total_pnl       REAL    NOT NULL DEFAULT 0.0,
                max_drawdown    REAL    NOT NULL DEFAULT 0.0,
                num_trades      INTEGER NOT NULL DEFAULT 0,
                config_json     TEXT    NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                timestamp       REAL    NOT NULL,
                equity_usdt     REAL    NOT NULL,
                spread          REAL    NOT NULL DEFAULT 0.0,
                pos_steps       INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Сессии
    # ------------------------------------------------------------------

    def start_session(self, pair: str, config: dict[str, Any] | None = None) -> int:
        """Начать новую торговую сессию, вернуть session_id."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO sessions (start_time, pair, config_json) VALUES (?, ?, ?)",
            (time.time(), pair, json.dumps(config or {}, ensure_ascii=False)),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def end_session(
        self,
        session_id: int,
        total_pnl: float = 0.0,
        max_drawdown: float = 0.0,
        num_trades: int = 0,
    ) -> None:
        """Завершить сессию, записать итоговые результаты."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE sessions
               SET end_time     = ?,
                   total_pnl    = ?,
                   max_drawdown = ?,
                   num_trades   = ?
             WHERE id = ?""",
            (time.time(), total_pnl, max_drawdown, num_trades, session_id),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Трейды
    # ------------------------------------------------------------------

    def log_trade(
        self,
        pair: str,
        side: str,
        symbol: str,
        amount: float,
        price: float,
        pnl_usdt: float = 0.0,
        spread_at_entry: float = 0.0,
        spread_at_exit: float = 0.0,
    ) -> int:
        """Записать одну сделку, вернуть trade_id."""
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO trades
               (timestamp, pair, side, symbol, amount, price,
                pnl_usdt, spread_at_entry, spread_at_exit)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                pair,
                side,
                symbol,
                amount,
                price,
                pnl_usdt,
                spread_at_entry,
                spread_at_exit,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def snapshot_equity(
        self,
        session_id: int,
        equity_usdt: float,
        spread: float = 0.0,
        pos_steps: int = 0,
    ) -> None:
        """Сохранить снэпшот equity для сессии."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO equity_snapshots
               (session_id, timestamp, equity_usdt, spread, pos_steps)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, time.time(), equity_usdt, spread, pos_steps),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    def get_session_stats(self, session_id: int) -> dict[str, Any]:
        """Вернуть статистику сессии: pnl, drawdown, число сделок, длительность."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return {}
        stats: dict[str, Any] = dict(row)

        trades = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_usdt), 0) as total_pnl "
            "FROM trades WHERE pair = ? AND timestamp >= ? AND timestamp <= ?",
            (row["pair"], row["start_time"], row["end_time"] or time.time()),
        ).fetchone()
        if trades:
            stats["trades_count"] = trades["cnt"]
            stats["trades_total_pnl"] = trades["total_pnl"]

        snapshots = conn.execute(
            "SELECT MIN(equity_usdt) as min_eq, MAX(equity_usdt) as max_eq "
            "FROM equity_snapshots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if snapshots and snapshots["max_eq"] is not None:
            stats["min_equity"] = snapshots["min_eq"]
            stats["max_equity"] = snapshots["max_eq"]

        return stats

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
