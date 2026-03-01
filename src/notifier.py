"""
Telegram и консольный нотификатор для торгового бота.

TelegramNotifier отправляет сообщения через Telegram Bot API
с rate-limiting (до 20 сообщений в минуту).
ConsoleNotifier — fallback, который просто логирует в stdout.
"""

import logging
import os
import time
from collections import deque
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class ConsoleNotifier:
    """Fallback-нотификатор: выводит сообщения в лог."""

    def send_message(self, text: str) -> None:
        logger.info("[NOTIFY] %s", text)

    def send_trade_alert(
        self, pair: str, side: str, price: float, pnl: Optional[float] = None
    ) -> None:
        pnl_str = f", PnL: {pnl:+.4f}" if pnl is not None else ""
        self.send_message(f"TRADE | {pair} {side} @ {price:.6f}{pnl_str}")

    def send_error(self, error_msg: str) -> None:
        logger.error("[NOTIFY][ERROR] %s", error_msg)

    def send_daily_summary(self, stats: dict[str, Any]) -> None:
        lines = ["--- Daily Summary ---"]
        for key, value in stats.items():
            lines.append(f"  {key}: {value}")
        self.send_message("\n".join(lines))


class TelegramNotifier:
    """
    Отправка уведомлений через Telegram Bot API.

    Параметры берутся из аргументов конструктора, конфига или env vars:
      TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    Rate limit: не более MAX_PER_MINUTE сообщений за 60 секунд.
    Если токен не задан — автоматически переключается на ConsoleNotifier.
    """

    MAX_PER_MINUTE = 20
    API_TIMEOUT = 10

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._timestamps: deque[float] = deque()
        self._fallback = ConsoleNotifier()

        if not self._token or not self._chat_id:
            logger.warning(
                "Telegram credentials not configured — using ConsoleNotifier fallback"
            )

    @property
    def _is_configured(self) -> bool:
        return bool(self._token) and bool(self._chat_id)

    # ---- rate limiting ----

    def _wait_rate_limit(self) -> None:
        """Блокирует до тех пор, пока не появится слот для отправки."""
        now = time.time()
        # Удаляем штампы старше 60 секунд
        while self._timestamps and self._timestamps[0] < now - 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.MAX_PER_MINUTE:
            wait = 60.0 - (now - self._timestamps[0])
            if wait > 0:
                logger.debug("Rate limit hit, sleeping %.1fs", wait)
                time.sleep(wait)

    def _record_send(self) -> None:
        self._timestamps.append(time.time())

    # ---- core send ----

    def send_message(self, text: str) -> None:
        """Отправить произвольное текстовое сообщение (Markdown)."""
        if not self._is_configured:
            self._fallback.send_message(text)
            return
        self._wait_rate_limit()
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.API_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("Telegram API error %s: %s", resp.status_code, resp.text)
        except requests.RequestException as exc:
            logger.warning("Telegram send failed: %s", exc)
        self._record_send()

    # ---- convenience methods ----

    def send_trade_alert(
        self, pair: str, side: str, price: float, pnl: Optional[float] = None
    ) -> None:
        """Уведомление о сделке."""
        pnl_str = f"\nPnL: `{pnl:+.4f}`" if pnl is not None else ""
        text = (
            f"*TRADE*\n"
            f"Pair: `{pair}`\n"
            f"Side: *{side}*\n"
            f"Price: `{price:.6f}`{pnl_str}"
        )
        self.send_message(text)

    def send_error(self, error_msg: str) -> None:
        """Уведомление об ошибке."""
        text = f"*ERROR*\n`{error_msg}`"
        self.send_message(text)

    def send_daily_summary(self, stats: dict[str, Any]) -> None:
        """Ежедневная сводка."""
        lines = ["*Daily Summary*"]
        for key, value in stats.items():
            lines.append(f"  {key}: `{value}`")
        self.send_message("\n".join(lines))


def create_notifier(config: Optional[dict[str, Any]] = None) -> ConsoleNotifier | TelegramNotifier:
    """
    Фабрика нотификатора по секции ``notifications`` конфига.

    Если Telegram отключен или не настроен — возвращает ConsoleNotifier.
    """
    if config is None:
        return ConsoleNotifier()
    tg_cfg = config.get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return ConsoleNotifier()
    return TelegramNotifier(
        bot_token=tg_cfg.get("bot_token", ""),
        chat_id=str(tg_cfg.get("chat_id", "")),
    )
