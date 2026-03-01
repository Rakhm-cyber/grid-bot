"""
Полный автоматизированный пайплайн для крипто-бота:

1. Обновление данных (run.py)
2. Аналитика пар (corr_pairs_excel)
3. Выбор лучших пар (pair_selector)
4. Генерация конфига бота
5. Запуск бота (опционально)

Использование:
    python pipeline.py --full                # полный цикл
    python pipeline.py --data-only           # только обновить данные
    python pipeline.py --analyze-only        # данные + аналитика
    python pipeline.py --full --dry-run      # всё, кроме реального запуска
    python pipeline.py --full --notify       # с уведомлениями в Telegram
    python pipeline.py --full --top-pairs 5  # выбрать топ-5 пар
"""

import argparse
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "okx_pair_bot" / "config.yaml"
CORR_OUTPUT = BASE_DIR / "anal" / "output" / "corr_pairs.csv"


logger = logging.getLogger("pipeline")


def _ts() -> str:
    """Человекочитаемая метка времени."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _get_notifier(config: dict[str, Any], notify_flag: bool) -> Any:
    """Создает нотификатор если --notify передан и конфиг настроен."""
    if not notify_flag:
        from src.notifier import ConsoleNotifier
        return ConsoleNotifier()
    try:
        from src.notifier import create_notifier
        notif_cfg = config.get("notifications", {})
        return create_notifier(notif_cfg)
    except Exception as exc:
        logger.warning("Failed to create notifier: %s", exc)
        from src.notifier import ConsoleNotifier
        return ConsoleNotifier()


# ---- Pipeline steps ----


def step_update_data(notifier: Any) -> bool:
    """Шаг 1: загрузка свежих данных с Binance (через run.py)."""
    logger.info("[%s] Step 1: Updating market data...", _ts())
    try:
        from run import run_download, DEFAULT_DAYS, DEFAULT_TIMEFRAMES, DEFAULT_COINS_FILE
        run_download(
            days=DEFAULT_DAYS,
            timeframes=DEFAULT_TIMEFRAMES,
            coins_file=DEFAULT_COINS_FILE,
        )
        logger.info("[%s] Step 1: Data update complete.", _ts())
        notifier.send_message("Pipeline step 1: data update complete")
        return True
    except Exception as exc:
        msg = f"Step 1 failed: {exc}"
        logger.error("[%s] %s", _ts(), msg)
        notifier.send_error(msg)
        return False


def step_run_analytics(notifier: Any) -> bool:
    """Шаг 2: расчет корреляций и метрик пар (corr_pairs_excel)."""
    logger.info("[%s] Step 2: Running pair analytics...", _ts())
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "anal" / "corr_pairs_excel.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"corr_pairs_excel exited with code {result.returncode}: {result.stderr}")
        logger.info("[%s] Step 2: Analytics complete. Output: %s", _ts(), CORR_OUTPUT)
        notifier.send_message("Pipeline step 2: analytics complete")
        return True
    except subprocess.TimeoutExpired:
        msg = "Step 2 failed: analytics timed out (600s)"
        logger.error("[%s] %s", _ts(), msg)
        notifier.send_error(msg)
        return False
    except Exception as exc:
        msg = f"Step 2 failed: {exc}"
        logger.error("[%s] %s", _ts(), msg)
        notifier.send_error(msg)
        return False


def step_select_pairs(top_n: int, notifier: Any) -> Optional[list[dict[str, str]]]:
    """
    Шаг 3: выбор лучших пар из аналитики.

    Использует anal.pair_selector.select_top_pairs если доступен.
    Возвращает список словарей вида [{"long": "DOT", "short": "FIL"}, ...].
    """
    logger.info("[%s] Step 3: Selecting top %d pairs...", _ts(), top_n)
    try:
        from anal.pair_selector import select_top_pairs
        pairs = select_top_pairs(top_n=top_n)
        if not pairs:
            raise ValueError("select_top_pairs returned empty list")
        logger.info("[%s] Step 3: Selected pairs: %s", _ts(), pairs)
        notifier.send_message(f"Pipeline step 3: selected {len(pairs)} pairs: {pairs}")
        return pairs
    except ImportError:
        logger.warning(
            "[%s] Step 3: anal.pair_selector not found — "
            "using pairs from existing config.",
            _ts(),
        )
        notifier.send_message(
            "Pipeline step 3: pair_selector not available, keeping config pairs"
        )
        return None
    except Exception as exc:
        msg = f"Step 3 failed: {exc}"
        logger.error("[%s] %s", _ts(), msg)
        notifier.send_error(msg)
        return None


def step_generate_config(pairs: Optional[list[dict[str, str]]], notifier: Any) -> bool:
    """Шаг 4: записать выбранные пары в config.yaml бота."""
    if pairs is None:
        logger.info("[%s] Step 4: Skipping config generation (no new pairs).", _ts())
        return True
    logger.info("[%s] Step 4: Generating bot config with %d pairs...", _ts(), len(pairs))
    try:
        config = _load_yaml(CONFIG_PATH)
        config["pairs"] = pairs
        _save_yaml(CONFIG_PATH, config)
        logger.info("[%s] Step 4: Config written to %s", _ts(), CONFIG_PATH)
        notifier.send_message(f"Pipeline step 4: config updated with {len(pairs)} pairs")
        return True
    except Exception as exc:
        msg = f"Step 4 failed: {exc}"
        logger.error("[%s] %s", _ts(), msg)
        notifier.send_error(msg)
        return False


def step_start_bot(dry_run: bool, notifier: Any) -> bool:
    """Шаг 5: запуск бота (subprocess)."""
    if dry_run:
        logger.info("[%s] Step 5: Dry run — bot NOT started.", _ts())
        notifier.send_message("Pipeline step 5: dry run, bot not started")
        return True
    logger.info("[%s] Step 5: Starting bot...", _ts())
    try:
        bot_script = BASE_DIR / "okx_pair_bot" / "main.py"
        proc = subprocess.Popen(
            [sys.executable, str(bot_script)],
            cwd=str(BASE_DIR),
        )
        logger.info("[%s] Step 5: Bot started (PID=%d).", _ts(), proc.pid)
        notifier.send_message(f"Pipeline step 5: bot started (PID={proc.pid})")
        return True
    except Exception as exc:
        msg = f"Step 5 failed: {exc}"
        logger.error("[%s] %s", _ts(), msg)
        notifier.send_error(msg)
        return False


# ---- CLI ----


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Автоматизированный пайплайн для крипто-бота."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-only",
        action="store_true",
        help="Только обновить рыночные данные",
    )
    group.add_argument(
        "--analyze-only",
        action="store_true",
        help="Обновить данные + запустить аналитику",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Полный цикл: данные -> аналитика -> выбор пар -> конфиг -> бот",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Не запускать бота по-настоящему",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Отправлять Telegram-уведомления после каждого шага",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=3,
        metavar="N",
        help="Количество лучших пар для торговли (по умолчанию 3)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(BASE_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    args = parse_args()
    start = time.perf_counter()
    logger.info("[%s] Pipeline started (mode=%s)", _ts(), _mode_label(args))

    # Загружаем конфиг для секции notifications
    try:
        config = _load_yaml(CONFIG_PATH)
    except Exception:
        config = {}

    notifier = _get_notifier(config, args.notify)
    notifier.send_message(f"Pipeline started (mode={_mode_label(args)})")

    results: dict[str, bool] = {}

    # Step 1: data
    ok = step_update_data(notifier)
    results["data_update"] = ok
    if args.data_only:
        return _finish(results, start, notifier)

    # Step 2: analytics
    if ok:
        ok = step_run_analytics(notifier)
    else:
        logger.warning("[%s] Skipping analytics (data update failed).", _ts())
        ok = False
    results["analytics"] = ok
    if args.analyze_only:
        return _finish(results, start, notifier)

    # Step 3: pair selection
    pairs = step_select_pairs(args.top_pairs, notifier) if ok else None
    results["pair_selection"] = pairs is not None

    # Step 4: config
    ok_cfg = step_generate_config(pairs, notifier)
    results["config_gen"] = ok_cfg

    # Step 5: bot
    ok_bot = step_start_bot(args.dry_run, notifier)
    results["bot_start"] = ok_bot

    return _finish(results, start, notifier)


def _mode_label(args: argparse.Namespace) -> str:
    if args.data_only:
        return "data-only"
    if args.analyze_only:
        return "analyze-only"
    return "full" + (" --dry-run" if args.dry_run else "")


def _finish(results: dict[str, bool], start: float, notifier: Any) -> int:
    elapsed = time.perf_counter() - start
    all_ok = all(results.values())
    status = "SUCCESS" if all_ok else "PARTIAL FAILURE"

    summary_lines = [f"Pipeline {status} ({elapsed:.1f}s)"]
    for step, ok in results.items():
        mark = "OK" if ok else "FAIL"
        summary_lines.append(f"  {step}: {mark}")

    summary = "\n".join(summary_lines)
    logger.info("[%s] %s", _ts(), summary)
    notifier.send_message(summary)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
