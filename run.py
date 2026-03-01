import argparse
from datetime import datetime, timedelta, timezone
from src.common.constants_enums import BinanceInterval
from src.clients.binance_client import BinanceClient

DEFAULT_DAYS = 180
DEFAULT_TIMEFRAMES = ['15m', '1h', '4h']
DEFAULT_COINS_FILE = 'coins.txt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Скачивание свечей с Binance для списка монет'
    )
    parser.add_argument(
        '--days', type=int, default=DEFAULT_DAYS,
        help=f'Глубина истории в днях (по умолчанию {DEFAULT_DAYS})'
    )
    parser.add_argument(
        '--timeframes', nargs='+', default=DEFAULT_TIMEFRAMES,
        help=f'Список таймфреймов (по умолчанию {" ".join(DEFAULT_TIMEFRAMES)})'
    )
    parser.add_argument(
        '--coins-file', default=DEFAULT_COINS_FILE,
        help=f'Путь к файлу со списком монет (по умолчанию {DEFAULT_COINS_FILE})'
    )
    return parser.parse_args()


def run_download(days: int, timeframes: list[str], coins_file: str) -> None:
    """Основная логика скачивания — используется и напрямую, и из scheduler."""
    client = BinanceClient()

    with open(coins_file, "r", encoding="utf-8") as f:
        coins = [line.strip() for line in f if line.strip()]  # убираем пустые строки и пробелы

    start_dt = datetime.now(timezone.utc) - timedelta(days=days)

    for symbol in coins:
        for tf in timeframes:
            print(f"Обработка {symbol} [{tf}]...")

            try:
                client.get_rates_and_save(
                    start_dt=start_dt,
                    interval=tf,
                    symbol=symbol,
                    filepath=f"data/binance/{symbol}_{tf}.csv",
                )
                print(f"  {symbol} [{tf}] — готово.")
            except Exception as e:
                print(f"  Ошибка при обработке {symbol} [{tf}]: {e}")


if __name__ == '__main__':
    args = parse_args()
    run_download(
        days=args.days,
        timeframes=args.timeframes,
        coins_file=args.coins_file,
    )
