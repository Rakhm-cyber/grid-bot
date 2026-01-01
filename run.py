from datetime import datetime, timedelta, timezone
from src.common.constants_enums import BinanceInterval
from src.clients.binance_client import BinanceClient

client = BinanceClient()

with open("coins.txt", "r", encoding="utf-8") as f:
    coins = [line.strip() for line in f if line.strip()]  # убираем пустые строки и пробелы

for symbol in coins:
    print(f"Обработка {symbol}...")

    try:
        client.get_rates_and_save(
            start_dt=datetime.now(timezone.utc) - timedelta(days=365),
            interval='1m',
            symbol=symbol,
            filepath=f"data/binance/{symbol}_1m.csv",
        )
        print(f"{symbol} — готово.")
    except Exception as e:
        print(f"Ошибка при обработке {symbol}: {e}")
