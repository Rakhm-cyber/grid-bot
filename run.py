from datetime import datetime, timedelta, timezone
from src.common.constants_enums import BinanceInterval
from src.clients.binance_client import BinanceClient

client = BinanceClient()

client.get_rates_and_save(
    start_dt=datetime.now(timezone.utc) - timedelta(days=7),
    interval='1m',
    symbol="BTCUSDT",
    filepath="data/binance/BTCUSDT_1m.csv",
)
