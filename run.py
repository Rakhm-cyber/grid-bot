from datetime import datetime, timedelta, timezone
from src.clients.binance_client import BinanceClient

client = BinanceClient()

start = datetime.now(timezone.utc) - timedelta(days=2)
points = client.get_rates(
    start_dt=start,
    interval="1h",
    base_currency="BTC",
    quote_currency="USDT",
)

client.to_csv(points, "exports/btcusdt_1h.csv")

for p in points[:5]:
    print(p.ts, p.close)
