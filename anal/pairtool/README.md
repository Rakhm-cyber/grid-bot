# Pairtool (pair analytics)

Quick start:

```bash
python -m anal.pairtool analyze --config anal/config.example.yaml
```

Fetch (optional, saves CSVs via Binance API):

```bash
python -m anal.pairtool fetch --symbol1 BTCUSDT --symbol2 ETHUSDT --tf 1m --start 2024-01-01
```

Report from existing run:

```bash
python -m anal.pairtool report --run_id 20250101_120000
```

Output folders are created under `anal/output/run_<id>`.
