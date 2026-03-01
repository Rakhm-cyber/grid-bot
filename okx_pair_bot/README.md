# OKX Pair Grid Bot (futures)

Pair trading bot for OKX swap futures using a **market grid** on the spread.
The spread is computed as a simple price ratio:
`price_big / price_small` or `price_small / price_big` depending on `ratio_mode`.
The base leg is the ratio numerator.
For each grid step, the bot sends **two market orders at once**:
short the base leg and long the hedge leg (or vice versa).

## Quick start

1) Install deps:
```bash
pip install -r okx_pair_bot/requirements.txt
```

2) Create credentials file:
```bash
copy okx_pair_bot\credentials.example.yaml okx_pair_bot\credentials.yaml
```

3) Edit:
- `okx_pair_bot/credentials.yaml` (API key/secret/passphrase)
- `okx_pair_bot/config.yaml` (pairs, leverage, grid)

4) Run:
```bash
python okx_pair_bot/bot.py
```

## Notes

- Default `dry_run: true` (no real orders).
- Set `dry_run: false` to trade live.
- To use OKX demo, set `use_demo: true`.
- Account should be in hedged mode for `posSide`.
- Logs are written to `okx_pair_bot/bot.log`.
- Pairs can be short form: `ETH`, `BTC`.
- Real-time spread plot opens in browser when enabled in `config.yaml`.

## Grid settings

In `config.yaml`, section `grid`:
- `spread_min`, `spread_max` — ratio range for the grid.
- `ratio_mode` — `small_over_big` (default) or `big_over_small`.
- `levels` — number of grid levels.
- `max_position_steps` — max steps in one direction.
- `per_step_usdt` — position size per step (per leg).
- `beta` — coefficient in spread.
- `refresh_seconds` — how often to refresh limit orders.
- `order_timeout_seconds` — cancel/rebuild if an order is too old.

In `config.yaml`, section `plot`:
- `enabled` — open live chart in browser.
- `max_points` — history length on the chart.
- `host`, `port` — local web server for the chart.
- `refresh_ms` — chart update interval.
