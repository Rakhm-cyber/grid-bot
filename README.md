# Grid Bot — аналитический блок (anal)

## Последовательность работы пользователя

1. Обновить данные из Binance через `run.py` или флаг `--refresh-data`.
2. Проверить список монет в `coins.txt`.
3. Запустить расчет парных метрик в `anal/corr_pairs_excel.py`.
4. Открыть `anal/output/corr_pairs.csv` и анализировать/фильтровать пары в Excel.

## Загрузка данных (коротко)

Данные подтягиваются через `run.py` и сохраняются в `data/binance/{SYMBOL}_15m.csv`.
Скрипт берет последние ~35 дней и **добавляет/обновляет** строки по `ts_ms`
(старые строки не удаляются, файл перезаписывается целиком после слияния).

При необходимости можно вручную изменить скрипт - поменять таймфрейм и также временной период(не забудьте поменять постфикс сохраняемых файлов)

Запуск:
```bash
python run.py
```

Альтернатива — запустить аналитику с обновлением данных:
```bash
python anal/corr_pairs_excel.py --refresh-data
```

## Аналитический блок (папка `anal`)

`anal` — финансово ориентированный блок аналитики для парного трейдинга
и оценки mean reversion. Цель — найти пары с устойчивой совместной динамикой
и спредом, склонным возвращаться к среднему.

### Используемые статистические методы

- Kendall correlation — ранговая связь доходностей по неделям/месяцу.
- Engle-Granger cointegration (p-value) — проверка коинтеграции по лог-ценам.
- Half-life (AR(1)) — скорость возврата спреда к среднему.
- Hurst exponent — склонность к mean reversion (H < 0.5).
- Z-score in band — доля времени, когда спред в “нормальном” коридоре.

## corr_pairs_excel.py — парные метрики и скоринг

Скрипт считает метрики для всех пар из `coins.txt` и пишет CSV:
`anal/output/corr_pairs.csv`.

Колонки:
- `kendall_week1..week4`, `kendall_month` — Kendall корреляция доходностей.
- `coint_p_week1..week4`, `coint_p_month` — p-value коинтеграции.
- `half_life_month`, `hurst_month`, `zscore_in_band_month` — только месяц.
- `s_c`, `gate`, `s_mr`, `s_k`, `final_score` — итоговый скоринг.

`gate = 1` означает, что пара прошла фильтр по коинтеграции/hurst.

Запуск:
```bash
python anal/corr_pairs_excel.py
```

Полезные флаги:
```bash
python anal/corr_pairs_excel.py --refresh-data
python anal/corr_pairs_excel.py --zscore-band 1.0
python anal/corr_pairs_excel.py --hl-min 20 --hl-max 800
python anal/corr_pairs_excel.py --output anal/output/corr_pairs.csv
```
