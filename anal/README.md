# Аналитика парной торговли (папка `anal`)

В этой папке лежит инструмент для анализа пары монет (pair trading research). Он читает два временных ряда, синхронизирует их, строит спред, проводит статистические тесты и сохраняет артефакты (JSON/CSV/графики/HTML).

## Что внутри

- `anal/config.example.yaml` — пример конфигурации пайплайна.
- `anal/simpe_pair.py` — минимальный пример запуска анализа из Python-кода.
- `anal/__init__.py` — делает папку пакетом.
- `anal/pairtool/` — сам модуль аналитики:
  - `__main__.py` — CLI-команды `analyze`, `fetch`, `report`.
  - `config.py` — парсинг YAML-конфига в объект `Config`.
  - `loaders.py` — загрузка данных из CSV/Parquet/через Binance API.
  - `preprocessing.py` — синхронизация рядов, лог-преобразование, обработка пропусков и выбросов.
  - `tests.py` — статистические тесты (корреляции, ADF/KPSS/PP, Engle–Granger, нормальность, Granger).
  - `rolling.py` — rolling-метрики (beta, corr, ADF p-value).
  - `reporting.py` — сохранение JSON/CSV/графиков/HTML.
  - `pipeline.py` — центральный пайплайн, который всё собирает.
  - `README.md` — краткая инструкция по запуску.

## Быстрый запуск

Полный анализ по конфигу:

```bash
python -m anal.pairtool analyze --config anal/config.example.yaml
```

Прогон всех пар из `coins.txt`:

```bash
python -m anal.pairtool analyze --config anal/config.example.yaml --coins-file coins.txt --jobs 4
```

Во время прогона выводится прогресс вида `[done/total] SYMBOL1/SYMBOL2 done`.

Если установлен `tqdm`, будет отображаться progress bar для списка пар
и отдельный progress bar для rolling-окон в одиночном запуске.

Отключить прогресс можно флагом `--no-progress`.

Минимальный пример из Python:

```bash
python anal/simpe_pair.py
```

Опционально скачать данные (Binance API):

```bash
python -m anal.pairtool fetch --symbol1 BTCUSDT --symbol2 ETHUSDT --tf 1m --start 2024-01-01
```

Сбор HTML-отчёта из уже выполненного прогона (подробный, с пояснениями):

```bash
python -m anal.pairtool report --run_id 20250101_120000
```

## Формат входных данных

Ожидается CSV с колонками:

- `ts_ms` (unix ms) или `ts_iso` (ISO8601) или `timestamp`
- `close` (и опционально `open`, `high`, `low`, `volume`)

Файлы берутся из `data/binance/` по шаблону `{SYMBOL}_{TF}.csv` (например, `BTCUSDT_1m.csv`).

## Выходные артефакты

Каждый запуск создаёт папку:

```
anal/output/run_<YYYYMMDD_HHMMSS>/
```

Внутри:

- `result.json` — полный результат расчёта.
- `aligned_series.csv` — синхронизированные ряды и спред.
- `rolling_metrics.csv` — rolling-метрики.
- `break_tests.csv` — Chow тест (если включён).
- `plots/` — графики (log-prices, spread, z-score, rolling beta/corr/adf, chow p-values).
- `report.html` — краткий HTML-отчёт (если включён).

## Конфигурация

Главные секции в `config.example.yaml`:

- `symbols` — пара монет.
- `timeframe` — таймфрейм, например `1m`, `1h`.
- `data_source` — `csv`, `parquet`, `binance`.
- `preprocessing` — лог-преобразование, выбросы, пропуски.
- `tests` — включаемые тесты (ADF/KPSS/PP, коинтеграция, нормальность, Granger).
- `rolling` — окно, шаг и минимальные периоды.
- `breaks` — Chow test.
- `reporting` — форматы вывода и графики.

## Заметки

- Инструмент не исполняет сделки — только аналитика.
- Для больших файлов может потребоваться время на расчёт rolling-метрик.
- Для `binance` требуется рабочий интернет и доступ к API.
