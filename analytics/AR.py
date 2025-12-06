import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import os

##--чистка от выбросов--##
def find_v(df, column):
    s = df[column]
    Q1 = np.quantile(s, 0.25)
    Q3 = np.quantile(s, 0.75)
    IQR = Q3 - Q1
    lim_max = Q3 + 5 * IQR
    lim_min = Q1 - 5 * IQR 
    return df[(s >= lim_min) & (s <= lim_max)]

##--поиск лучшего лага--##
def the_best_lag(series, max_lag=15):
    best_aic = float('inf')
    best_lag = 1
    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(series, lags=p).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_lag = p
        except:
            continue
    return best_lag

##--поиск лучшего интервала--##
def best_interval(df, max_time=180, max_lag=15):
    df = df.copy()
    df['ts_iso'] = pd.to_datetime(df['ts_iso'])
    df.set_index('ts_iso', inplace=True)

    best_model = None
    best_interval = None
    best_df = None
    best_aic = float('inf')

    for p in range(1, max_time + 1):

        df_p = df['close'].resample(f'{p}min').last().dropna()
        if len(df_p) < 30:
            continue

        lag = the_best_lag(df_p)
        try:
            model = AutoReg(df_p, lags=lag).fit()
        except:
            continue

        if model.aic < best_aic:
            best_aic = model.aic
            best_interval = p
            best_model = model
            best_df = df_p.reset_index()

    # fallback
    if best_model is None:
        df_p = df['close'].dropna()
        best_df = df_p.reset_index()
        best_interval = 1
        best_model = AutoReg(df_p, lags=1).fit()

    return best_model, best_interval, best_df

##--cтационирование ряда после ресэмплинга--##
def sts_check(df, column):
    series = df[column].copy()
    original_last_value = series.iloc[-1]

    log_used = False
    diff_count = 0

    # Логарифм
    if (series > 0).all():
        series = np.log(series)
        log_used = True

    # Проверка ADF
    p_value = adfuller(series.dropna())[1]

    # Дифференцирование до стационарности
    while p_value > 0.05:
        series = series.diff().dropna()
        diff_count += 1
        p_value = adfuller(series)[1]

    # для восстановления
    last_log_value = np.log(original_last_value) if log_used else original_last_value

    return (
        log_used,
        diff_count,
        pd.DataFrame({column: series}),
        original_last_value,
        last_log_value
    )

##--oбратное восстановление прогноза--##
def restore_predictions(forecast_diff, log_used, diff_count, last_log_value, last_original_value):

    # Прогноз на стационарном ряду
    restored = forecast_diff.copy()

    # Интегрирование n раз
    for i in range(diff_count):
        last_value = last_log_value if (i == 0 and log_used) else restored.iloc[0] - forecast_diff.iloc[0]
        new_series = []
        acc = last_value
        for x in forecast_diff:
            acc = acc + x
            new_series.append(acc)
        restored = pd.Series(new_series)

    # Обратный лог
    if log_used:
        restored = np.exp(restored)

    return restored

with open("coins.txt", "r", encoding="utf-8") as f:
    coins = [line.strip() for line in f if line.strip()] 
for symbol in coins:
    if os.path.exists(f'./data/binance/{symbol}_1m.csv'):
        print(f'Строится прогноз для {symbol}')
        df_start = pd.read_csv(f'./data/binance/{symbol}_1m.csv')
        df_start['ts_iso'] = pd.to_datetime(df_start['ts_iso'])

        df_clean = find_v(df_start, 'close')

        # Находится лучший интервал на неизменённом ряде
        model0, best_int, best_df = best_interval(df_clean)

        # Стационирование уже РЕСЭМПЛИРОВАННОГО ряда
        log_used, diff_count, df_st, last_original_value, last_log_value = sts_check(best_df, 'close')

        # Обучение модели на стационарном ряду
        lag = the_best_lag(df_st['close'])
        final_model = AutoReg(df_st['close'], lags=lag).fit()

        # Прогноз стационарного ряда
        forecast_diff = final_model.predict(
            start=len(df_st),
            end=len(df_st)+9 ## тут на 9 значений
        )

        # Обратное восстановление
        forecast = restore_predictions(
            forecast_diff,
            log_used,
            diff_count,
            last_log_value,
            last_original_value
        )

        # Формируем df прогноза
        forecast_times = pd.date_range(
            start=best_df['ts_iso'].iloc[-1] + pd.Timedelta(minutes=best_int),
            periods=len(forecast),
            freq=f'{best_int}min'
        )

        df_forecast = pd.DataFrame({
            'ts_iso': forecast_times,
            'close': forecast.values
        })

        df_forecast.to_csv(f'./data/forecast/{symbol}.csv', index=False)
    else:
        print(f'Монеты {symbol} нет')
        