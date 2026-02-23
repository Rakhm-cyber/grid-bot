import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import glob
import os

# Папка с CSV
data_folder = "C:/Users/nice-/Documents/projects/grid-bot/data/binance" ##  поменять название папки, у меня все наебнулось, запускается так

# Загружаем все CSV
files = glob.glob(os.path.join(data_folder, "*.csv"))
dfs = {}
for file in files:
    symbol = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, usecols=['ts_iso', 'close'])
    df['ts_iso'] = pd.to_datetime(df['ts_iso'])
    df.set_index('ts_iso', inplace=True)
    dfs[symbol] = df

# Все возможные пары
all_pairs = list(combinations(dfs.keys(), 2))

results = []

for sym1, sym2 in all_pairs:
    df1 = dfs[sym1]
    df2 = dfs[sym2]
    
    # Объединяем по времени
    df = df1.join(df2, lsuffix='_1', rsuffix='_2').dropna()
    if df.empty:
        continue
    
    # Лог-спред
    df['log_spread'] = np.log(df['close_1']) - np.log(df['close_2'])
    
    # Списки для p-value
    week_pvalues = []
    month_pvalues = []
    
    # ADF p-value для недель
    week_len = 7 * 24 * 4
    for i in range(0, len(df), week_len):
        window = df['log_spread'].iloc[i:i+week_len]
        if len(window) == week_len:
            pvalue = adfuller(window)[1]
            week_pvalues.append(pvalue)
    
    # ADF p-value для месяцев (28 дней)
    month_len = 28 * 24 * 4
    for i in range(0, len(df), month_len):
        window = df['log_spread'].iloc[i:i+month_len]
        if len(window) == month_len:
            pvalue = adfuller(window)[1]
            month_pvalues.append(pvalue)
    
    # Формируем строку результата
    row = {"pair": f"{sym1}-{sym2}"}
    
    # Добавляем week1, week2, ...
    for idx, p in enumerate(week_pvalues, 1):
        row[f"week{idx}"] = p
    
    # Добавляем month1, month2, ...
    for idx, p in enumerate(month_pvalues, 1):
        row[f"month{idx}"] = p
    
    results.append(row)

# Финальный DataFrame
adf_df = pd.DataFrame(results)

periods = [col for col in adf_df.columns if col != 'pair']
start_weight = 0.1
end_weight = 0.5
weights = np.linspace(start_weight, end_weight, len(periods))
weights_dict = dict(zip(periods, weights))

adf_df['score'] = sum((1 - adf_df[p]) * w for p, w in weights_dict.items())
adf_df['score'] = adf_df['score'] / sum(weights_dict.values())

# Сохраняем
adf_df_final = adf_df.sort_values(by = 'score', ascending = False)
adf_df_final.to_excel("RES1.xlsx", index=False)
print("Готово! Результат в Test.xlsx")