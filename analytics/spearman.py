import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import glob
import os

# Папка с CSV
data_folder = "../data/binance" 

# Загружаем все CSV
files = glob.glob(os.path.join(data_folder, "*.csv"))
dfs = {}
for file in files:
    symbol = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, usecols=['ts_iso', 'close'])
    df['ts_iso'] = pd.to_datetime(df['ts_iso'])
    df.set_index('ts_iso', inplace=True)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna().reset_index()
    dfs[symbol] = df
    
# Все возможные пары
all_pairs = list(combinations(dfs.keys(), 2))

results = []

for sym1, sym2 in all_pairs:
    df1 = dfs[sym1]
    df2 = dfs[sym2]
    df = df1.merge(df2, on = 'ts_iso')
    df = df1.join(df2, lsuffix='_1', rsuffix='_2').dropna()
    if df.empty:
        continue
    pair = (sym1, sym2)
    rho = df['log_ret_1'].corr(df['log_ret_2'], method = 'spearman')
    results.append({
        'pair': f'{sym1}-{sym2}',
        'rho': rho
    })

final = pd.DataFrame(results)
final.to_excel('spearman.xlsx', index=False)