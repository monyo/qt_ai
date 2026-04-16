"""
_build_ohlcv_cache.py

建立 OHLCV 快取（High / Low / Volume），供停損回測使用。
Close-only 快取不動，兩個快取並存。

輸出：data/_protection_bt_ohlcv.pkl
格式：{'Close': df, 'High': df, 'Low': df, 'Volume': df}
      每個 df 形狀與 _protection_bt_prices.pkl 相同（1695 日 × 501 股）

使用：
    conda run -n qt_env python _build_ohlcv_cache.py
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import yfinance as yf
from src.data_loader import get_sp500_tickers

CACHE_PATH = "data/_protection_bt_ohlcv.pkl"
DATA_START = "2019-06-01"
DATA_END   = "2026-03-01"
BATCH_SIZE = 50

if os.path.exists(CACHE_PATH):
    print(f"快取已存在：{CACHE_PATH}，跳過下載。")
    ohlcv = pd.read_pickle(CACHE_PATH)
    for k, df in ohlcv.items():
        print(f"  {k}: {df.shape}")
    sys.exit(0)

tickers = get_sp500_tickers()
print(f"下載 {len(tickers)} 檔 OHLCV（每批 {BATCH_SIZE}，共 {len(tickers)//BATCH_SIZE+1} 批）...")
print(f"預計時間：20~30 分鐘")

batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

closes, highs, lows, volumes = [], [], [], []

for i, batch in enumerate(batches):
    print(f"  批次 {i+1}/{len(batches)}...", end=" ", flush=True)
    try:
        raw = yf.download(batch, start=DATA_START, end=DATA_END,
                          auto_adjust=True, progress=False)
        if raw.empty:
            print("空")
            continue

        def extract(field):
            df = raw[field]
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)
            elif isinstance(df, pd.Series):
                df = df.to_frame(name=batch[0])
            return df

        closes.append(extract("Close"))
        highs.append(extract("High"))
        lows.append(extract("Low"))
        volumes.append(extract("Volume"))
        print(f"OK ({extract('Close').shape[1]} 檔)")
    except Exception as e:
        print(f"ERROR: {e}")

def merge(dfs):
    df = pd.concat(dfs, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None) \
        if df.index.tz is None else pd.to_datetime(df.index).tz_convert(None)
    return df

ohlcv = {
    "Close":  merge(closes),
    "High":   merge(highs),
    "Low":    merge(lows),
    "Volume": merge(volumes),
}

# 對齊欄位（只保留四個欄位都有的股票）
common_cols = (set(ohlcv["Close"].columns)
               & set(ohlcv["High"].columns)
               & set(ohlcv["Low"].columns)
               & set(ohlcv["Volume"].columns))
for k in ohlcv:
    ohlcv[k] = ohlcv[k][sorted(common_cols)]

os.makedirs("data", exist_ok=True)
pd.to_pickle(ohlcv, CACHE_PATH)

print(f"\n完成！已儲存至 {CACHE_PATH}")
for k, df in ohlcv.items():
    print(f"  {k}: {df.shape}")
