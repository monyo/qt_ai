import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_stock_data(symbol, period="3y", interval="1d"):
    # 建立存放數據的資料夾
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 定義檔案名稱，例如 data/NVDA_3y_1d.csv
    file_path = f"data/{symbol}_{period}_{interval}.csv"
    
    # 檢查檔案是否存在
    if os.path.exists(file_path):
        # 取得檔案最後修改時間
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
        # 如果檔案是今天更新的，就直接讀取
        if file_time == datetime.now().date():
            print(f"從本地緩存讀取 {symbol} 數據...")
            return pd.read_csv(file_path, index_col=0, parse_dates=True)

    # 如果沒有緩存或太舊，則重新抓取
    print(f"正在從 Yahoo Finance 抓取 {symbol} 最新數據...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    # 存到本地備用
    if not df.empty:
        df.to_csv(file_path)
        
    return df
