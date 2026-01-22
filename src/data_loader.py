from io import StringIO
import yfinance as yf
import pandas as pd
import os
import requests
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

def get_sp500_tickers():
    """從維基百科抓取 S&P 500 成份股代碼 (帶 User-Agent 偽裝)"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # 模擬瀏覽器的 Header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 使用 requests 先把網頁抓下來
    response = requests.get(url, headers=headers)
    
    # 再讓 pandas 解析抓回來的 HTML 內容
    table = pd.read_html(StringIO(response.text))
    df = table[0]
    return df['Symbol'].tolist()
