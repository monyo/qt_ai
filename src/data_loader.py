from io import StringIO
import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime

def fetch_stock_data(symbol, period="3y", start=None, end=None):
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 定義檔案名稱：如果是指定日期，檔名要包含日期以免混淆
    if start and end:
        file_path = f"data/{symbol}_{start}_{end}.csv"
    else:
        file_path = f"data/{symbol}_{period}.csv"
    
    # 這裡我們先簡化邏輯：如果有檔案就讀取，沒有就抓新的
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    print(f"正在抓取 {symbol} 數據...")
    ticker = yf.Ticker(symbol)
    
    if start and end:
        df = ticker.history(start=start, end=end, auto_adjust=True)
    else:
        df = ticker.history(period=period, auto_adjust=True)
        
    if not df.empty:
        df.to_csv(file_path)
    if df.empty or len(df) < 100:
        return pd.DataFrame()

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


def fetch_current_prices(symbols):
    """批次取得最新收盤價（前一個交易日 Close）

    盤前執行時，yfinance 回傳的最新 Close 即為昨日收盤。
    回傳 dict: {symbol: price}
    """
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", auto_adjust=True)
            if not hist.empty:
                prices[symbol] = round(hist["Close"].iloc[-1], 2)
        except Exception as e:
            print(f"⚠ 無法取得 {symbol} 報價: {e}")
    return prices
