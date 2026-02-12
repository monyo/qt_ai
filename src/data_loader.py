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


def get_tw50_tickers():
    """取得台灣50成分股 + 熱門概念股代碼（yfinance 格式）"""
    return [
        # === 台灣50成分股 ===
        "2330.TW",  # 台積電
        "2317.TW",  # 鴻海
        "2454.TW",  # 聯發科
        "2308.TW",  # 台達電
        "2881.TW",  # 富邦金
        "2882.TW",  # 國泰金
        "2891.TW",  # 中信金
        "2303.TW",  # 聯電
        "2412.TW",  # 中華電
        "1301.TW",  # 台塑
        "1303.TW",  # 南亞
        "2002.TW",  # 中鋼
        "2886.TW",  # 兆豐金
        "3711.TW",  # 日月光投控
        "2884.TW",  # 玉山金
        "2357.TW",  # 華碩
        "2382.TW",  # 廣達
        "2892.TW",  # 第一金
        "5880.TW",  # 合庫金
        "2880.TW",  # 華南金
        "3045.TW",  # 台灣大
        "2885.TW",  # 元大金
        "2207.TW",  # 和泰車
        "1216.TW",  # 統一
        "2301.TW",  # 光寶科
        "4904.TW",  # 遠傳
        "2395.TW",  # 研華
        "5871.TW",  # 中租-KY
        "2327.TW",  # 國巨
        "3008.TW",  # 大立光
        "2912.TW",  # 統一超
        "1326.TW",  # 台化
        "2379.TW",  # 瑞昱
        "6505.TW",  # 台塑化
        "2883.TW",  # 開發金
        "4938.TW",  # 和碩
        "2345.TW",  # 智邦
        "3034.TW",  # 聯詠
        "2603.TW",  # 長榮
        "6669.TW",  # 緯穎
        "2890.TW",  # 永豐金
        "3037.TW",  # 欣興
        "2609.TW",  # 陽明
        "1101.TW",  # 台泥
        "2615.TW",  # 萬海
        "2801.TW",  # 彰銀
        "5876.TW",  # 上海商銀
        "9910.TW",  # 豐泰
        "2408.TW",  # 南亞科
        "2474.TW",  # 可成
        # === 記憶體/儲存概念股 ===
        "2344.TW",  # 華邦電
        "2337.TW",  # 旺宏
        # === 低軌衛星概念股 ===
        "2314.TW",  # 台揚
        "6285.TW",  # 啟碁
        "6271.TW",  # 同欣電
        "3044.TW",  # 健鼎
        "2439.TW",  # 美律
    ]


# 台股代碼對照表
TW_STOCK_NAMES = {
    # 台灣50
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科",
    "2308.TW": "台達電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "2891.TW": "中信金", "2303.TW": "聯電", "2412.TW": "中華電",
    "1301.TW": "台塑", "1303.TW": "南亞", "2002.TW": "中鋼",
    "2886.TW": "兆豐金", "3711.TW": "日月光", "2884.TW": "玉山金",
    "2357.TW": "華碩", "2382.TW": "廣達", "2892.TW": "第一金",
    "5880.TW": "合庫金", "2880.TW": "華南金", "3045.TW": "台灣大",
    "2885.TW": "元大金", "2207.TW": "和泰車", "1216.TW": "統一",
    "2301.TW": "光寶科", "4904.TW": "遠傳", "2395.TW": "研華",
    "5871.TW": "中租-KY", "2327.TW": "國巨", "3008.TW": "大立光",
    "2912.TW": "統一超", "1326.TW": "台化", "2379.TW": "瑞昱",
    "6505.TW": "台塑化", "2883.TW": "開發金", "4938.TW": "和碩",
    "2345.TW": "智邦", "3034.TW": "聯詠", "2603.TW": "長榮",
    "6669.TW": "緯穎", "2890.TW": "永豐金", "3037.TW": "欣興",
    "2609.TW": "陽明", "1101.TW": "台泥", "2615.TW": "萬海",
    "2801.TW": "彰銀", "5876.TW": "上海商銀", "9910.TW": "豐泰",
    "2408.TW": "南亞科", "2474.TW": "可成",
    # 記憶體/儲存概念股
    "2344.TW": "華邦電", "2337.TW": "旺宏",
    # 低軌衛星概念股
    "2314.TW": "台揚", "6285.TW": "啟碁", "6271.TW": "同欣電",
    "3044.TW": "健鼎", "2439.TW": "美律",
}
