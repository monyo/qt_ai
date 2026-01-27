import requests
from bs4 import BeautifulSoup

def get_sentiment_score(symbol):
    """
    獲取新聞標題並回傳簡短摘要 (Phase 3 預計對接 Gemini)
    """
    print(f"🔍 正在獲取 {symbol} 的即時新聞...")
    url = f"https://www.google.com/search?q={symbol}+stock+news&tbm=nws"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        headlines = [g.text for g in soup.find_all('div', dict(role='heading'))[:3]]
        
        if not headlines:
            return "無顯著新聞"
        return " | ".join(headlines)
    except:
        return "新聞抓取失敗"

# 整合進掃描器的範例：
# news = get_sentiment_score("GOOGL")
# print(f"最新消息：{news}")
