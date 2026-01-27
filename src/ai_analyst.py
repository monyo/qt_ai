import os
import requests
from bs4 import BeautifulSoup
import datetime
import time
import json
import re
from google import genai
from google.api_core import exceptions
from dotenv import load_dotenv

def fetch_latest_news_yf(symbol, lookback_hours=24, limit=5):
    """
    用 yfinance 取新聞，回傳最近 lookback_hours 內的 title list
    """
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", None) or []
    except Exception as e:
        return [f"新聞取得失敗: {type(e).__name__}: {e}"]

    now = int(time.time())
    cutoff = now - lookback_hours * 3600

    titles = []
    for item in news:
        ts = item.get("providerPublishTime")
        title = item.get("title")
        if not title:
            continue
        # 若沒時間戳，就保守收下，但你也可以選擇丟掉
        if ts is None or ts >= cutoff:
            titles.append(title)

    return titles[:limit] if titles else ["查無顯著即時新聞"]

def fetch_latest_news(symbol):
    """抓取該標的過去 24 小時的 Google 新聞標題"""
    print(f"🔍 正在獵取 {symbol} 的即時消息...")

    # 搜尋「股票代碼 + stock news」，設定為最近一小時或一天的結果
    url = f"https://www.google.com/search?q={symbol}+stock+news&tbm=nws&hl=en" # 強制用英文

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }

    try:
        import time
        import random
        # 增加一點隨機延遲，避免被 Google 發現是機器人
        time.sleep(random.uniform(1, 3))

        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 嘗試多種可能的標題標籤 (Google News 結構多變)
        headlines = []

        # 結構 1: 常見的 div 標題
        for item in soup.find_all('div', attrs={'role': 'heading'}):
            headlines.append(item.text)

        # 結構 2: 備用選擇器 (針對新版 Google News)
        if not headlines:
            for item in soup.select('div.n0Odbb, div.mCBkyc'):
                headlines.append(item.get_text())

        # 結構 3: 傳統新聞連結標題
        if not headlines:
            for item in soup.find_all('h3'):
                headlines.append(item.text)

        return headlines[:5] if headlines else ["查無顯著即時新聞"]
    except Exception as e:
        return [f"新聞抓取發生技術錯誤: {str(e)}"]

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def analyze_sentiment_batch_with_gemini(symbol_to_headlines, model="gemini-2.0-flash", max_retries=3):
    """
    一次把多檔(symbol->headlines)送給 Gemini，回傳 dict:
    { "AAPL": {"score": 0.2, "reason": "..."}, ... }
    """
    if not client:
        return {sym: {"score": 0.0, "reason": "API Key 未設定"} for sym in symbol_to_headlines}

    # 把輸入整理乾淨（避免 token 浪費）
    payload = []
    for sym, headlines in symbol_to_headlines.items():
        if not headlines or "查無顯著即時新聞" in str(headlines):
            payload.append({"symbol": sym, "headlines": []})
        else:
            payload.append({"symbol": sym, "headlines": headlines[:5]})

    prompt = f"""
你是專業金融新聞分析師。請只根據每檔股票的新聞標題，判斷 24~72 小時的情緒。
分數範圍：-1.0(極度利空) 到 1.0(極度利多)

輸入 JSON（每檔最多5條標題）：
{json.dumps(payload, ensure_ascii=False)}

請嚴格輸出 JSON（不要有多餘文字），格式如下：
{{
  "results": [
    {{"symbol":"AAPL","score":0.2,"reason":"不超過30字"}},
    ...
  ]
}}
"""

    backoff = 5
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt
            )

            content = (resp.text or "").strip()

            # 抽出 JSON 區塊（避免模型多講話）
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in model output")

            obj = json.loads(m.group())
            results = obj.get("results", [])

            out = {}
            for r in results:
                sym = str(r.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                try:
                    score = float(r.get("score", 0.0))
                except Exception:
                    score = 0.0
                # clamp
                score = max(-1.0, min(1.0, score))
                reason = str(r.get("reason", "")).strip()[:120]
                out[sym] = {"score": score, "reason": reason or "無原因"}

            # 對沒回到的 symbol 補中立（避免 KeyError）
            for sym in symbol_to_headlines:
                sym_u = sym.upper()
                if sym_u not in out:
                    out[sym_u] = {"score": 0.0, "reason": "AI 無回覆(降級中立)"}

            return out

        except Exception as e:
            msg = str(e).lower()

            # 配額/帳務：重試通常沒用，直接降級
            if ("check your plan" in msg) or ("billing" in msg) or ("quota" in msg):
                return {sym.upper(): {"score": 0.0, "reason": "⚠️ AI 額度/帳務限制(降級中立)"} for sym in symbol_to_headlines}

            # 429：用退避重試
            if ("429" in msg) or ("resource_exhausted" in msg):
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return {sym.upper(): {"score": 0.0, "reason": "⚠️ AI 觸發頻率限制(降級中立)"} for sym in symbol_to_headlines}

            # 其他錯誤：降級
            return {sym.upper(): {"score": 0.0, "reason": f"⚠️ AI 錯誤降級: {str(e)[:60]}"} for sym in symbol_to_headlines}

    return {sym.upper(): {"score": 0.0, "reason": "分析流程異常(降級中立)"} for sym in symbol_to_headlines}

def analyze_sentiment_with_gemini(symbol, headlines):
    """
    強化版 Gemini 情緒分析：整合自動重試與錯誤降級機制
    """
    if not client:
        return 0.0, "API Key 未設定"

    # 如果沒有新聞，直接回傳中立，不浪費 API 額度
    if not headlines or "查無顯著即時新聞" in str(headlines):
        return 0.0, "⚖️ 無即時新聞 (依技術面決策)"

    prompt = f"""
你是專業金融新聞分析師。只根據標題判斷短期情緒（24~72h）。
標的: {symbol}
標題: {headlines}

嚴格輸出 JSON:
{{"score": -1.0到1.0的小數, "reason": "不超過30字"}}
"""

    max_retries = 3  # 最大重試次數
    backoff = 5

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            # 使用更強健的 JSON 提取
            content = (response.text or "").strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                return 0.0, "AI 回傳格式不符"
            result = json.loads(m.group())
            return float(result.get("score", 0.0)), result.get("reason", "無原因")

        except Exception as e:
            msg = str(e)

            # ✅ 這種是「額度/帳務」：重試通常沒用，直接降級
            if ("check your plan" in msg.lower()) or ("billing" in msg.lower()) or ("quota" in msg.lower()):
                return 0.0, "⚠️ AI 額度/帳務限制 (降級為中立)"

            # ✅ 這種才比較像「太快」：用 backoff 重試
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return 0.0, "⚠️ AI 觸發頻率限制 (降級為中立)"

            return 0.0, f"AI 分析降級: {msg[:80]}"

    return 0.0, "分析流程異常"
