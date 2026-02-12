"""台股全市場掃描器

掃描台股上市櫃公司，用成交量過濾出高流動性標的。
"""
import json
import os
import requests
import pandas as pd
import yfinance as yf
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed


CACHE_FILE = "data/tw_liquid_stocks.json"
MIN_VOLUME = 1000  # 最低日均成交量（張）


def fetch_tw_stock_list():
    """從 FinMind 取得台股清單"""
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {"dataset": "TaiwanStockInfo"}

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if data.get("status") == 200:
            df = pd.DataFrame(data["data"])
            # 只取 4 碼股票（排除 ETF、權證等）
            df = df[df["stock_id"].str.match(r'^\d{4}$', na=False)]
            return df
    except Exception as e:
        print(f"取得台股清單失敗: {e}")
    return pd.DataFrame()


def get_volume_batch(stock_ids, stock_type_map, max_workers=20):
    """批次取得成交量

    Args:
        stock_ids: 股票代碼列表
        stock_type_map: {stock_id: 'twse'/'tpex'} 類型對照
        max_workers: 最大並行數

    Returns:
        list of (stock_id, symbol, volume, name)
    """
    def get_one(stock_id):
        stock_type = stock_type_map.get(stock_id, "twse")
        # 上市用 .TW，上櫃用 .TWO
        suffix = ".TW" if stock_type == "twse" else ".TWO"
        symbol = f"{stock_id}{suffix}"

        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if not hist.empty and len(hist) >= 3:
                vol = hist["Volume"].mean() / 1000  # 轉換成張
                return stock_id, symbol, vol
        except:
            pass

        # 上櫃股票有時候用 .TW 也可以
        if suffix == ".TWO":
            try:
                symbol = f"{stock_id}.TW"
                hist = yf.Ticker(symbol).history(period="5d")
                if not hist.empty and len(hist) >= 3:
                    vol = hist["Volume"].mean() / 1000
                    return stock_id, symbol, vol
            except:
                pass

        return stock_id, None, None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_one, sid): sid for sid in stock_ids}
        for future in as_completed(futures):
            stock_id, symbol, vol = future.result()
            if symbol and vol is not None:
                results.append((stock_id, symbol, vol))

    return results


def scan_tw_market(min_volume=MIN_VOLUME, force_refresh=False):
    """掃描台股市場，回傳高流動性股票清單

    Args:
        min_volume: 最低日均成交量（張）
        force_refresh: 是否強制重新掃描

    Returns:
        list of {"symbol": str, "stock_id": str, "name": str, "volume": float}
    """
    # 檢查快取
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            # 快取有效期 7 天
            cache_date = cache.get("date", "")
            if cache_date:
                from datetime import datetime
                cache_dt = datetime.strptime(cache_date, "%Y-%m-%d").date()
                if (date.today() - cache_dt).days < 7:
                    print(f"使用快取（{cache_date}），共 {len(cache['stocks'])} 檔")
                    return cache["stocks"]

    print("正在掃描台股市場...")

    # 1. 取得股票清單
    df = fetch_tw_stock_list()
    if df.empty:
        print("無法取得台股清單")
        return []

    print(f"取得 {len(df)} 檔台股")

    # 建立類型對照表
    stock_type_map = dict(zip(df["stock_id"], df["type"]))
    stock_name_map = dict(zip(df["stock_id"], df["stock_name"]))

    # 2. 批次取得成交量
    print(f"正在取得成交量資料（這可能需要幾分鐘）...")
    all_ids = df["stock_id"].tolist()

    # 分批處理避免過載
    batch_size = 100
    all_results = []

    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i+batch_size]
        print(f"  處理 {i+1}-{min(i+batch_size, len(all_ids))}/{len(all_ids)}...")
        results = get_volume_batch(batch, stock_type_map)
        all_results.extend(results)

    # 3. 過濾高流動性股票（去重：同一 stock_id 只保留一筆）
    seen = set()
    liquid_stocks = []
    for stock_id, symbol, vol in all_results:
        if vol >= min_volume and stock_id not in seen:
            seen.add(stock_id)
            liquid_stocks.append({
                "symbol": symbol,
                "stock_id": stock_id,
                "name": stock_name_map.get(stock_id, ""),
                "volume": round(vol, 0),
            })

    # 按成交量排序
    liquid_stocks.sort(key=lambda x: x["volume"], reverse=True)

    print(f"找到 {len(liquid_stocks)} 檔高流動性股票（日均量 > {min_volume} 張）")

    # 4. 儲存快取
    os.makedirs("data", exist_ok=True)
    cache = {
        "date": str(date.today()),
        "min_volume": min_volume,
        "total_scanned": len(all_ids),
        "stocks": liquid_stocks,
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"已儲存至 {CACHE_FILE}")

    return liquid_stocks


def get_tw_liquid_tickers(min_volume=MIN_VOLUME):
    """取得高流動性台股代碼列表（給動能計算用）

    Returns:
        list of yfinance symbols (e.g., ["2330.TW", "2317.TW", ...])
    """
    stocks = scan_tw_market(min_volume=min_volume)
    return [s["symbol"] for s in stocks]


if __name__ == "__main__":
    # 測試
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="強制重新掃描")
    parser.add_argument("--volume", type=int, default=1000, help="最低成交量（張）")
    args = parser.parse_args()

    stocks = scan_tw_market(min_volume=args.volume, force_refresh=args.refresh)

    print(f"\n=== 高流動性台股（日均量 > {args.volume} 張）===")
    print(f"共 {len(stocks)} 檔\n")

    print("前 20 名：")
    for s in stocks[:20]:
        print(f"  {s['symbol']:<10} {s['name']:<8} {s['volume']:>10,.0f} 張")
