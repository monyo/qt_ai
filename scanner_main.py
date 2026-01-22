# scanner_main.py
from src.data_loader import get_sp500_tickers, fetch_stock_data
from src.strategy import apply_double_factor_strategy

def run_scanner():
    print("正在獲取 S&P 500 清單...")
    tickers = get_sp500_tickers()
    # 測試時可以先取前 10 個，不然會跑很久
    tickers = tickers[:10] 
    
    pearls = []
    
    print(f"開始掃描 {len(tickers)} 支標的...")
    for symbol in tickers:
        try:
            # 替換掉點號（有些代碼如 BRK.B 在 Yahoo 會報錯，需轉成 BRK-B）
            symbol = symbol.replace('.', '-')
            df = fetch_stock_data(symbol, period="1y") # 掃描只需要一年數據
            
            if df.empty: continue
            
            df = apply_double_factor_strategy(df)
            
            # 檢查最後一天的信號是否為 1 (買入)
            if df['Signal'].iloc[-1] == 1:
                pearls.append(symbol)
                print(f"找到珍珠: {symbol}")
        except Exception as e:
            continue
            
    print("\n--- 今日掃描結果 ---")
    if pearls:
        print("以下標的目前符合雙因子買入條件:", ", ".join(pearls))
    else:
        print("今日無符合條件的標的。")

if __name__ == "__main__":
    run_scanner()
