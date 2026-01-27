import time
import pandas as pd
from src.backtester import run_backtest
from src.data_loader import get_sp500_tickers, fetch_stock_data
from src.strategy import apply_double_factor_strategy
from src.visualizer import plot_result

def calculate_simple_return(df):
    """計算該策略的累計報酬率"""
    df['Daily_Return'] = df['Close'].pct_change()
    # 修正警告後的寫法
    df['Position'] = df['Signal'].replace(0, float('nan')).ffill().shift(1).fillna(0)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    final_return = (1 + df['Strategy_Return']).cumprod().iloc[-1] - 1
    return final_return * 100

def run_elite_scanner():
    print("🚀 啟動全美股精英掃描器...")
    tickers = get_sp500_tickers()
    
    tickers = tickers[:50] # 測試時建議先縮小範圍
    
    elite_pearls = []
    total = len(tickers)

    for i, symbol in enumerate(tickers):
        symbol = symbol.replace('.', '-')
        print(f"[{i+1}/{total}] 正在分析 {symbol}...", end='\r')

        try:
            df = fetch_stock_data(symbol, period="3y")
            if df.empty or len(df) < 100:
                continue

            # 1. 應用策略
            df = apply_double_factor_strategy(df)

            # 2. 檢查「今日」是否有買入訊號
            if df['Signal'].iloc[-1] == 1:
                # 3. 進行「歷史戰績」與「風險指標」計算
                df, metrics = run_backtest(df)

                if metrics["Return%"] > 0: # 只要歷史戰績是正的就入選
                    elite_pearls.append({
                        "Symbol": symbol,
                        **metrics,
                        "Price": round(df['Close'].iloc[-1], 2)
                    })
                    print(f"\n🌟 發現精英: {symbol} (報酬: {metrics["Return%"]}%)")

            # 4. 防封鎖延遲
            time.sleep(0.2)
        except: continue
 
    if not elite_pearls: return

    # 輸出表格
    res_df = pd.DataFrame(elite_pearls)
    print("\n" + "🏆 今日精英掃描報告 🏆")
    sorted_df = res_df.sort_values(by="Return%", ascending=False)
    sorted_df.to_csv(f"data/scan_result_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
    print(sorted_df)

    # 自動為前三名的精英畫圖
    top_3 = sorted_df.head(3)['Symbol'].tolist()
    for s in top_3:
        print(f"正在為珍珠 {s} 繪製回測圖...")
        # 繪圖時直接使用剛才掃描好的邏輯即可，不一定要重新 fetch
        df_to_plot = fetch_stock_data(s, period="3y")
        df_to_plot = apply_double_factor_strategy(df_to_plot)
        df_plot, _ = run_backtest(df_to_plot) # 確保畫圖前欄位齊全
        plot_result(df_to_plot, s)

def get_action_plan(elite_pearls, total_balance=10000):
    print("\n" + "📢 今日作戰指令 📢")
    print("-" * 50)
    for p in elite_pearls:
        # 每支分配 20% 資金
        allocation = total_balance * 0.2
        shares = int(allocation / p['Price'])
        
        print(f"【買入訊號】 {p['Symbol']}")
        print(f"   👉 建議買入數量: {shares} 股")
        print(f"   👉 預計投入金額: ${shares * p['Price']:.2f}")
        print(f"   👉 歷史勝率參考: {p['WinRate%']}%")
        print(f"   👉 風險警示 (MDD): {p['MDD%']}%")
        print("-" * 50)

def print_execution_plan(elite_pearls, total_cash=10000):
    """
    根據掃描結果，給出具體的買入建議與數量
    """
    if not elite_pearls: return

    print("\n" + "📢 實戰操作指令 (模擬資金: ${:,.0f}) 📢".format(total_cash))
    print("=" * 60)
    
    # 假設我們將資金平分給掃描到的前 5 名精英 (每支最多 20%)
    max_positions = 5
    per_stock_budget = total_cash / max_positions
    
    # 依照 Return% 排序挑選前幾名
    sorted_pearls = sorted(elite_pearls, key=lambda x: x['Return%'], reverse=True)[:max_positions]
    
    for p in sorted_pearls:
        shares = int(per_stock_budget / p['Price'])
        actual_cost = shares * p['Price']
        
        print(f"【買入】 {p['Symbol']:<6} | 建議數量: {shares:>3} 股 | 預計投入: ${actual_cost:>8.2f}")
        print(f"      📊 風險備註: 勝率 {p['WinRate%']}% | 歷史最大回撤 {p['MDD%']}%")
        print("-" * 60)
    
    print(f"💡 剩餘購買力 (預留現金): ${total_cash - sum([int(per_stock_budget/p['Price'])*p['Price'] for p in sorted_pearls]):.2f}")

if __name__ == "__main__":
    run_elite_scanner()
