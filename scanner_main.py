import os
import time
import pandas as pd
from src.data_loader import get_sp500_tickers, fetch_stock_data
from src.strategy import apply_double_factor_strategy
from src.backtester import run_backtest
from src.ai_analyst import fetch_latest_news_yf, analyze_sentiment_batch_with_gemini
from src.visualizer import plot_result

def calculate_simple_return(df):
    """計算該策略的累計報酬率"""
    df['Daily_Return'] = df['Close'].pct_change()
    # 修正警告後的寫法
    df['Position'] = df['Signal'].replace(0, float('nan')).ffill().shift(1).fillna(0)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    final_return = (1 + df['Strategy_Return']).cumprod().iloc[-1] - 1
    return final_return * 100

def run_elite_scanner(top_n_for_ai: int = 10, lookback_hours: int = 24):
    os.makedirs("data", exist_ok=True)

    print("🚀 啟動全美股精英掃描器...")
    tickers = get_sp500_tickers()

    tickers = tickers[:50]  # 測試時建議先縮小範圍

    elite_pearls = []
    cache_for_plot = {}  # symbol -> df_plot（含累積績效欄位）
    total = len(tickers)

    for i, symbol in enumerate(tickers):
        symbol = symbol.replace('.', '-')
        print(f"[{i+1}/{total}] 正在分析 {symbol}...", end='\r')

        try:
            df = fetch_stock_data(symbol, period="3y")
            if df is None or df.empty or len(df) < 100:
                continue

            # 1) 應用策略（產出 Signal）
            df = apply_double_factor_strategy(df)

            # 2) 先跑回測：把「事件訊號」轉成 long-only Position，並算出績效欄位
            df_plot, metrics = run_backtest(df)

            # 3) 檢查是否剛出現買入事件（用 Entry_Signal；沒有就用 Position 變化）
            if 'Entry_Signal' in df_plot.columns:
                is_today_entry = (df_plot['Entry_Signal'].iloc[-1] == 1)
            else:
                is_today_entry = (df_plot['Position'].diff().fillna(0).iloc[-1] > 0)

            if not is_today_entry:
                continue

            # 4) 入選條件：歷史報酬為正
            if metrics.get("Return%", -999) <= 0:
                continue

            elite_pearls.append({
                "Symbol": symbol,
                **metrics,
                "Price": round(float(df_plot['Close'].iloc[-1]), 2)
            })

            # cache：後面 Top3 畫圖不用再 fetch
            cache_for_plot[symbol] = df_plot

            print(f"\n🌟 發現精英: {symbol} (報酬: {metrics.get('Return%', 'NA')}%)")

            # 防封鎖延遲（抓資料端）
            time.sleep(0.2)

        except Exception as e:
            print(f"\n⚠️ {symbol} 分析失敗：{type(e).__name__}: {e}")
            continue

    if not elite_pearls:
        print("\n😶 今日沒有找到符合條件的標的。")
        return None

    # --- 輸出初選表格（先排序） ---
    res_df = pd.DataFrame(elite_pearls)
    sorted_df = res_df.sort_values(by="Return%", ascending=False)

    print("\n🏆 今日精英掃描報告 🏆")
    scan_path = f"data/scan_result_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    sorted_df.to_csv(scan_path, index=False)
    print(sorted_df)

    # --- AI 介入環節：只對前 N 名做一次 Batch（超省額度） ---
    top_n_for_ai = max(0, int(top_n_for_ai))
    ai_candidates = sorted_df.head(top_n_for_ai) if top_n_for_ai > 0 else sorted_df.head(0)

    symbol_to_headlines = {}
    if len(ai_candidates) > 0:
        print(f"\n📰 正在為前 {len(ai_candidates)} 名抓取 {lookback_hours} 小時內新聞（yfinance）...")
        for _, row in ai_candidates.iterrows():
            sym = row["Symbol"]
            try:
                symbol_to_headlines[sym] = fetch_latest_news_yf(sym, lookback_hours=lookback_hours, limit=5)
            except Exception as e:
                symbol_to_headlines[sym] = [f"新聞取得失敗: {type(e).__name__}: {e}"]

        print(f"🧠 將 {len(symbol_to_headlines)} 檔標的送交 Gemini 批次審核...")
        ai_map = analyze_sentiment_batch_with_gemini(symbol_to_headlines)
    else:
        ai_map = {}

    # --- 組裝最終結果（未送 AI 的標的，維持中立/未審核） ---
    final_recommendations = []

    for _, row in sorted_df.iterrows():
        sym = row["Symbol"]
        ai = ai_map.get(sym.upper())

        if ai is None:
            sentiment_score = 0.0
            ai_reason = "未進行 AI 審核（節省額度）"
        else:
            sentiment_score = float(ai.get("score", 0.0))
            ai_reason = ai.get("reason", "無原因")

        if sentiment_score > 0.3:
            action = "✅ 強烈買入 (技術與消息雙重利多)"
        elif sentiment_score < -0.3:
            action = "❌ 暫緩執行 (注意利空)"
        else:
            action = "⚖️ 技術面買入 (消息面中立/無消息)"

        rec = row.to_dict()
        rec.update({
            "Sentiment": sentiment_score,
            "Action": action,
            "Reason": ai_reason
        })
        final_recommendations.append(rec)

    # --- 輸出最終戰報 ---
    final_df = pd.DataFrame(final_recommendations)
    sorted_final_df = final_df.sort_values(by="Return%", ascending=False)

    print("\n🛡️ AI 終極戰術板 🛡️")
    final_path = f"data/final_result_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    sorted_final_df.to_csv(final_path, index=False)
    print(sorted_final_df[['Symbol', 'Return%', 'MDD%', 'Sentiment', 'Action', 'Reason']])

    # --- 自動為前三名畫圖（用 cache，不重抓） ---
    top_3 = sorted_final_df.head(3)['Symbol'].tolist()
    for s in top_3:
        print(f"正在為珍珠 {s} 繪製回測圖...")

        df_plot = cache_for_plot.get(s)
        if df_plot is None:
            try:
                df_to_plot = fetch_stock_data(s, period="3y")
                df_to_plot = apply_double_factor_strategy(df_to_plot)
                df_plot, _ = run_backtest(df_to_plot)
            except Exception as e:
                print(f"⚠️ {s} 畫圖資料準備失敗：{type(e).__name__}: {e}")
                continue

        try:
            plot_result(df_plot, s)
        except Exception as e:
            print(f"⚠️ {s} 畫圖失敗：{type(e).__name__}: {e}")

    return sorted_final_df

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
