import argparse
import json
import os
from datetime import datetime, date

from src.portfolio import (
    load_portfolio, save_portfolio, get_individual_count,
    load_watchlist, save_watchlist, add_to_watchlist,
)
from src.data_loader import get_sp500_tickers, fetch_current_prices
from src.risk import check_position_limit
from src.premarket import generate_actions, VERSION
from src.ai_analyst import fetch_latest_news_yf, analyze_sentiment_batch_with_gemini
from scanner_main import scan_candidates


def run_init():
    """互動式建立初始 portfolio.json"""
    print("=== 初始化投資組合 ===\n")

    cash = float(input("請輸入可用現金 (USD): "))
    portfolio = {
        "cash": cash,
        "updated": str(date.today()),
        "positions": {},
        "transactions": [],
    }

    # VOO 核心持倉
    has_voo = input("\n是否持有 VOO？(y/n): ").strip().lower()
    if has_voo == "y":
        shares = int(input("  VOO 股數: "))
        avg_price = float(input("  VOO 平均成本: "))
        entry_date = input("  VOO 首次買入日期 (YYYY-MM-DD): ").strip()
        portfolio["positions"]["VOO"] = {
            "shares": shares,
            "avg_price": avg_price,
            "cost_basis": round(avg_price * shares, 2),
            "first_entry": entry_date,
            "core": True,
        }
        portfolio["transactions"].append({
            "date": entry_date, "symbol": "VOO", "action": "ADD",
            "shares": shares, "price": avg_price,
        })

    # 其他個股
    print("\n輸入現有個股持倉（輸入空白 symbol 結束）：")
    while True:
        symbol = input("\n  Symbol (留空結束): ").strip().upper()
        if not symbol:
            break
        shares = int(input(f"  {symbol} 股數: "))
        avg_price = float(input(f"  {symbol} 平均成本: "))
        entry_date = input(f"  {symbol} 首次買入日期 (YYYY-MM-DD): ").strip()

        portfolio["positions"][symbol] = {
            "shares": shares,
            "avg_price": avg_price,
            "cost_basis": round(avg_price * shares, 2),
            "first_entry": entry_date,
            "core": False,
        }
        portfolio["transactions"].append({
            "date": entry_date, "symbol": symbol, "action": "ADD",
            "shares": shares, "price": avg_price,
        })

    save_portfolio(portfolio)
    individual = get_individual_count(portfolio)
    print(f"\n投資組合已建立：{len(portfolio['positions'])} 檔持倉（個股 {individual} 檔），現金 ${cash:,.2f}")
    print(f"已儲存至 data/portfolio.json")


def run_premarket():
    """產出盤前建議"""
    os.makedirs("data", exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    # 1. 載入持倉
    portfolio = load_portfolio()
    positions = portfolio.get("positions", {})

    if not positions and portfolio.get("cash", 0) == 0:
        print("尚未建立投資組合。請先執行: python premarket_main.py --init")
        return

    print(f"=== 盤前分析 {date.today()} ===\n")
    individual = get_individual_count(portfolio)
    print(f"持倉：{len(positions)} 檔（個股 {individual}/30），現金 ${portfolio.get('cash', 0):,.2f}\n")

    # 2. 取得所有持倉的最新報價
    held_symbols = list(positions.keys())
    print(f"正在取得 {len(held_symbols)} 檔持倉報價...")
    current_prices = fetch_current_prices(held_symbols)

    # 3. 組合候選池：SP500 前 50 + 白名單
    available_slots = check_position_limit(portfolio)
    candidates = []
    cache_for_plot = {}

    if available_slots > 0:
        # 取得 S&P 500 + watchlist
        sp500 = get_sp500_tickers()[:50]
        watchlist = load_watchlist()
        wl_symbols = watchlist.get("symbols", [])

        # 合併去重（也包含已持有的，以取得策略訊號）
        all_tickers = list(dict.fromkeys(sp500 + wl_symbols + held_symbols))

        print(f"\n正在掃描 {len(all_tickers)} 檔標的（SP500 前50 + 白名單 {len(wl_symbols)} 檔 + 持倉）...")
        candidates, cache_for_plot = scan_candidates(all_tickers)

        # 標記 source
        wl_set = set(s.upper() for s in wl_symbols)
        for c in candidates:
            if c["Symbol"].upper() in wl_set:
                c["source"] = "watchlist"
            else:
                c["source"] = "scanner"

        # 更新 current_prices
        for c in candidates:
            if c["Symbol"] not in current_prices and c.get("Price"):
                current_prices[c["Symbol"]] = c["Price"]
    else:
        # 持倉已滿，只掃描已持有的標的（取得策略訊號）
        print(f"\n個股已達 30 檔上限，僅檢查持倉訊號...")
        candidates, cache_for_plot = scan_candidates(held_symbols)

    # 4. AI 情緒分析（對新候選）
    sentiment_scores = {}
    new_buy_candidates = [
        c for c in candidates
        if c.get("has_today_signal")
        and c.get("Return%", -999) > 0
        and c["Symbol"] not in positions
    ]

    if new_buy_candidates:
        top_for_ai = new_buy_candidates[:10]
        symbol_to_headlines = {}
        print(f"\n正在為 {len(top_for_ai)} 檔候選抓取新聞...")
        for c in top_for_ai:
            sym = c["Symbol"]
            try:
                symbol_to_headlines[sym] = fetch_latest_news_yf(sym, lookback_hours=24, limit=5)
            except Exception as e:
                symbol_to_headlines[sym] = [f"新聞取得失敗: {e}"]

        if symbol_to_headlines:
            print(f"送交 Gemini 批次審核 {len(symbol_to_headlines)} 檔...")
            ai_map = analyze_sentiment_batch_with_gemini(symbol_to_headlines)
            for sym, result in ai_map.items():
                sentiment_scores[sym] = {
                    "score": float(result.get("score", 0.0)),
                    "reason": result.get("reason", ""),
                }

    # 5. 產出 actions
    actions = generate_actions(portfolio, current_prices, candidates, sentiment_scores)

    # 6. 計算投組總值
    total_value = portfolio.get("cash", 0)
    for symbol, pos in positions.items():
        price = current_prices.get(symbol, pos["avg_price"])
        total_value += price * pos["shares"]

    # 7. 儲存 actions
    actions_output = {
        "date": str(date.today()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": VERSION,
        "portfolio_snapshot": {
            "total_value": round(total_value, 2),
            "cash": portfolio.get("cash", 0),
            "individual_count": individual,
        },
        "actions": actions,
    }

    actions_path = f"data/actions_{today_str}.json"
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(actions_output, f, indent=2, ensure_ascii=False)

    # 8. 印出盤前報告
    print(f"\n{'='*60}")
    print(f"  盤前報告 {date.today()}  |  版本 {VERSION}")
    print(f"{'='*60}")
    print(f"  投組總值: ${total_value:>12,.2f}")
    print(f"  現金:     ${portfolio.get('cash', 0):>12,.2f}")
    print(f"  個股:     {individual}/30 檔")
    print(f"{'='*60}\n")

    # 分類印出
    exits = [a for a in actions if a["action"] == "EXIT"]
    holds = [a for a in actions if a["action"] == "HOLD"]
    adds = [a for a in actions if a["action"] == "ADD"]

    if exits:
        print("--- EXIT (建議出場) ---")
        for a in exits:
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            print(f"  [{a['source']}] {a['symbol']:<6} {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {pnl}")
            print(f"         原因: {a['reason']}")
        print()

    if holds:
        print("--- HOLD (繼續持有) ---")
        for a in holds:
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            tag = "[core]" if a["source"] == "core_hold" else "      "
            print(f"  {tag} {a['symbol']:<6} {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {pnl}")
        print()

    if adds:
        print("--- ADD (建議買入) ---")
        for a in adds:
            sentiment_str = f"  情緒: {a.get('sentiment', 0):.1f}" if a.get("sentiment") else ""
            bt_str = f"  回測: {a.get('backtest_return_pct', 0):.1f}%" if a.get("backtest_return_pct") else ""
            print(f"  [{a['source']}] {a['symbol']:<6} 建議 {a['suggested_shares']} 股 @ ${a.get('current_price', 0):.2f}{sentiment_str}{bt_str}")
            print(f"         原因: {a['reason']}")
        print()

    print(f"Actions 已儲存至: {actions_path}")
    print(f"確認執行請執行: python confirm_main.py {date.today()}")


def run_watch(symbols):
    """新增白名單標的"""
    wl = add_to_watchlist(symbols)
    print(f"白名單已更新：{wl['symbols']}")


def main():
    parser = argparse.ArgumentParser(description="盤前建議系統")
    parser.add_argument("--init", action="store_true", help="互動式建立初始投資組合")
    parser.add_argument("--watch", nargs="+", metavar="SYMBOL", help="新增白名單標的")
    args = parser.parse_args()

    if args.init:
        run_init()
    elif args.watch:
        run_watch(args.watch)
    else:
        run_premarket()


if __name__ == "__main__":
    main()
