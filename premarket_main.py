import argparse
import json
import os
from datetime import datetime, date

import pandas as pd
import yfinance as yf

from src.portfolio import (
    load_portfolio, save_portfolio, get_individual_count,
    load_watchlist, save_watchlist, add_to_watchlist,
    update_high_prices, initialize_high_prices,
)
from src.data_loader import get_sp500_tickers, fetch_current_prices
from src.risk import check_position_limit
from src.premarket import generate_actions, VERSION
from src.sector_monitor import get_sector_summary, check_holdings_sector_exposure
from src.snapshot import load_snapshot, calculate_yearly_pnl, create_year_start_snapshot, save_snapshot
from src.momentum import rank_by_momentum, print_momentum_report, calculate_alpha_batch
from src.notifier import GmailNotifier


def fetch_ma200_prices(symbols):
    """取得多檔標的的 MA200 值

    Args:
        symbols: 股票代碼列表

    Returns:
        dict: {symbol: ma200_value}
    """
    ma200_prices = {}
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period="1y")
            if len(df) >= 200:
                ma200 = df['Close'].rolling(200).mean().iloc[-1]
                if not pd.isna(ma200):
                    ma200_prices[symbol] = round(ma200, 2)
        except Exception:
            pass
    return ma200_prices


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
    """產出盤前建議（動能策略 + 三層出場）"""
    os.makedirs("data", exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    # 1. 載入持倉
    portfolio = load_portfolio()
    positions = portfolio.get("positions", {})

    if not positions and portfolio.get("cash", 0) == 0:
        print("尚未建立投資組合。請先執行: python premarket_main.py --init")
        return

    print(f"=== 盤前分析 {date.today()} （三層出場策略）===\n")
    individual = get_individual_count(portfolio)
    print(f"持倉：{len(positions)} 檔（個股 {individual}/30），現金 ${portfolio.get('cash', 0):,.2f}\n")

    # 1.5 板塊相對強弱檢查
    print("正在檢查板塊相對強弱...")
    sector_summary = get_sector_summary(lookback_days=5)
    held_symbols = list(positions.keys())
    sector_exposure = check_holdings_sector_exposure(held_symbols)

    # 2. 組合候選池：SP500 前 100 + 白名單 + 持倉
    sp500 = get_sp500_tickers()
    watchlist = load_watchlist()
    wl_symbols = watchlist.get("symbols", [])
    all_tickers = list(dict.fromkeys(sp500 + wl_symbols + held_symbols))

    print(f"\n正在計算 {len(all_tickers)} 檔標的動能（SP500 前100 + 白名單 {len(wl_symbols)} 檔 + 持倉）...")

    # 3. 計算動能排名
    momentum_ranks = rank_by_momentum(all_tickers, period=21)

    # 4. 取得報價（動能前 20 名 + 持倉）
    top_symbols = [m["symbol"] for m in momentum_ranks[:20]]
    symbols_for_price = list(set(top_symbols + held_symbols))
    print(f"正在取得 {len(symbols_for_price)} 檔報價...")
    current_prices = fetch_current_prices(symbols_for_price)

    # 4.5 取得持倉的 MA200 資料（用於出場判斷）
    print(f"正在取得 {len(held_symbols)} 檔持倉的 MA200...")
    ma200_prices = fetch_ma200_prices(held_symbols)

    # 4.6 初始化/更新最高價追蹤
    initialize_high_prices(portfolio, current_prices)
    high_updated = update_high_prices(portfolio, current_prices)
    if high_updated:
        save_portfolio(portfolio)
        print("已更新持倉最高價記錄")

    # 4.7 計算 1 年超額報酬（ADD 候選 + 持倉，用於長期績效參考）
    add_candidates = [m["symbol"] for m in momentum_ranks[:10] if m["symbol"] not in positions]
    alpha_symbols = list(set(add_candidates + held_symbols))
    print(f"正在計算 {len(alpha_symbols)} 檔標的的 1 年超額報酬...")
    alpha_1y_map = calculate_alpha_batch(alpha_symbols)

    # 5. 產出 actions（使用動能排名 + 三層出場）
    actions = generate_actions(portfolio, current_prices, ma200_prices, momentum_ranks, alpha_1y_map)

    # 6. 計算投組總值
    total_value = portfolio.get("cash", 0)
    for symbol, pos in positions.items():
        price = current_prices.get(symbol, pos["avg_price"])
        total_value += price * pos["shares"]

    # 7. 載入年度快照並計算年度 P&L（用於儲存）
    current_year = date.today().year
    snapshot = load_snapshot(current_year)
    yearly_pnl = calculate_yearly_pnl(total_value, snapshot) if snapshot else None

    # 7.5 儲存 actions
    actions_output = {
        "date": str(date.today()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": VERSION,
        "portfolio_snapshot": {
            "total_value": round(total_value, 2),
            "cash": portfolio.get("cash", 0),
            "individual_count": individual,
            "yearly_pnl": yearly_pnl,
        },
        "sector_status": {
            "status": sector_summary["status"],
            "alerts": [a["message"] for a in sector_summary["alerts"]],
            "tech_ratio": sector_exposure["tech_ratio"],
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
    if yearly_pnl:
        pnl_sign = "+" if yearly_pnl["pnl_amount"] >= 0 else ""
        print(f"  {current_year}年度:  {pnl_sign}${yearly_pnl['pnl_amount']:>10,.2f} ({pnl_sign}{yearly_pnl['pnl_pct']:.1f}%)")
    else:
        print(f"  {current_year}年度:  (尚無快照，執行 --snapshot 建立)")
    print(f"{'='*60}")

    # 板塊健康狀態
    print(f"\n--- 板塊相對強弱 (過去5日) {sector_summary['status_emoji']} ---")
    if sector_summary.get("benchmark"):
        print(f"  大盤 SPY: {sector_summary['benchmark']['return']*100:+.1f}%")
    for sym, data in sector_summary.get("sectors", {}).items():
        rel = data["relative"]
        emoji = "🔴" if rel < -0.05 else ("🟡" if rel < 0 else "🟢")
        print(f"  {emoji} {data['name']:<6} {data['return']*100:+.1f}% (vs SPY: {rel*100:+.1f}%)")

    if sector_summary["alerts"]:
        print(f"\n  ⚠️  板塊警告：")
        for alert in sector_summary["alerts"]:
            print(f"     - {alert['message']}")

    if sector_exposure["warning"]:
        print(f"\n  🚨 注意：你的持股 {sector_exposure['tech_ratio']*100:.0f}% 是科技相關，而科技板塊正在走弱！")

    print()

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
            momentum = f"動能: {a['momentum']:+.1f}%" if a.get("momentum") is not None else ""
            alpha = a.get('alpha_1y')
            if alpha is not None:
                alpha_emoji = "🟢" if alpha > 0 else ("🟡" if alpha > -20 else "🔴")
                alpha_str = f"  1Y: {alpha:+.0f}% {alpha_emoji}"
            else:
                alpha_str = ""
            tag = "[core]" if a["source"] == "core_hold" else "      "
            print(f"  {tag} {a['symbol']:<6} {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {pnl}  {momentum}{alpha_str}")
        print()

    if adds:
        print("--- ADD (建議買入) ---")
        for a in adds:
            momentum_str = f"動能: +{a.get('momentum', 0):.1f}%"
            alpha = a.get('alpha_1y')
            if alpha is not None:
                alpha_emoji = "🟢" if alpha > 0 else ("🟡" if alpha > -20 else "🔴")
                alpha_str = f"  1Y vs SPY: {alpha:+.0f}% {alpha_emoji}"
            else:
                alpha_str = ""
            print(f"  [#{a.get('momentum_rank', '?')}] {a['symbol']:<6} 建議 {a['suggested_shares']} 股 @ ${a.get('current_price', 0):.2f}  {momentum_str}{alpha_str}")
            print(f"         原因: {a['reason']}")
        print()

    # 9. 發送 Email 通知
    notifier = GmailNotifier()
    if notifier.is_configured():
        print("正在發送 Email 通知...")
        if notifier.send_premarket_report(actions_output):
            print(f"Email 已發送至 {notifier.recipient}")
        else:
            print("Email 發送失敗，請檢查 .env 設定")

    print(f"\nActions 已儲存至: {actions_path}")
    print(f"確認執行請執行: python confirm_main.py {date.today()}")


def run_watch(symbols):
    """新增白名單標的"""
    wl = add_to_watchlist(symbols)
    print(f"白名單已更新：{wl['symbols']}")


def run_momentum(top_n: int = 20):
    """顯示動能排名"""
    portfolio = load_portfolio()
    positions = portfolio.get("positions", {})
    held_symbols = list(positions.keys())

    # 候選池
    sp500 = get_sp500_tickers()[:100]
    watchlist = load_watchlist()
    wl_symbols = watchlist.get("symbols", [])
    all_tickers = list(dict.fromkeys(sp500 + wl_symbols + held_symbols))

    print_momentum_report(all_tickers, period=21, top_n=top_n)

    # 標記持倉
    print("  持倉標記: ", end="")
    ranks = rank_by_momentum(all_tickers, period=21)
    held_in_top = [r for r in ranks[:top_n] if r["symbol"] in positions]
    if held_in_top:
        for r in held_in_top:
            print(f"{r['symbol']}(#{r['rank']}) ", end="")
    else:
        print("無持倉在前 {} 名".format(top_n), end="")
    print()


def run_snapshot(year: int = None):
    """建立年度快照"""
    if year is None:
        year = date.today().year

    portfolio = load_portfolio()
    if not portfolio.get("positions"):
        print("尚未建立投資組合。請先執行: python premarket_main.py --init")
        return

    # 檢查是否已存在
    existing = load_snapshot(year)
    if existing:
        print(f"警告: {year} 年快照已存在 (建立於 {existing.get('created_at')})")
        print(f"  年初總值: ${existing['total_value']:,.2f}")
        confirm = input("是否覆蓋？(y/n): ").strip().lower()
        if confirm != "y":
            print("已取消")
            return

    snapshot = create_year_start_snapshot(portfolio, year)
    path = save_snapshot(snapshot, year)

    print(f"\n{'='*50}")
    print(f"  {year} 年度快照已建立")
    print(f"{'='*50}")
    print(f"  基準日期: {snapshot['date']}")
    print(f"  年初總值: ${snapshot['total_value']:,.2f}")
    print(f"  現金:     ${snapshot['cash']:,.2f}")
    print(f"  持倉數:   {len(snapshot['positions'])} 檔")
    print(f"{'='*50}")
    print(f"\n已儲存至: {path}")


def main():
    parser = argparse.ArgumentParser(description="盤前建議系統（動能策略版）")
    parser.add_argument("--init", action="store_true", help="互動式建立初始投資組合")
    parser.add_argument("--watch", nargs="+", metavar="SYMBOL", help="新增白名單標的")
    parser.add_argument("--snapshot", nargs="?", const=date.today().year, type=int,
                        metavar="YEAR", help="建立年度快照（預設當年）")
    parser.add_argument("--momentum", nargs="?", const=20, type=int,
                        metavar="N", help="查看動能排名（預設前20名）")
    args = parser.parse_args()

    if args.init:
        run_init()
    elif args.watch:
        run_watch(args.watch)
    elif args.snapshot:
        run_snapshot(args.snapshot)
    elif args.momentum:
        run_momentum(args.momentum)
    else:
        run_premarket()


if __name__ == "__main__":
    main()
