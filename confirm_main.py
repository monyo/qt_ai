import argparse
import json
import os
from datetime import date

from src.portfolio import load_portfolio, save_portfolio, apply_confirmed_actions


def run_confirm(date_str):
    """讀取 actions 檔，讓使用者逐筆確認/跳過，更新 portfolio"""
    # 解析日期
    date_str = date_str.replace("-", "")
    actions_path = f"data/actions_{date_str}.json"

    if not os.path.exists(actions_path):
        print(f"找不到 {actions_path}")
        return

    with open(actions_path, "r", encoding="utf-8") as f:
        actions_data = json.load(f)

    actions = actions_data.get("actions", [])
    pending = [a for a in actions if a.get("status") == "pending"]

    if not pending:
        print("沒有待確認的 actions。")
        return

    print(f"=== 確認 {actions_data['date']} 的 Actions ===\n")
    print(f"共 {len(pending)} 筆待確認：\n")

    confirmed_actions = []

    for a in pending:
        action_type = a["action"]

        if action_type == "EXIT":
            symbol = a["symbol"]
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            tranche_str = f" | 第{a['tranche_n']}批" if a.get("tranche_n") else ""
            print(f"[{a['id']}] EXIT {symbol}{tranche_str} | {a['shares']} 股 | P&L: {pnl}")
            print(f"    原因: {a['reason']}")
        elif action_type == "ADD":
            symbol = a["symbol"]
            if a.get("is_pyramid"):
                direction_arrow = "↑" if a.get("direction") == "up" else "↓"
                print(f"[{a['id']}] ADD  {symbol} | 金字塔{direction_arrow}第{a['tranche_n']}批 | 建議 {a.get('suggested_shares', 0)} 股 @ ${a.get('current_price', 0):.2f}")
            else:
                print(f"[{a['id']}] ADD  {symbol} | 建議 {a.get('suggested_shares', 0)} 股 @ ${a.get('current_price', 0):.2f}")
            print(f"    原因: {a['reason']}")
        elif action_type == "ROTATE":
            sell_sym = a["sell_symbol"]
            buy_sym = a["buy_symbol"]
            print(f"[{a['id']}] ROTATE 賣 {sell_sym} {a['sell_shares']} 股 → 買 {buy_sym} {a['buy_shares']} 股")
            print(f"    原因: {a['reason']}")
        else:
            symbol = a.get("symbol", "?")
            print(f"[{a['id']}] {action_type} {symbol}")
            print(f"    原因: {a.get('reason', '')}")

        choice = input("    確認執行？(y=確認 / n=跳過): ").strip().lower()

        if choice == "y":
            if action_type == "ADD":
                default_shares = a.get("suggested_shares", 0)
                default_price = a.get("current_price", 0)
                shares_input = input(f"    實際買入股數 [{default_shares}]: ").strip()
                price_input = input(f"    實際成交價 [{default_price:.2f}]: ").strip()
                a["actual_shares"] = int(shares_input) if shares_input else default_shares
                a["actual_price"] = float(price_input) if price_input else default_price
            elif action_type == "EXIT":
                default_shares = a.get("shares", 0)
                default_price = a.get("current_price", 0)
                shares_input = input(f"    實際賣出股數 [{default_shares}]: ").strip()
                price_input = input(f"    實際成交價 [{default_price:.2f}]: ").strip()
                a["actual_shares"] = int(shares_input) if shares_input else default_shares
                a["actual_price"] = float(price_input) if price_input else default_price
            elif action_type == "ROTATE":
                default_sell_shares = a.get("sell_shares", 0)
                default_sell_price = a.get("sell_price", 0)
                default_buy_shares = a.get("buy_shares", 0)
                default_buy_price = a.get("buy_price", 0)
                sell_shares_input = input(f"    賣出 {a['sell_symbol']} 股數 [{default_sell_shares}]: ").strip()
                sell_price_input = input(f"    賣出成交價 [{default_sell_price:.2f}]: ").strip()
                buy_shares_input = input(f"    買入 {a['buy_symbol']} 股數 [{default_buy_shares}]: ").strip()
                buy_price_input = input(f"    買入成交價 [{default_buy_price:.2f}]: ").strip()
                a["actual_sell_shares"] = int(sell_shares_input) if sell_shares_input else default_sell_shares
                a["actual_sell_price"] = float(sell_price_input) if sell_price_input else default_sell_price
                a["actual_buy_shares"] = int(buy_shares_input) if buy_shares_input else default_buy_shares
                a["actual_buy_price"] = float(buy_price_input) if buy_price_input else default_buy_price

            a["status"] = "confirmed"
            a["confirm_date"] = str(date.today())
            confirmed_actions.append(a)
            print(f"    -> 已確認\n")

        else:
            a["status"] = "skipped"
            print(f"    -> 已跳過\n")

    # 更新 actions 檔
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(actions_data, f, indent=2, ensure_ascii=False)
    print(f"Actions 狀態已更新至 {actions_path}")

    # 套用到 portfolio
    if confirmed_actions:
        portfolio = load_portfolio()
        apply_confirmed_actions(portfolio, confirmed_actions)
        save_portfolio(portfolio)

        print(f"\n投資組合已更新：")
        print(f"  現金: ${portfolio.get('cash', 0):,.2f}")
        print(f"  持倉: {len(portfolio.get('positions', {}))} 檔")
        print(f"  交易紀錄: {len(portfolio.get('transactions', []))} 筆")
    else:
        print("\n無確認的 actions，投資組合未變更。")


def main():
    parser = argparse.ArgumentParser(description="確認盤前建議執行結果")
    parser.add_argument("date", help="要確認的日期 (YYYY-MM-DD 或 YYYYMMDD)")
    args = parser.parse_args()

    run_confirm(args.date)


if __name__ == "__main__":
    main()
