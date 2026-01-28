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
        symbol = a["symbol"]

        if action_type == "EXIT":
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            print(f"[{a['id']}] EXIT {symbol} | {a['shares']} 股 | P&L: {pnl}")
            print(f"    原因: {a['reason']}")
        elif action_type == "ADD":
            print(f"[{a['id']}] ADD  {symbol} | 建議 {a.get('suggested_shares', 0)} 股 @ ${a.get('current_price', 0):.2f}")
            print(f"    原因: {a['reason']}")

        choice = input("    確認執行？(y=確認 / n=跳過 / m=修改數量和價格): ").strip().lower()

        if choice == "y":
            a["status"] = "confirmed"
            a["confirm_date"] = str(date.today())
            confirmed_actions.append(a)
            print(f"    -> 已確認\n")

        elif choice == "m":
            if action_type == "ADD":
                shares_input = input(f"    實際買入股數 (預設 {a.get('suggested_shares', 0)}): ").strip()
                price_input = input(f"    實際成交價 (預設 {a.get('current_price', 0):.2f}): ").strip()
                a["actual_shares"] = int(shares_input) if shares_input else a.get("suggested_shares", 0)
                a["actual_price"] = float(price_input) if price_input else a.get("current_price", 0)
            elif action_type == "EXIT":
                shares_input = input(f"    實際賣出股數 (預設 {a.get('shares', 0)}): ").strip()
                price_input = input(f"    實際成交價 (預設 {a.get('current_price', 0):.2f}): ").strip()
                a["actual_shares"] = int(shares_input) if shares_input else a.get("shares", 0)
                a["actual_price"] = float(price_input) if price_input else a.get("current_price", 0)

            a["status"] = "confirmed"
            a["confirm_date"] = str(date.today())
            confirmed_actions.append(a)
            print(f"    -> 已確認（修改後）\n")

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
