import argparse
import json
import os
from datetime import date

from src.portfolio import load_portfolio, save_portfolio, apply_confirmed_actions
from src.risk import confirm_winner_cycle_exit, confirm_winner_cycle_reentry


def _ask_shares_price(prompt_sym, default_shares, default_price, action="買入"):
    s = input(f"  {action}股數 [{default_shares}]: ").strip()
    p = input(f"  成交價 [{default_price:.2f}]: ").strip()
    return (
        int(s) if s else default_shares,
        float(p) if p else default_price,
    )


def run_confirm(date_str):
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

    exits   = [a for a in pending if a["action"] == "EXIT"]
    rotates = [a for a in pending if a["action"] == "ROTATE"]
    adds    = [a for a in pending if a["action"] == "ADD"]

    today_str = str(date.today())
    confirmed_actions = []  # 傳給 apply_confirmed_actions 的合成清單

    # ── 賣出區 ──────────────────────────────────────────────
    sells_section = exits + rotates
    if sells_section:
        print("── 賣出 ──")

    for a in exits:
        sym = a["symbol"]
        pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
        tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
        print(f"EXIT {sym}{tranche_str}  {a['shares']}股  P&L: {pnl}")
        print(f"  原因: {a['reason']}")
        if input("  賣出了嗎？(y/n): ").strip().lower() == "y":
            sh, pr = _ask_shares_price(sym, a.get("shares", 0), a.get("current_price", 0), "賣出")
            a["actual_shares"] = sh
            a["actual_price"] = pr
            a["status"] = "confirmed"
            a["confirm_date"] = today_str
            confirmed_actions.append(a)
            if a.get("source") == "winner_cycle":
                confirm_winner_cycle_exit(sym, sh, pr, a.get("avg_price", 0))
                print("  -> 已確認（已加入特殊池觀察名單，等待回補）\n")
            else:
                print("  -> 已確認\n")
        else:
            a["status"] = "skipped"
            print("  -> 已跳過\n")

    for a in rotates:
        sym = a["sell_symbol"]
        print(f"賣出 {sym}  {a['sell_shares']}股  （動能差 {a.get('momentum_diff', 0):+.0f}%，P&L {a.get('sell_pnl_pct', 0):+.1f}%）")
        if input("  賣出了嗎？(y/n): ").strip().lower() == "y":
            sh, pr = _ask_shares_price(sym, a.get("sell_shares", 0), a.get("sell_price", 0), "賣出")
            a["actual_sell_shares"] = sh
            a["actual_sell_price"] = pr
            a["sell_status"] = "confirmed"
            # 合成 EXIT action 交給 apply_confirmed_actions
            confirmed_actions.append({
                "action": "EXIT",
                "status": "confirmed",
                "symbol": sym,
                "actual_shares": sh,
                "actual_price": pr,
                "current_price": pr,
                "confirm_date": today_str,
            })
            print("  -> 已確認\n")
        else:
            a["sell_status"] = "skipped"
            print("  -> 已跳過\n")

    # ── 買入區（依標的去重合併）──────────────────────────────
    # 建立 buy_groups: symbol → {suggested, price, add_action, rotate_actions}
    buy_groups = {}

    for a in adds:
        sym = a["symbol"]
        if sym not in buy_groups:
            buy_groups[sym] = {
                "suggested": 0, "price": a.get("current_price", 0),
                "add_action": None, "rotate_actions": [],
            }
        buy_groups[sym]["suggested"] += a.get("suggested_shares", 0) or 0
        buy_groups[sym]["add_action"] = a

    for a in rotates:
        sym = a["buy_symbol"]
        if sym not in buy_groups:
            buy_groups[sym] = {
                "suggested": 0, "price": a.get("buy_price", 0),
                "add_action": None, "rotate_actions": [],
            }
        buy_groups[sym]["suggested"] += a.get("buy_shares", 0)
        if not buy_groups[sym]["price"]:
            buy_groups[sym]["price"] = a.get("buy_price", 0)
        buy_groups[sym]["rotate_actions"].append(a)

    if buy_groups:
        print("── 買入 ──")

    for sym, grp in buy_groups.items():
        add_a = grp["add_action"]
        rot_as = grp["rotate_actions"]

        # 組合說明行
        parts = []
        if rot_as:
            rotate_shares = sum(r.get("buy_shares", 0) for r in rot_as)
            parts.append(f"ROTATE {rotate_shares}股")
        if add_a:
            add_shares = add_a.get("suggested_shares", 0) or 0
            if add_shares:
                tag = f"金字塔第{add_a['tranche_n']}批" if add_a.get("is_pyramid") else "ADD"
                parts.append(f"{tag} {add_shares}股")

        detail = " + ".join(parts) if parts else ""
        print(f"買入 {sym}  建議 {grp['suggested']} 股（{detail}）@ ${grp['price']:.2f}")

        if input("  買入了嗎？(y/n): ").strip().lower() == "y":
            sh, pr = _ask_shares_price(sym, grp["suggested"], grp["price"])

            # 合成 ADD action — 保留金字塔欄位
            synthetic = {
                "action": "ADD",
                "status": "confirmed",
                "symbol": sym,
                "actual_shares": sh,
                "actual_price": pr,
                "current_price": pr,
                "confirm_date": today_str,
            }
            if add_a and add_a.get("is_pyramid"):
                synthetic["is_pyramid"] = True
                synthetic["tranche_n"] = add_a.get("tranche_n")
                synthetic["direction"] = add_a.get("direction", "up")

            confirmed_actions.append(synthetic)

            # 標記原始 actions 狀態
            if add_a:
                add_a["actual_shares"] = sh
                add_a["actual_price"] = pr
                add_a["status"] = "confirmed"
                add_a["confirm_date"] = today_str
                if add_a.get("source") == "winner_cycle_reentry":
                    confirm_winner_cycle_reentry(sym)
                    print("  -> 已確認（已從特殊池觀察名單移除）\n")
                    continue
            for r in rot_as:
                r["actual_buy_shares"] = sh
                r["actual_buy_price"] = pr
                r["buy_status"] = "confirmed"

            print("  -> 已確認\n")
        else:
            if add_a:
                add_a["status"] = "skipped"
            for r in rot_as:
                r["buy_status"] = "skipped"
            print("  -> 已跳過\n")

    # 最終化 ROTATE 整體狀態
    for a in rotates:
        sell_done = a.get("sell_status") == "confirmed"
        buy_done  = a.get("buy_status") == "confirmed"
        if sell_done and buy_done:
            a["status"] = "confirmed"
        elif sell_done or buy_done:
            a["status"] = "partial"
        else:
            a["status"] = "skipped"
        a["confirm_date"] = today_str

    # 儲存 actions 檔
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(actions_data, f, indent=2, ensure_ascii=False)
    print(f"Actions 已更新至 {actions_path}")

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

    # ── 台股確認 ──────────────────────────────────────────────
    tw_pending = [a for a in actions_data.get("tw_actions", []) if a.get("status") == "pending"]
    if tw_pending:
        print(f"\n=== 🇹🇼 台股確認（{len(tw_pending)} 筆）===\n")
        portfolio = load_portfolio()
        tw_confirmed = []

        for a in tw_pending:
            if a["action"] == "TW_EXIT":
                print(f"TW_EXIT  {a['symbol']}  {a['shares']}股  P&L: {a.get('pnl_pct', 0):+.1f}%")
                print(f"  原因: {a['reason']}")
            elif a["action"] == "TW_ADD":
                print(f"TW_ADD   {a['symbol']} {a['name']}  建議 {a['suggested_shares']} 股 @ NT${a['current_price']:.0f}")
                print(f"  動能: {a['momentum']:+.1f}%  {a['trend_state']}")

            if input("  確認執行？(y/n): ").strip().lower() == "y":
                default_shares = a.get("suggested_shares") or a.get("shares", 0)
                default_price  = a.get("current_price", 0)
                s = input(f"  實際股數 [{default_shares}]: ").strip()
                p = input(f"  成交價 NT$ [{default_price:.0f}]: ").strip()
                a["actual_shares"] = int(s) if s else default_shares
                a["actual_price"]  = float(p) if p else default_price
                a["status"] = "confirmed"
                a["confirm_date"] = today_str
                tw_confirmed.append(a)
                print("  -> 已確認\n")
            else:
                a["status"] = "skipped"
                print("  -> 已跳過\n")

        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions_data, f, indent=2, ensure_ascii=False)

        if tw_confirmed:
            _apply_tw_actions(portfolio, tw_confirmed)
            save_portfolio(portfolio)
            print(f"🇹🇼 台股更新：現金 NT${portfolio.get('tw_cash', 0):,.0f}  持倉 {len(portfolio.get('tw_positions', {}))} 檔")


def _apply_tw_actions(portfolio, confirmed_actions):
    today_str = str(date.today())
    tw_positions = portfolio.setdefault("tw_positions", {})
    tw_transactions = portfolio.setdefault("tw_transactions", [])

    for a in confirmed_actions:
        sym    = a["symbol"]
        shares = a["actual_shares"]
        price  = a["actual_price"]

        if a["action"] == "TW_ADD":
            cost = shares * price
            if portfolio.get("tw_cash", 0) >= cost * 0.95:
                portfolio["tw_cash"] = round(portfolio.get("tw_cash", 0) - cost, 2)
                if sym in tw_positions:
                    pos = tw_positions[sym]
                    total_shares = pos["shares"] + shares
                    pos["avg_price"]  = round((pos["avg_price"] * pos["shares"] + price * shares) / total_shares, 2)
                    pos["cost_basis"] = round(pos.get("cost_basis", 0) + cost, 2)
                    pos["shares"]     = total_shares
                    pos["high_since_entry"] = max(pos.get("high_since_entry", price), price)
                else:
                    tw_positions[sym] = {
                        "shares": shares, "avg_price": round(price, 2),
                        "cost_basis": round(cost, 2), "first_entry": today_str,
                        "high_since_entry": price,
                    }
                tw_transactions.append({"date": today_str, "symbol": sym, "action": "BUY", "shares": shares, "price": price})
            else:
                print(f"  ⚠ 台幣現金不足，跳過 {sym}")

        elif a["action"] == "TW_EXIT":
            if sym in tw_positions:
                portfolio["tw_cash"] = round(portfolio.get("tw_cash", 0) + shares * price, 2)
                pos = tw_positions[sym]
                if shares >= pos["shares"]:
                    del tw_positions[sym]
                else:
                    pos["shares"] -= shares
                tw_transactions.append({"date": today_str, "symbol": sym, "action": "SELL", "shares": shares, "price": price})


def main():
    parser = argparse.ArgumentParser(description="確認盤前建議執行結果")
    parser.add_argument("date", help="要確認的日期 (YYYY-MM-DD 或 YYYYMMDD)")
    args = parser.parse_args()
    run_confirm(args.date)


if __name__ == "__main__":
    main()
