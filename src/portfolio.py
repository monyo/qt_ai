import json
import os
from datetime import date


PORTFOLIO_PATH = os.path.join("data", "portfolio.json")
WATCHLIST_PATH = os.path.join("data", "watchlist.json")


def load_portfolio(path=PORTFOLIO_PATH):
    """讀取持倉狀態，不存在則回傳空投組"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cash": 0, "updated": "", "positions": {}, "transactions": []}


def save_portfolio(portfolio, path=PORTFOLIO_PATH):
    """寫入持倉狀態"""
    portfolio["updated"] = str(date.today())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def get_individual_count(portfolio):
    """回傳非 core 持倉數量"""
    return sum(
        1 for pos in portfolio["positions"].values()
        if not pos.get("core", False)
    )


def calc_avg_price(current_avg, current_shares, new_price, new_shares):
    """加權平均成本計算"""
    total_cost = current_avg * current_shares + new_price * new_shares
    total_shares = current_shares + new_shares
    if total_shares == 0:
        return 0.0
    return total_cost / total_shares


def apply_confirmed_actions(portfolio, confirmed_actions):
    """套用 confirmed actions，更新 positions + 追加 transactions"""
    for action in confirmed_actions:
        if action.get("status") != "confirmed":
            continue

        symbol = action["symbol"]
        act = action["action"]
        tx_date = action.get("confirm_date", str(date.today()))

        if act == "ADD":
            shares = action.get("actual_shares", action.get("suggested_shares", 0))
            price = action.get("actual_price", action["current_price"])
            if shares <= 0:
                continue

            if symbol in portfolio["positions"]:
                pos = portfolio["positions"][symbol]
                pos["avg_price"] = calc_avg_price(
                    pos["avg_price"], pos["shares"], price, shares
                )
                pos["shares"] += shares
                pos["cost_basis"] = pos["avg_price"] * pos["shares"]
            else:
                cost_basis = price * shares
                portfolio["positions"][symbol] = {
                    "shares": shares,
                    "avg_price": price,
                    "cost_basis": cost_basis,
                    "first_entry": tx_date,
                    "core": False,
                }

            portfolio["cash"] -= price * shares
            portfolio["transactions"].append({
                "date": tx_date, "symbol": symbol, "action": "ADD",
                "shares": shares, "price": price,
            })

        elif act == "EXIT":
            shares = action.get("actual_shares", action.get("shares", 0))
            price = action.get("actual_price", action["current_price"])
            if shares <= 0:
                continue

            if symbol in portfolio["positions"]:
                pos = portfolio["positions"][symbol]
                if shares >= pos["shares"]:
                    del portfolio["positions"][symbol]
                else:
                    pos["shares"] -= shares
                    pos["cost_basis"] = pos["avg_price"] * pos["shares"]

            portfolio["cash"] += price * shares
            portfolio["transactions"].append({
                "date": tx_date, "symbol": symbol, "action": "EXIT",
                "shares": shares, "price": price,
            })


def load_watchlist(path=WATCHLIST_PATH):
    """讀取白名單"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"symbols": [], "updated": ""}


def save_watchlist(watchlist, path=WATCHLIST_PATH):
    """寫入白名單"""
    watchlist["updated"] = str(date.today())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, indent=2, ensure_ascii=False)


def add_to_watchlist(symbols, path=WATCHLIST_PATH):
    """新增標的到白名單"""
    wl = load_watchlist(path)
    for s in symbols:
        s = s.upper()
        if s not in wl["symbols"]:
            wl["symbols"].append(s)
    save_watchlist(wl, path)
    return wl
