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


def _ensure_tranches(pos):
    """Lazy 遷移：確保 pos 有 tranches 欄位（舊有持倉自動補上 tranche 1）"""
    if "tranches" not in pos:
        pos["tranches"] = [{
            "n": 1,
            "shares": pos["shares"],
            "entry_price": pos["avg_price"],
            "entry_date": pos.get("first_entry", "unknown"),
            "high": pos.get("high_since_entry", pos["avg_price"]),
            "stop_type": "standard",
        }]
    return pos


def apply_confirmed_actions(portfolio, confirmed_actions):
    """套用 confirmed actions，更新 positions + 追加 transactions"""
    for action in confirmed_actions:
        if action.get("status") != "confirmed":
            continue

        act = action["action"]
        tx_date = action.get("confirm_date", str(date.today()))

        if act == "ADD":
            symbol = action["symbol"]
            shares = action.get("actual_shares", action.get("suggested_shares", 0))
            price = action.get("actual_price", action["current_price"])
            if shares <= 0:
                continue

            if symbol in portfolio["positions"]:
                pos = portfolio["positions"][symbol]
                # 先確保 tranches 存在（使用更新前的 avg_price 做 lazy migrate）
                _ensure_tranches(pos)

                # stop_type 依實際成交價 vs 前一批成本重算
                # （盤前推薦的 stop_type 是依當時市價，確認時用實際價格校正）
                prev_entry = pos["tranches"][-1]["entry_price"] if pos["tranches"] else 0
                n_next = max((t["n"] for t in pos["tranches"]), default=0) + 1
                if price > prev_entry:
                    # 實際買入比前批更貴 = 上行金字塔 = 收緊停損
                    stop_type = "tight_2" if n_next == 2 else "tight_3"
                else:
                    # 實際買入比前批便宜或相同 = 下行/持平 = 標準停損
                    stop_type = "standard"

                pos["avg_price"] = calc_avg_price(
                    pos["avg_price"], pos["shares"], price, shares
                )
                pos["shares"] += shares
                pos["cost_basis"] = pos["avg_price"] * pos["shares"]
                # 追加新批次
                pos["tranches"].append({
                    "n": n_next,
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": tx_date,
                    "high": price,
                    "stop_type": stop_type,
                })
            else:
                portfolio["positions"][symbol] = {
                    "shares": shares,
                    "avg_price": price,
                    "cost_basis": price * shares,
                    "first_entry": tx_date,
                    "high_since_entry": price,
                    "core": False,
                    "tranches": [{
                        "n": 1,
                        "shares": shares,
                        "entry_price": price,
                        "entry_date": tx_date,
                        "high": price,
                        "stop_type": stop_type,
                    }],
                }

            portfolio["cash"] -= price * shares
            portfolio["transactions"].append({
                "date": tx_date, "symbol": symbol, "action": "ADD",
                "shares": shares, "price": price,
            })

        elif act == "EXIT":
            symbol = action["symbol"]
            shares = action.get("actual_shares", action.get("shares", 0))
            price = action.get("actual_price", action["current_price"])
            tranche_n = action.get("tranche_n")
            if shares <= 0:
                continue

            if symbol in portfolio["positions"]:
                pos = portfolio["positions"][symbol]

                if tranche_n is not None and "tranches" in pos:
                    # 逐批出場：找到對應批次
                    tranche_idx = next(
                        (i for i, t in enumerate(pos["tranches"]) if t["n"] == tranche_n), None
                    )
                    if tranche_idx is not None:
                        t = pos["tranches"][tranche_idx]
                        actual_exit_shares = min(shares, t["shares"])
                        # 重算剩餘 avg_price（用批次進場成本移除，非市場成交價）
                        removed_cost = t["entry_price"] * actual_exit_shares
                        if actual_exit_shares >= t["shares"]:
                            pos["tranches"].pop(tranche_idx)
                        else:
                            t["shares"] -= actual_exit_shares
                        pos["shares"] = sum(tr["shares"] for tr in pos["tranches"])
                        if pos["shares"] <= 0 or not pos["tranches"]:
                            del portfolio["positions"][symbol]
                        else:
                            remaining_cost = pos["avg_price"] * (pos["shares"] + actual_exit_shares) - removed_cost
                            pos["avg_price"] = remaining_cost / pos["shares"] if pos["shares"] > 0 else 0
                            pos["cost_basis"] = pos["avg_price"] * pos["shares"]
                    else:
                        # tranche 找不到，退化為標準出場
                        if shares >= pos["shares"]:
                            del portfolio["positions"][symbol]
                        else:
                            pos["shares"] -= shares
                            pos["cost_basis"] = pos["avg_price"] * pos["shares"]
                else:
                    # 標準出場（全部或部分）
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

        elif act == "ROTATE":
            sell_sym = action["sell_symbol"]
            sell_shares = action.get("actual_sell_shares", action.get("sell_shares", 0))
            sell_price = action.get("actual_sell_price", action.get("sell_price", 0))
            buy_sym = action["buy_symbol"]
            buy_shares = action.get("actual_buy_shares", action.get("buy_shares", 0))
            buy_price = action.get("actual_buy_price", action.get("buy_price", 0))

            # 賣出部分
            if sell_shares > 0 and sell_sym in portfolio["positions"]:
                pos = portfolio["positions"][sell_sym]
                if sell_shares >= pos["shares"]:
                    del portfolio["positions"][sell_sym]
                else:
                    pos["shares"] -= sell_shares
                    pos["cost_basis"] = pos["avg_price"] * pos["shares"]
                portfolio["cash"] += sell_price * sell_shares
                portfolio["transactions"].append({
                    "date": tx_date, "symbol": sell_sym, "action": "EXIT",
                    "shares": sell_shares, "price": sell_price,
                })

            # 買入部分
            if buy_shares > 0:
                if buy_sym in portfolio["positions"]:
                    pos = portfolio["positions"][buy_sym]
                    pos["avg_price"] = calc_avg_price(
                        pos["avg_price"], pos["shares"], buy_price, buy_shares
                    )
                    pos["shares"] += buy_shares
                    pos["cost_basis"] = pos["avg_price"] * pos["shares"]
                else:
                    portfolio["positions"][buy_sym] = {
                        "shares": buy_shares,
                        "avg_price": buy_price,
                        "cost_basis": buy_price * buy_shares,
                        "first_entry": tx_date,
                        "high_since_entry": buy_price,
                        "core": False,
                    }
                portfolio["cash"] -= buy_price * buy_shares
                portfolio["transactions"].append({
                    "date": tx_date, "symbol": buy_sym, "action": "ADD",
                    "shares": buy_shares, "price": buy_price,
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


def update_high_prices(portfolio, current_prices):
    """更新每個持倉的最高價記錄

    Args:
        portfolio: 持倉 dict
        current_prices: {symbol: price}

    Returns:
        updated: bool，是否有更新
    """
    updated = False
    for symbol, pos in portfolio.get("positions", {}).items():
        price = current_prices.get(symbol)
        if price is None:
            continue

        current_high = pos.get("high_since_entry", pos.get("avg_price", 0))

        # 如果沒有 high_since_entry，用成本價初始化
        if "high_since_entry" not in pos:
            pos["high_since_entry"] = max(price, pos.get("avg_price", 0))
            updated = True
        elif price > current_high:
            pos["high_since_entry"] = price
            updated = True

        # 同步更新各批次高點
        if "tranches" in pos:
            pos_high = pos["high_since_entry"]
            for t in pos["tranches"]:
                if pos_high > t.get("high", 0):
                    t["high"] = pos_high
                    updated = True

    return updated


def initialize_high_prices(portfolio, current_prices):
    """初始化所有持倉的最高價（首次使用時）

    對於沒有 high_since_entry 的持倉，設為 max(current_price, avg_price)
    """
    for symbol, pos in portfolio.get("positions", {}).items():
        if "high_since_entry" not in pos:
            price = current_prices.get(symbol, pos.get("avg_price", 0))
            pos["high_since_entry"] = max(price, pos.get("avg_price", 0))
