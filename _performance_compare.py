"""
三場景績效比較：Buy&Hold vs 實際 vs 系統版（近似）

場景說明
--------
A. Buy & Hold (Jan 1)
   - 持有 2026/1/1 快照的 13 支股票，完全不動，放到今天。

B. 實際
   - portfolio.json 目前狀態。

C. 系統版（近似）
   - 假設三重警告期間（2/12–2/27）所有 ADD 全部暫緩，現金不動。
   - ROTATE sell 端仍執行（賣出弱勢），但 buy 端的新資金也保留為現金。
   - 2/28 之後的 ADD（DELL, HWM, CF, AKAM, EQT, EBAY）因為廣度仍低，
     一樣暫緩 → 保留為現金。
   - 技術上：Actual - 所有警告期 ADD 的損益，換回成本現金。

使用：
    conda run -n qt_env python _performance_compare.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

warnings.filterwarnings("ignore")

# ── 三重警告期間（依稽核結果） ───────────────────────────────────────────────
WARNING_START = "2026-02-12"
WARNING_END   = "2026-02-27"
# 3/3 以後：DELL/HWM 是警告後期，今天才買的 (3/17) 太早評估，不計入

# ── 讀取資料 ─────────────────────────────────────────────────────────────────

def load_data():
    with open("data/snapshot_2026.json") as f:
        snap = json.load(f)
    with open("data/portfolio.json") as f:
        port = json.load(f)
    return snap, port


def fetch_prices(symbols):
    """批次抓取目前收盤價"""
    if not symbols:
        return {}
    raw = yf.download(list(symbols), period="5d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw.rename(columns={raw.columns[0]: list(symbols)[0]}) if len(symbols) == 1 else raw
    close = close.ffill()
    result = {}
    for sym in symbols:
        if sym in close.columns:
            s = close[sym].dropna()
            if not s.empty:
                result[sym] = float(s.iloc[-1])
    return result


def fetch_prices_on_date(symbols, target_date):
    """抓取特定日期前後最近的收盤價（用於計算起始值）"""
    if not symbols:
        return {}
    raw = yf.download(list(symbols), start="2025-12-20", end="2026-01-15",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw
    close = close.ffill()
    td = pd.Timestamp(target_date)
    close_before = close[close.index <= td]
    if close_before.empty:
        return {}
    last = close_before.iloc[-1]
    return {sym: float(last[sym]) for sym in symbols if sym in last.index and pd.notna(last[sym])}


# ── 場景計算 ──────────────────────────────────────────────────────────────────

def scenario_bnh(snap, current_prices, jan1_prices):
    """B&H：用 1/1 持倉 × 目前價格"""
    positions = snap["positions"]
    cash      = snap.get("cash", 0)

    rows = []
    total = cash
    for sym, data in positions.items():
        shares = data["shares"]
        curr   = current_prices.get(sym)
        jan1   = jan1_prices.get(sym)
        if curr is None:
            print(f"  [B&H] {sym} 無現價，略過")
            continue
        curr_val = shares * curr
        total += curr_val
        pct = (curr / jan1 - 1) * 100 if jan1 else None
        rows.append({"symbol": sym, "shares": shares, "curr_price": curr,
                     "curr_val": curr_val, "ret_pct": pct})
    return total, cash, rows


def scenario_actual(port, current_prices):
    """Actual：目前 portfolio.json × 目前價格（重新計算確認）"""
    positions = port["positions"]
    cash      = port.get("cash", 0)
    total = cash
    rows = []
    for sym, data in positions.items():
        if data.get("core"):
            continue  # VOO 用 portfolio 的計算即可，這裡一起算
        pass
    for sym, data in positions.items():
        shares = data["shares"]
        curr   = current_prices.get(sym)
        if curr is None:
            continue
        curr_val = shares * curr
        avg_p    = data.get("avg_price", 0) or 0
        ret_pct  = (curr / avg_p - 1) * 100 if avg_p > 0 else None
        total += curr_val
        rows.append({"symbol": sym, "shares": shares, "avg_price": avg_p,
                     "curr_price": curr, "curr_val": curr_val, "ret_pct": ret_pct})
    return total, cash, rows


def scenario_system(port, current_prices):
    """
    系統版近似：
    - 找出三重警告期間 ADD 的每一筆
    - 計算各筆的損益（已出場 or 未出場）
    - 系統版 = Actual + sum(cost - current_value) for still-held warning ADDs
                      + sum(-net_pnl_for_exited_loss) for exited warning ADDs
    等價於：系統版 = Actual - 所有警告 ADD 的累計損益（若負，系統版更好）
    """
    txns = port["transactions"]

    # 找出警告期 ADD（只計主要警告期，不含今天）
    warning_adds = [
        t for t in txns
        if t["action"] == "ADD"
        and WARNING_START <= t["date"] <= WARNING_END
    ]

    # 對應的 EXIT（找同 symbol 在 ADD 之後的第一筆 EXIT）
    exits_by_sym = {}
    for t in txns:
        if t["action"] == "EXIT":
            exits_by_sym.setdefault(t["symbol"], []).append(t)
    for v in exits_by_sym.values():
        v.sort(key=lambda x: x["date"])

    used_exits = set()

    detail = []
    total_pnl = 0.0
    total_cost = 0.0

    for add in warning_adds:
        sym    = add["symbol"]
        shares = add["shares"]
        cost   = shares * add["price"]

        # 找配對的 EXIT
        exit_rec = None
        for i, ex in enumerate(exits_by_sym.get(sym, [])):
            key = (sym, ex["date"], i)
            if ex["date"] >= add["date"] and key not in used_exits:
                exit_rec = ex
                used_exits.add(key)
                break

        if exit_rec:
            # 已出場
            proceeds = exit_rec["shares"] * exit_rec["price"]
            # 若 shares 不同（部分出場）用比例計算
            matched_shares = min(shares, exit_rec["shares"])
            proceeds = matched_shares * exit_rec["price"]
            cost_matched = matched_shares * add["price"]
            pnl  = proceeds - cost_matched
            curr = exit_rec["price"]
            status = f"已出場 {exit_rec['date']}"
        else:
            # 仍持有
            curr = current_prices.get(sym)
            if curr is None:
                continue
            pnl = shares * curr - cost
            status = "持有中"

        total_pnl  += pnl
        total_cost += cost
        detail.append({
            "date":   add["date"],
            "symbol": sym,
            "shares": shares,
            "cost":   cost,
            "pnl":    pnl,
            "pnl_pct": pnl / cost * 100 if cost > 0 else 0,
            "status": status,
        })

    return total_pnl, total_cost, detail


# ── SPY 基準 ─────────────────────────────────────────────────────────────────

def spy_return():
    spy = yf.Ticker("SPY").history(start="2026-01-02", end=str(date.today()))
    if spy.empty:
        return None, None
    p0   = float(spy["Close"].iloc[0])
    p_now = float(spy["Close"].iloc[-1])
    return p0, p_now


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    snap, port = load_data()
    jan1_total = snap["total_value"]  # 126,970.93
    jan1_cash  = snap.get("cash", 0)

    # 收集所有需要現價的 symbol
    jan1_syms = set(snap["positions"].keys())
    curr_syms = set(port["positions"].keys())
    txn_syms  = {t["symbol"] for t in port["transactions"]}
    all_syms  = jan1_syms | curr_syms | txn_syms | {"SPY"}

    print("下載現價...")
    current_prices = fetch_prices(all_syms)
    print("下載 1/1 價格...")
    jan1_prices    = fetch_prices_on_date(jan1_syms, "2026-01-02")

    # ── 場景 A：B&H ───────────────────────────────────────────────────────
    bnh_total, bnh_cash, bnh_rows = scenario_bnh(snap, current_prices, jan1_prices)

    # ── 場景 B：Actual ────────────────────────────────────────────────────
    actual_total, actual_cash, actual_rows = scenario_actual(port, current_prices)

    # ── 場景 C：系統版 ────────────────────────────────────────────────────
    warning_pnl, warning_cost, warning_detail = scenario_system(port, current_prices)
    system_total = actual_total - warning_pnl   # 如果warning_pnl < 0，系統版更高

    # ── SPY 基準 ──────────────────────────────────────────────────────────
    spy_jan1, spy_now = spy_return()
    spy_ret = (spy_now / spy_jan1 - 1) * 100 if spy_jan1 and spy_now else None
    spy_bnh = jan1_total * (1 + spy_ret / 100) if spy_ret else None

    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 65)
    print("  三場景績效比較  2026/01/01 → 今日")
    print("=" * 65)
    print(f"  起始資產（2026/1/1）：${jan1_total:>12,.2f}")
    print()

    def show_line(label, val, cash=None):
        pct = (val / jan1_total - 1) * 100
        sign = "+" if pct >= 0 else ""
        bar_len = int(abs(pct) / 1)
        bar = ("█" * min(bar_len, 20)) if pct >= 0 else ("░" * min(bar_len, 20))
        cash_str = f"  （現金 ${cash:,.0f}）" if cash is not None else ""
        print(f"  {label:<26} ${val:>12,.2f}  {sign}{pct:+.1f}%  {bar}{cash_str}")

    show_line("A. Buy & Hold（不動）", bnh_total, bnh_cash)
    show_line("B. 實際（目前）",       actual_total, actual_cash)
    show_line("C. 系統版（近似）",     system_total)
    if spy_ret is not None:
        show_line(f"   SPY 基準（{spy_ret:+.1f}%）", spy_bnh if spy_bnh else 0)

    print()
    print("─" * 65)
    print(f"  B vs A（交易 vs 不動）：  {actual_total - bnh_total:>+10,.2f}")
    print(f"  C vs B（系統 vs 實際）：  {system_total - actual_total:>+10,.2f}")
    print(f"  C vs A（系統 vs 不動）：  {system_total - bnh_total:>+10,.2f}")

    # ─── 警告期 ADD 明細 ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  三重警告期 ADD 明細（{WARNING_START} ~ {WARNING_END}）")
    print(f"  共 {len(warning_detail)} 筆，總投入 ${warning_cost:,.0f}，累計損益 {warning_pnl:+,.0f}")
    print("=" * 65)
    print(f"  {'日期':<12} {'標的':<7} {'成本':>8}  {'損益':>8}  {'損益%':>7}  狀態")
    print("─" * 65)

    for d in sorted(warning_detail, key=lambda x: x["pnl"]):
        win = "✅" if d["pnl"] >= 0 else "❌"
        print(f"  {d['date']:<12} {d['symbol']:<7} ${d['cost']:>7,.0f}  {d['pnl']:>+8,.0f}  {d['pnl_pct']:>+6.1f}%  {win} {d['status']}")

    print("─" * 65)
    win_cnt  = sum(1 for d in warning_detail if d["pnl"] >= 0)
    loss_cnt = len(warning_detail) - win_cnt
    print(f"  合計：{win_cnt} 勝 / {loss_cnt} 敗   累計 {warning_pnl:+,.0f}  "
          f"（{'系統版可省下此損失' if warning_pnl < 0 else '實際較優'}）")

    # ─── B&H 明細（1/1 持倉現況） ───────────────────────────────────────
    print()
    print("=" * 65)
    print("  B&H 明細（1/1 持倉，若放到今天）")
    print("=" * 65)
    print(f"  {'標的':<7} {'股數':>6}  {'現價':>9}  {'現值':>10}  {'vs Jan1':>8}")
    print("─" * 65)
    for row in sorted(bnh_rows, key=lambda x: -(x["ret_pct"] or 0)):
        ret_str = f"{row['ret_pct']:+.1f}%" if row["ret_pct"] is not None else "N/A"
        win = "✅" if (row.get("ret_pct") or 0) >= 0 else "❌"
        print(f"  {row['symbol']:<7} {row['shares']:>6}  ${row['curr_price']:>8.2f}  "
              f"${row['curr_val']:>9,.0f}  {ret_str:>7}  {win}")
    print(f"  {'現金':>14}                    ${bnh_cash:>9,.0f}")
    print(f"  {'合計':>14}                    ${bnh_total:>9,.0f}")


if __name__ == "__main__":
    main()
