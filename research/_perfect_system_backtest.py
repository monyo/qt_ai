"""
_perfect_system_backtest.py

模擬「完美執行系統建議（無廣度限制）」的績效。

四條線比較：
  A. Buy & Hold（2026/1/1 持倉完全不動）
  B. 實際（portfolio.json 目前狀態）
  C. 系統版（原有，假設廣度警告期暫緩 ADD）
  D. 完美系統（所有 ADD/ROTATE/EXIT 建議照單全收，含 skipped）

執行：
    conda run -n qt_env python research/_perfect_system_backtest.py
"""
import os, sys, json, glob, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── 讀取初始快照 ──────────────────────────────────────────────────────────
with open("data/snapshot_2026.json") as f:
    snap = json.load(f)

with open("data/portfolio.json") as f:
    port = json.load(f)

# ── 讀取所有 actions 檔案（按日期排序）───────────────────────────────────
action_files = sorted(glob.glob("data/actions_2026*.json"))
print(f"讀取 {len(action_files)} 個 actions 檔案...")

all_actions = []  # [(date_str, actions_list)]
for fn in action_files:
    with open(fn) as f:
        a = json.load(f)
    all_actions.append((a["date"], a["actions"]))

# ── 下載所有需要的股票現價 ────────────────────────────────────────────────
# 找出所有出現過的標的
all_syms = set()
for _, actions in all_actions:
    for x in actions:
        if x.get("symbol"):
            all_syms.add(x["symbol"])
        if x.get("buy_symbol"):
            all_syms.add(x["buy_symbol"])
        if x.get("sell_symbol"):
            all_syms.add(x["sell_symbol"])
all_syms.update(snap["positions"].keys())
all_syms.update(port["positions"].keys())
all_syms.discard(None)

print(f"下載 {len(all_syms)} 支標的現價...")
raw = yf.download(list(all_syms), period="5d", auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    latest = raw["Close"].iloc[-1].to_dict()
else:
    sym = list(all_syms)[0]
    latest = {sym: float(raw["Close"].iloc[-1])}


def get_price(sym, fallback=None):
    p = latest.get(sym)
    if p and not np.isnan(p):
        return float(p)
    return fallback


# ── 場景 A：Buy & Hold ───────────────────────────────────────────────────
pos_a = {sym: info["shares"] for sym, info in snap["positions"].items()}
cash_a = snap["cash"]
val_a  = cash_a + sum(
    shares * (get_price(sym) or info["price"])
    for sym, (shares, info) in
    {s: (pos_a[s], snap["positions"][s]) for s in pos_a}.items()
)

# ── 場景 B：實際 ─────────────────────────────────────────────────────────
cash_b = port["cash"]
val_b  = cash_b + sum(
    pos["shares"] * (get_price(sym) or pos["avg_price"])
    for sym, pos in port["positions"].items()
)

# ── 場景 C：原系統版（廣度警告期暫緩）────────────────────────────────────
# 計算警告期 ADD 的損益（直接複用 _performance_compare.py 邏輯）
WARNING_ADDS = [
    # (symbol, shares, cost_per_share, confirmed_date_str)
    # 從 actions 檔撈出 2/12-2/27 已 confirmed 的 ADD
]
for date_str, actions in all_actions:
    if "2026-02-12" <= date_str <= "2026-02-27":
        for x in actions:
            if x["action"] == "ADD" and x.get("status") == "confirmed" and x.get("symbol"):
                WARNING_ADDS.append((
                    x["symbol"],
                    x.get("confirmed_shares") or x.get("suggested_shares", 0),
                    x.get("confirmed_price") or x.get("current_price", 0),
                ))

warning_pnl = 0
for sym, shares, cost in WARNING_ADDS:
    if shares > 0 and cost > 0:
        cur = get_price(sym, cost)
        # 如果已出場（不在現在持倉），需從 transactions 追蹤，
        # 這裡用簡化版：若仍持有用現價，否則假設已停損
        if sym in port["positions"]:
            warning_pnl += shares * (cur - cost)
        # else: 已出場，損益已實現在 B 中，C 要把它還原

# C 的近似：B - 警告期仍在持倉的 ADD 損益（移除那些倉位換回現金）
# 更精確的做法：用原 _performance_compare.py 的結果
# 這裡讀取快取值
val_c_approx = val_b  # 先設同，下面計算差值

# 計算 warning period 所有 confirmed ADD 的「持有損益」
warning_period_pnl = 0
warning_cost_total = 0
for sym, shares, cost in WARNING_ADDS:
    if shares > 0 and cost > 0:
        cur = get_price(sym, cost)
        warning_period_pnl += shares * (cur - cost)
        warning_cost_total += shares * cost

val_c = val_b - warning_period_pnl  # C ≈ 不買那些，留現金

# ── 場景 D：完美系統（含 skipped/pending，不限廣度）────────────────────────
print("\n模擬完美系統執行...")

pos_d  = {sym: {"shares": info["shares"], "avg_price": info["price"]}
          for sym, info in snap["positions"].items()}
cash_d = float(snap["cash"])

# 追蹤已執行的動作
executed_adds   = 0
executed_exits  = 0
executed_rotates = 0

for date_str, actions in all_actions:
    for x in actions:
        act = x.get("action")

        # ── EXIT ──────────────────────────────────────────────────────
        if act == "EXIT":
            sym    = x.get("symbol")
            shares = x.get("shares", 0)
            price  = x.get("actual_price") or x.get("current_price", 0)
            if sym and sym in pos_d and shares > 0 and price > 0:
                actual_shares = min(shares, pos_d[sym]["shares"])
                cash_d += actual_shares * price
                pos_d[sym]["shares"] -= actual_shares
                if pos_d[sym]["shares"] <= 0:
                    del pos_d[sym]
                executed_exits += 1

        # ── ADD（含 skipped/pending，只要 suggested_shares > 0）─────────
        elif act == "ADD":
            sym    = x.get("symbol")
            shares = x.get("suggested_shares", 0)
            price  = x.get("current_price", 0)
            if sym and shares > 0 and price > 0 and cash_d >= shares * price:
                cost = shares * price
                if sym in pos_d:
                    old_s = pos_d[sym]["shares"]
                    old_p = pos_d[sym]["avg_price"]
                    pos_d[sym]["avg_price"] = (old_s * old_p + cost) / (old_s + shares)
                    pos_d[sym]["shares"]    = old_s + shares
                else:
                    pos_d[sym] = {"shares": shares, "avg_price": price}
                cash_d -= cost
                executed_adds += 1

        # ── ROTATE ────────────────────────────────────────────────────
        elif act == "ROTATE":
            s_sym    = x.get("sell_symbol")
            s_shares = x.get("sell_shares", 0)
            s_price  = x.get("sell_price", 0)
            b_sym    = x.get("buy_symbol")
            b_shares = x.get("buy_shares", 0)
            b_price  = x.get("buy_price", 0)

            # 先賣
            if s_sym and s_sym in pos_d and s_shares > 0 and s_price > 0:
                actual_sell = min(s_shares, pos_d[s_sym]["shares"])
                cash_d += actual_sell * s_price
                pos_d[s_sym]["shares"] -= actual_sell
                if pos_d[s_sym]["shares"] <= 0:
                    del pos_d[s_sym]

                # 再買
                if b_sym and b_shares > 0 and b_price > 0 and cash_d >= b_shares * b_price:
                    cost = b_shares * b_price
                    if b_sym in pos_d:
                        old_s = pos_d[b_sym]["shares"]
                        old_p = pos_d[b_sym]["avg_price"]
                        pos_d[b_sym]["avg_price"] = (old_s * old_p + cost) / (old_s + b_shares)
                        pos_d[b_sym]["shares"]    = old_s + b_shares
                    else:
                        pos_d[b_sym] = {"shares": b_shares, "avg_price": b_price}
                    cash_d -= cost
                    executed_rotates += 1

# ── 計算 D 最終價值 ──────────────────────────────────────────────────────
val_d = cash_d + sum(
    pos["shares"] * (get_price(sym) or pos["avg_price"])
    for sym, pos in pos_d.items()
)

START = snap["total_value"]

def bar(pct, width=10):
    n = max(0, int(pct / 3))
    return "█" * n

# ── 場景 D2：完美系統 + alpha_1y > 0 過濾 ────────────────────────────────
print("模擬 D2（加 alpha_1y > 0 過濾）...")

pos_d2  = {sym: {"shares": info["shares"], "avg_price": info["price"]}
           for sym, info in snap["positions"].items()}
cash_d2 = float(snap["cash"])
exec_d2_adds = 0

for date_str, actions in all_actions:
    for x in actions:
        act = x.get("action")

        if act == "EXIT":
            sym    = x.get("symbol")
            shares = x.get("shares", 0)
            price  = x.get("actual_price") or x.get("current_price", 0)
            if sym and sym in pos_d2 and shares > 0 and price > 0:
                actual_shares = min(shares, pos_d2[sym]["shares"])
                cash_d2 += actual_shares * price
                pos_d2[sym]["shares"] -= actual_shares
                if pos_d2[sym]["shares"] <= 0:
                    del pos_d2[sym]

        elif act == "ADD":
            sym    = x.get("symbol")
            shares = x.get("suggested_shares", 0)
            price  = x.get("current_price", 0)
            a1y    = x.get("alpha_1y")
            # alpha_1y 過濾：若有資料且 <= 0 則跳過
            if a1y is not None and a1y <= 0:
                continue
            if sym and shares > 0 and price > 0 and cash_d2 >= shares * price:
                cost = shares * price
                if sym in pos_d2:
                    old_s = pos_d2[sym]["shares"]
                    old_p = pos_d2[sym]["avg_price"]
                    pos_d2[sym]["avg_price"] = (old_s * old_p + cost) / (old_s + shares)
                    pos_d2[sym]["shares"]    = old_s + shares
                else:
                    pos_d2[sym] = {"shares": shares, "avg_price": price}
                cash_d2 -= cost
                exec_d2_adds += 1

        elif act == "ROTATE":
            s_sym    = x.get("sell_symbol")
            s_shares = x.get("sell_shares", 0)
            s_price  = x.get("sell_price", 0)
            b_sym    = x.get("buy_symbol")
            b_shares = x.get("buy_shares", 0)
            b_price  = x.get("buy_price", 0)
            b_a1y    = x.get("buy_alpha_1y")
            # ROTATE 買端也過濾
            if b_a1y is not None and b_a1y <= 0:
                continue
            if s_sym and s_sym in pos_d2 and s_shares > 0 and s_price > 0:
                actual_sell = min(s_shares, pos_d2[s_sym]["shares"])
                cash_d2 += actual_sell * s_price
                pos_d2[s_sym]["shares"] -= actual_sell
                if pos_d2[s_sym]["shares"] <= 0:
                    del pos_d2[s_sym]
                if b_sym and b_shares > 0 and b_price > 0 and cash_d2 >= b_shares * b_price:
                    cost = b_shares * b_price
                    if b_sym in pos_d2:
                        old_s = pos_d2[b_sym]["shares"]
                        old_p = pos_d2[b_sym]["avg_price"]
                        pos_d2[b_sym]["avg_price"] = (old_s * old_p + cost) / (old_s + b_shares)
                        pos_d2[b_sym]["shares"]    = old_s + b_shares
                    else:
                        pos_d2[b_sym] = {"shares": b_shares, "avg_price": b_price}
                    cash_d2 -= cost

val_d2 = cash_d2 + sum(
    pos["shares"] * (get_price(sym) or pos["avg_price"])
    for sym, pos in pos_d2.items()
)

print(f"""
=================================================================
  五場景績效比較  2026/01/01 → 今日
=================================================================
  起始資產（2026/1/1）：${START:>12,.2f}

  A. Buy & Hold（不動）             ${val_a:>12,.2f}  {(val_a/START-1)*100:>+5.1f}%  {bar((val_a/START-1)*100)}
  D. 完美系統（無廣度限制）              ${val_d:>12,.2f}  {(val_d/START-1)*100:>+5.1f}%  {bar((val_d/START-1)*100)}
     SPY 基準（+5.8%）              ${START*1.058:>12,.2f}  {5.8:>+5.1f}%  {bar(5.8)}
  D2. 完美系統（alpha_1y>0 過濾）       ${val_d2:>12,.2f}  {(val_d2/START-1)*100:>+5.1f}%  {bar((val_d2/START-1)*100)}
  C. 系統版（廣度警告限制）             ${val_c:>12,.2f}  {(val_c/START-1)*100:>+5.1f}%  {bar((val_c/START-1)*100)}
  B. 實際（目前）                    ${val_b:>12,.2f}  {(val_b/START-1)*100:>+5.1f}%  {bar((val_b/START-1)*100)}

─────────────────────────────────────────────────────────────────
  D2 vs D（alpha 過濾的效果）：  ${val_d2 - val_d:>+10,.2f}
  D2 vs B（與實際的差距）：      ${val_d2 - val_b:>+10,.2f}
  D2 vs C（vs 廣度限制版）：     ${val_d2 - val_c:>+10,.2f}

=================================================================
  執行統計：
    D  ADD：{executed_adds} 筆（含所有 skipped/pending）
    D2 ADD：{exec_d2_adds} 筆（alpha_1y > 0 過濾後）
    D2 最終持倉：{len(pos_d2)} 支，現金 ${cash_d2:,.2f}
=================================================================
""")

# ── D2 持倉明細（被 alpha 過濾擋掉的輸家）───────────────────────────────
d_syms  = set(pos_d.keys())
d2_syms = set(pos_d2.keys())
filtered_out = d_syms - d2_syms
if filtered_out:
    print(f"  alpha_1y 過濾擋掉的標的（共 {len(filtered_out)} 支）：")
    for sym in sorted(filtered_out):
        cur = get_price(sym)
        avg = pos_d.get(sym, {}).get("avg_price", 0)
        pnl = (cur / avg - 1) * 100 if avg and cur else 0
        print(f"    {sym:8}  損益 {pnl:>+6.1f}%  {'← 好' if pnl < 0 else '← 錯誤過濾'}")
