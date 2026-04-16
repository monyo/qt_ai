"""
_system_perf_sim.py

系統自動執行績效模擬（System-Perfect Simulation）

問題：若 2026-02-11 系統啟用後，所有建議都自動執行（無人介入），績效會是？

方法：
  - 起點：2026-02-11 實際持倉（從 actions_20260211 的 HOLD + 待賣倉位還原）
  - 模擬：依序處理每個 actions_*.json，無論 status 為何，
          EXIT 全賣（系統建議價）、ADD 全買（建議股數 × 建議價）
  - 終點：今日收盤價計算所有剩餘持倉 + 現金
  - 比較：Actual（真實組合）vs System-Perfect vs SPY

使用：
    conda run -n qt_env python _system_perf_sim.py
"""
import os, sys, json, glob, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf
from datetime import date, datetime

# ── 載入資料 ──────────────────────────────────────────────────────────────────
with open("data/portfolio.json") as f:
    port = json.load(f)

actions_files = sorted(glob.glob("data/actions_2026*.json"))
START_DATE = "20260211"   # 系統正式啟用日

# ── 取得今日收盤價 ─────────────────────────────────────────────────────────────
# 蒐集所有曾出現的 ticker
all_syms = set(["SPY"])
for fname in actions_files:
    with open(fname) as f:
        acts = json.load(f)
    if isinstance(acts, dict):
        acts = acts.get("actions", [])
    for a in acts:
        for key in ("symbol", "buy_symbol", "sell_symbol"):
            if s := a.get(key):
                all_syms.add(s)
for pos in port["positions"]:
    all_syms.add(pos)
for t in port["transactions"]:
    all_syms.add(t["symbol"])

print(f"取得 {len(all_syms)} 檔最新報價...")
raw = yf.download(list(all_syms), period="5d", auto_adjust=True, progress=False)
prices_now = raw["Close"].iloc[-1]

def px_now(sym):
    v = prices_now.get(sym, np.nan)
    return float(v) if v is not None and not np.isnan(float(v)) else None

# 取得 SPY 在 2026-02-11 的收盤價（用作 SPY 起始點）
spy_hist = yf.download("SPY", start="2026-02-10", end="2026-02-15",
                        auto_adjust=True, progress=False)
SPY_START_PX = float(spy_hist["Close"].iloc[0])
SPY_NOW_PX   = px_now("SPY") or SPY_START_PX

# ── 還原系統啟用日（2026-02-11）的起始投組 ────────────────────────────────────
# 從 actions_20260211 的 HOLD 欄位 + 當日 EXIT 待賣倉位，還原完整起始持倉
with open("data/actions_20260211.json") as f:
    day0_acts = json.load(f)
if isinstance(day0_acts, dict):
    day0_acts = day0_acts.get("actions", [])

# 起始持倉（包含 HOLD 和即將 EXIT 的倉位）
start_positions = {}  # {sym: shares}
for a in day0_acts:
    sym = a.get("symbol")
    if not sym:
        continue
    if a["action"] in ("HOLD", "EXIT"):
        sh = a.get("shares") or a.get("tranche_shares", 0)
        if sh and sh > 0:
            start_positions[sym] = start_positions.get(sym, 0) + sh

# 起始現金（從 snapshot 取，代入 Feb 11 當日報價重算總值）
START_CASH = 155.0   # snapshot 快照現金

# 計算 Feb 11 起始總值（用 actions 裡的 current_price 作為 Feb 11 報價）
day0_prices = {}
for a in day0_acts:
    sym = a.get("symbol")
    if sym and a.get("current_price"):
        day0_prices[sym] = a["current_price"]

start_value = START_CASH
for sym, sh in start_positions.items():
    p = day0_prices.get(sym, 0)
    start_value += sh * p

print(f"\n系統啟用日（2026-02-11）起始投組:")
for sym, sh in sorted(start_positions.items()):
    p = day0_prices.get(sym, 0)
    print(f"  {sym:<6} {sh:>4}股 @ ${p:.2f} = ${sh*p:,.0f}")
print(f"  現金: ${START_CASH:.0f}")
print(f"  合計: ${start_value:,.2f}")
SPY_SHARES_EQUIV = start_value / SPY_START_PX
print(f"  若全買SPY: {SPY_SHARES_EQUIV:.1f}股 @ ${SPY_START_PX:.2f}\n")

# ── System-Perfect 模擬 ────────────────────────────────────────────────────────
sim_positions = dict(start_positions)   # {sym: shares}
sim_cash      = START_CASH
sim_log       = []    # 記錄每筆交易

for fname in actions_files:
    date_str = os.path.basename(fname).replace("actions_", "").replace(".json", "")
    if date_str < START_DATE:
        continue

    with open(fname) as f:
        acts = json.load(f)
    if isinstance(acts, dict):
        acts = acts.get("actions", [])

    day_exits = []
    day_adds  = []
    day_rotates = []

    for a in acts:
        action = a.get("action")
        if action == "EXIT":
            sym    = a.get("symbol")
            px     = a.get("current_price", 0)
            sh     = a.get("shares") or a.get("tranche_shares") or sim_positions.get(sym, 0)
            if sym and px > 0 and sh > 0:
                day_exits.append((sym, sh, px))
        elif action == "ADD":
            sym        = a.get("symbol")
            px         = a.get("current_price", 0)
            sh         = a.get("suggested_shares", 0) or 0
            is_pyramid = a.get("is_pyramid", False)
            if sym and px > 0 and sh > 0:
                day_adds.append((sym, sh, px, is_pyramid))
        elif action == "ROTATE":
            sell_sym = a.get("sell_symbol")
            sell_px  = a.get("sell_price", a.get("current_price", 0))
            sell_sh  = a.get("sell_shares", 0)
            buy_sym  = a.get("buy_symbol")
            buy_px   = a.get("buy_price", a.get("current_price", 0))
            buy_sh   = a.get("buy_shares", 0)
            if sell_sym and buy_sym:
                day_rotates.append((sell_sym, sell_sh, sell_px, buy_sym, buy_sh, buy_px))

    # 先執行 EXIT
    for sym, sh, px in day_exits:
        held = sim_positions.get(sym, 0)
        if held <= 0:
            continue
        actual_sell = min(sh, held)
        sim_cash += actual_sell * px
        sim_positions[sym] = held - actual_sell
        if sim_positions[sym] <= 0:
            del sim_positions[sym]
        sim_log.append(f"{date_str} EXIT {sym} {actual_sell}股 @${px:.2f} → 現金+${actual_sell*px:,.0f}")

    # 執行 ROTATE
    for sell_sym, sell_sh, sell_px, buy_sym, buy_sh, buy_px in day_rotates:
        held = sim_positions.get(sell_sym, 0)
        if held > 0 and sell_px > 0:
            actual_sell = min(sell_sh or held, held)
            sim_cash += actual_sell * sell_px
            sim_positions[sell_sym] = held - actual_sell
            if sim_positions[sell_sym] <= 0:
                del sim_positions[sell_sym]
            sim_log.append(f"{date_str} ROTATE-SELL {sell_sym} {actual_sell}股 @${sell_px:.2f}")
        if buy_sh > 0 and buy_px > 0:
            cost = buy_sh * buy_px
            if sim_cash >= cost:
                sim_cash -= cost
                sim_positions[buy_sym] = sim_positions.get(buy_sym, 0) + buy_sh
                sim_log.append(f"{date_str} ROTATE-BUY  {buy_sym} {buy_sh}股 @${buy_px:.2f} → 現金-${cost:,.0f}")

    # 執行 ADD（依建議順序，現金不足就跳過）
    # 規則：同標的已持有就不重複 ADD（真實系統買後變 HOLD，不再推 ADD）
    # 例外：is_pyramid=True 的金字塔加碼允許疊加
    for sym, sh, px, is_pyramid in day_adds:
        already_held = sym in sim_positions
        if already_held and not is_pyramid:
            continue   # 已持有非金字塔 → 系統會顯示 HOLD，不重複買
        cost = sh * px
        if sim_cash >= cost:
            sim_cash -= cost
            sim_positions[sym] = sim_positions.get(sym, 0) + sh
            tag = "(金字塔)" if is_pyramid else ""
            sim_log.append(f"{date_str} ADD  {sym} {sh}股 @${px:.2f} → 現金-${cost:,.0f} {tag}")
        else:
            sim_log.append(f"{date_str} ADD  {sym} {sh}股 @${px:.2f} [現金不足 ${sim_cash:,.0f}，跳過]")

# ── System-Perfect 終值 ─────────────────────────────────────────────────────────
sim_stock_value = 0
sim_rows = []
for sym, sh in sorted(sim_positions.items(), key=lambda x: -x[1]):
    p = px_now(sym)
    if p is None:
        p = day0_prices.get(sym, 0)
    val = sh * p
    sim_stock_value += val
    sim_rows.append((sym, sh, p, val))

sim_total = sim_stock_value + sim_cash

# ── Actual 終值 ───────────────────────────────────────────────────────────────
actual_stock = 0
actual_rows  = []
for sym, pos in sorted(port["positions"].items(), key=lambda x: -x[1].get("shares",0)*px_now(x[0]) if px_now(x[0]) else 0):
    p   = px_now(sym) or pos["avg_price"]
    val = pos["shares"] * p
    actual_stock += val
    actual_rows.append((sym, pos["shares"], p, val))
actual_total = actual_stock + port["cash"]

# ── SPY 終值 ──────────────────────────────────────────────────────────────────
spy_total = SPY_SHARES_EQUIV * SPY_NOW_PX

# ── 輸出結果 ──────────────────────────────────────────────────────────────────
print("=" * 68)
print(f"  System-Perfect 模擬  2026-02-11 → {date.today()}")
print(f"  起始值: ${start_value:,.2f}  (SPY 起點: ${SPY_START_PX:.2f})")
print("=" * 68)

def pct(now, base): return (now - base) / base * 100

print(f"\n  {'策略':<26} {'終值':>12} {'損益':>10} {'報酬率':>8} {'vs SPY':>8}")
print(f"  {'-'*64}")
rows = [
    ("A. Actual（實際操作）",      actual_total),
    ("B. System-Perfect（自動）",   sim_total),
    ("C. SPY B&H",                  spy_total),
]
for label, val in rows:
    gain     = val - start_value
    gain_pct = pct(val, start_value)
    vs_spy   = pct(val, spy_total)
    print(f"  {label:<26} ${val:>11,.2f}  ${gain:>+9,.0f}  {gain_pct:>+7.1f}%  {vs_spy:>+7.1f}%")

print(f"\n  SPY: ${SPY_START_PX:.2f} → ${SPY_NOW_PX:.2f}  ({pct(SPY_NOW_PX, SPY_START_PX):+.1f}%)")

# ── System-Perfect 持倉明細 ────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  System-Perfect 最終持倉")
print(f"  {'標的':<7} {'股數':>5} {'今日價':>9} {'市值':>10}")
print(f"  {'-'*35}")
for sym, sh, p, val in sorted(sim_rows, key=lambda x: -x[3]):
    print(f"  {sym:<7} {sh:>5} ${p:>8.2f} ${val:>9,.0f}")
print(f"  {'-'*35}")
print(f"  股票市值: ${sim_stock_value:>9,.0f}  現金: ${sim_cash:>9,.0f}  合計: ${sim_total:>10,.0f}")

# ── Actual 持倉明細 ───────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  Actual 最終持倉")
print(f"  {'標的':<7} {'股數':>5} {'今日價':>9} {'市值':>10}")
print(f"  {'-'*35}")
for sym, sh, p, val in sorted(actual_rows, key=lambda x: -x[3]):
    print(f"  {sym:<7} {sh:>5} ${p:>8.2f} ${val:>9,.0f}")
print(f"  {'-'*35}")
print(f"  股票市值: ${actual_stock:>9,.0f}  現金: ${port['cash']:>9,.0f}  合計: ${actual_total:>10,.0f}")

# ── 交易差異分析 ──────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  System-Perfect 執行紀錄（摘要）")
print(f"  {'-'*64}")
exits_log = [l for l in sim_log if "EXIT" in l or "ROTATE-SELL" in l]
adds_log  = [l for l in sim_log if "ADD" in l]
skips_log = [l for l in sim_log if "跳過" in l]
print(f"  EXIT 執行: {len(exits_log)} 筆  ADD 執行: {len([l for l in adds_log if '跳過' not in l])} 筆  現金不足跳過: {len(skips_log)} 筆")

if skips_log:
    print(f"\n  ⚠️  現金不足跳過的 ADD：")
    for l in skips_log:
        print(f"    {l}")

# ── 持倉差異 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  持倉差異（System vs Actual）")
print(f"  {'-'*64}")
sim_syms    = set(sim_positions.keys())
actual_syms = set(port["positions"].keys())

only_in_sys = sim_syms - actual_syms
only_in_act = actual_syms - sim_syms
common      = sim_syms & actual_syms

if only_in_sys:
    print(f"  📌 System 持有但 Actual 沒有（系統買了，你沒跟）:")
    for sym in sorted(only_in_sys):
        sh  = sim_positions[sym]
        p   = px_now(sym) or 0
        print(f"     {sym:<6} {sh}股 現值 ${sh*p:,.0f}")

if only_in_act:
    print(f"  📌 Actual 持有但 System 沒有（你自己買，系統沒推）:")
    for sym in sorted(only_in_act):
        sh  = port["positions"][sym]["shares"]
        p   = px_now(sym) or 0
        avg = port["positions"][sym]["avg_price"]
        pnl = (p - avg) / avg * 100 if avg > 0 else 0
        print(f"     {sym:<6} {sh}股 成本${avg:.0f} 現值${p:.0f} P&L{pnl:+.1f}%")

if common:
    print(f"  📌 兩邊都有但股數不同:")
    diffs = [(sym, sim_positions[sym], port["positions"][sym]["shares"])
             for sym in sorted(common)
             if abs(sim_positions[sym] - port["positions"][sym]["shares"]) > 0.5]
    if diffs:
        for sym, s_sh, a_sh in diffs:
            p = px_now(sym) or 0
            diff_val = (s_sh - a_sh) * p
            print(f"     {sym:<6} System:{s_sh:>4}股  Actual:{a_sh:>4}股  差值${diff_val:+,.0f}")
    else:
        print(f"     （無差異）")

print(f"\n{'='*68}")
delta = sim_total - actual_total
print(f"  結論：System-Perfect {'優於' if delta > 0 else '劣於'} Actual  差距 ${delta:+,.0f}  ({pct(sim_total, actual_total):+.1f}%)")
print(f"{'='*68}")
