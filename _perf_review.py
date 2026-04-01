"""
_perf_review.py

績效三路比較（2026 YTD）：
  A. Buy & Hold  ── 1/1 快照持倉完全不動
  B. Actual      ── 實際操作後的目前組合
  C. System      ── 盡量依系統信號（TSLA 延遲 exit 校正）
  D. SPY B&H     ── 等值 SPY 持有

使用：
    conda run -n qt_env python _perf_review.py
"""
import os, sys, json, glob, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf
from datetime import date

# ── 讀取快照 & 持倉 ──────────────────────────────────────────────────────────
with open("data/snapshot_2026.json") as f:
    snap = json.load(f)

with open("data/portfolio.json") as f:
    port = json.load(f)

SNAP_DATE     = snap["date"]          # "2026-01-02"
SNAP_VALUE    = snap["total_value"]   # 126970.93
SNAP_CASH     = snap["cash"]          # 155
SNAP_POSITIONS = snap["positions"]    # {sym: {shares, price, value}}

# ── 取得今日收盤價 ───────────────────────────────────────────────────────────
all_syms = list(set(list(SNAP_POSITIONS.keys()) + list(port["positions"].keys()) + ["SPY"]))

print(f"正在取得 {len(all_syms)} 檔最新報價...")
raw = yf.download(all_syms, period="5d", auto_adjust=True, progress=False)
close = raw["Close"].iloc[-1]  # 最後一天

def px(sym):
    v = close.get(sym, np.nan)
    return float(v) if (v is not None and not np.isnan(float(v))) else None

today_str = str(date.today())

# ── A. Buy & Hold（快照持倉） ────────────────────────────────────────────────
bnh_value = SNAP_CASH
bnh_rows  = []
for sym, info in SNAP_POSITIONS.items():
    shares   = info["shares"]
    cost_px  = info["price"]            # 快照時的市價
    cost_val = info["value"]            # 快照時的市值
    cur_px   = px(sym)
    if cur_px is None:
        cur_px = cost_px  # 無法取得就用快照價
        note = " (無報價)"
    else:
        note = ""
    cur_val  = shares * cur_px
    pnl_pct  = (cur_px - cost_px) / cost_px * 100
    bnh_value += cur_val
    bnh_rows.append({
        "sym": sym, "shares": shares,
        "cost_px": cost_px, "cur_px": cur_px,
        "cost_val": cost_val, "cur_val": cur_val,
        "pnl_pct": pnl_pct, "note": note,
    })
bnh_rows.sort(key=lambda r: r["cost_val"], reverse=True)

# ── B. Actual（目前組合市值） ────────────────────────────────────────────────
actual_cash  = port["cash"]
actual_value = actual_cash
actual_rows  = []
for sym, pos in port["positions"].items():
    shares   = pos["shares"]
    avg_px   = pos["avg_price"]
    cur_px   = px(sym)
    if cur_px is None:
        cur_px = avg_px
    cur_val  = shares * cur_px
    pnl_pct  = (cur_px - avg_px) / avg_px * 100
    actual_value += cur_val
    actual_rows.append({
        "sym": sym, "shares": shares, "avg_px": avg_px,
        "cur_px": cur_px, "cur_val": cur_val, "pnl_pct": pnl_pct,
    })
actual_rows.sort(key=lambda r: r["cur_val"], reverse=True)

# ── C. System-perfect（校正已知執行落差） ───────────────────────────────────
# 唯一追蹤到的 skipped EXIT：
#   TSLA 12股，系統建議 2026-03-16 @ $391.20，實際 2026-03-20 @ $371.91
#   差額：12 × (391.20 - 371.91) = $231.48（損失，系統應表現更好）
#
# 注意：無法追溯系統建議 ADD 中哪些被繞過，或 2026-02-12 前的手動操作。
# 因此「system-perfect」≈ actual + 已知 skipped exits 校正。
TSLA_DELAY_LOSS  = 12 * (391.20 - 371.91)   # $231.48
system_value     = actual_value + TSLA_DELAY_LOSS
system_note      = f"+ TSLA 延遲 exit 校正 +${TSLA_DELAY_LOSS:.0f}"

# ── D. SPY B&H ───────────────────────────────────────────────────────────────
SPY_SNAP_PX = 628.30   # 快照中 VOO 是 628.3，SPY 也幾乎相同（用 SPY 快照日收盤）
# 用 yfinance 取實際 2026-01-02 SPY 收盤
try:
    spy_hist = yf.download("SPY", start="2025-12-31", end="2026-01-07",
                           auto_adjust=True, progress=False)
    SPY_SNAP_PX = float(spy_hist["Close"].iloc[0])
except Exception:
    SPY_SNAP_PX = 628.30   # fallback

spy_cur_px    = px("SPY") or SPY_SNAP_PX
spy_shares    = SNAP_VALUE / SPY_SNAP_PX
spy_value     = spy_shares * spy_cur_px

# ── 摘要 ─────────────────────────────────────────────────────────────────────
def pct_chg(now, base):
    return (now - base) / base * 100

print()
print("=" * 65)
print(f"  績效三路比較  {SNAP_DATE} → {today_str}")
print(f"  起始資金: ${SNAP_VALUE:,.2f}")
print("=" * 65)
print(f"  {'策略':<22} {'目前市值':>12} {'損益':>10} {'vs 快照':>8} {'vs SPY':>8}")
print("-" * 65)

scenarios = [
    ("A. Buy & Hold（快照不動）", bnh_value),
    ("B. Actual（實際操作）",     actual_value),
    ("C. System（校正落差）",     system_value),
    ("D. SPY B&H",              spy_value),
]

for label, val in scenarios:
    gain      = val - SNAP_VALUE
    gain_pct  = pct_chg(val, SNAP_VALUE)
    spy_diff  = pct_chg(val, spy_value)
    gain_str  = f"${gain:+,.0f}"
    print(f"  {label:<22} ${val:>11,.2f}  {gain_str:>9}  {gain_pct:>+6.1f}%  {spy_diff:>+6.1f}%")

print()
spy_ytd = pct_chg(spy_value, SNAP_VALUE)
print(f"  SPY 同期 YTD: {spy_ytd:+.1f}%  (${SPY_SNAP_PX:.2f} → ${spy_cur_px:.2f})")
print(f"  System 備註: {system_note}")
print()

# ── 快照持倉拆解（B&H 各標的表現） ──────────────────────────────────────────
print("=" * 65)
print("  Buy & Hold 各標的表現（假設 1/1 持倉完全不動）")
print("-" * 65)
print(f"  {'標的':<7} {'股數':>5} {'1/1價':>8} {'今日價':>8} {'漲跌':>7} {'1/1市值':>10} {'今日市值':>10}")
print("-" * 65)
for r in bnh_rows:
    print(f"  {r['sym']:<7} {r['shares']:>5} "
          f"${r['cost_px']:>7.2f} ${r['cur_px']:>7.2f} "
          f"{r['pnl_pct']:>+6.1f}% "
          f"${r['cost_val']:>9,.0f} ${r['cur_val']:>9,.0f}{r['note']}")
print("-" * 65)
bnh_stock_cost = sum(r["cost_val"] for r in bnh_rows)
bnh_stock_now  = sum(r["cur_val"] for r in bnh_rows)
print(f"  {'合計':<7} {'':>5} {'':>8} {'':>8} "
      f"{pct_chg(bnh_stock_now, bnh_stock_cost):>+6.1f}% "
      f"${bnh_stock_cost:>9,.0f} ${bnh_stock_now:>9,.0f}")
print(f"  現金: ${SNAP_CASH:.0f}")
print()

# ── 目前組合拆解（Actual） ────────────────────────────────────────────────────
print("=" * 65)
print("  目前組合持倉（Actual）")
print("-" * 65)
print(f"  {'標的':<7} {'股數':>5} {'成本均價':>9} {'今日價':>8} {'損益%':>7} {'市值':>10}")
print("-" * 65)
for r in actual_rows:
    print(f"  {r['sym']:<7} {r['shares']:>5} "
          f"${r['avg_px']:>8.2f} ${r['cur_px']:>7.2f} "
          f"{r['pnl_pct']:>+6.1f}% ${r['cur_val']:>9,.0f}")
print("-" * 65)
total_stock = sum(r["cur_val"] for r in actual_rows)
print(f"  股票市值: ${total_stock:>10,.0f}  現金: ${actual_cash:>9,.0f}  合計: ${actual_value:>10,.0f}")
print()

# ── 交易紀錄分析 ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  交易紀錄分析（2026-02-12 起，系統啟用後）")
print("-" * 65)
txns = port["transactions"]
adds  = [t for t in txns if t["action"] == "ADD"]
exits = [t for t in txns if t["action"] == "EXIT"]
print(f"  ADD 交易 {len(adds)} 筆  EXIT 交易 {len(exits)} 筆")
print()

# 每個標的的 P&L（配對 ADD/EXIT）
sym_adds  = {}
sym_exits = {}
for t in adds:
    sym_adds.setdefault(t["symbol"], []).append(t)
for t in exits:
    sym_exits.setdefault(t["symbol"], []).append(t)

print(f"  {'標的':<7} {'買入成本':>10} {'賣出收回':>10} {'已實現損益':>10} {'狀態'}")
print("-" * 65)
realized_total = 0
for sym in sorted(set(list(sym_adds.keys()) + list(sym_exits.keys()))):
    cost   = sum(t["shares"] * t["price"] for t in sym_adds.get(sym, []))
    recv   = sum(t["shares"] * t["price"] for t in sym_exits.get(sym, []))
    still_held = sym in port["positions"]
    if still_held:
        # 未完全賣出：加上剩餘市值
        cur = px(sym) or port["positions"][sym]["avg_price"]
        rem_shares = port["positions"][sym]["shares"]
        unrealized = rem_shares * cur
        pnl    = recv + unrealized - cost
        pnl_str = f"${pnl:+,.0f}"
        status = f"持有中 {rem_shares}股 @${cur:.0f}"
    else:
        pnl    = recv - cost
        pnl_str = f"${pnl:+,.0f}"
        status = "已出場"
    if cost > 0 or recv > 0:
        realized_total += pnl if not still_held else recv - cost
        print(f"  {sym:<7} ${cost:>9,.0f} ${recv:>9,.0f} {pnl_str:>10}  {status}")

print("-" * 65)
print()
print(f"  已實現損益合計（不含持倉市值）：${realized_total:+,.0f}")
print()

# ── 結論 ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  結論")
print("-" * 65)
best_label, best_val = max(scenarios, key=lambda x: x[1])
print(f"  ✅ 最高績效：{best_label}  ${best_val:,.0f}")
print()
# A vs D（B&H vs SPY）
bnh_vs_spy = pct_chg(bnh_value, spy_value)
if bnh_vs_spy > 0:
    print(f"  📌 1/1 持倉不動 vs SPY：超越 {bnh_vs_spy:+.1f}%（主動持股勝指數）")
else:
    print(f"  📌 1/1 持倉不動 vs SPY：落後 {bnh_vs_spy:.1f}%（原始持股輸指數）")

# B vs A（主動管理 vs 不動）
active_alpha = pct_chg(actual_value, bnh_value)
if active_alpha > 0:
    print(f"  📌 主動管理 vs 不動：多賺 {active_alpha:+.1f}%（操作有加分）")
else:
    print(f"  📌 主動管理 vs 不動：少賺 {active_alpha:.1f}%（操作有減分）")

# B vs D（Actual vs SPY）
actual_vs_spy = pct_chg(actual_value, spy_value)
if actual_vs_spy > 0:
    print(f"  📌 實際組合 vs SPY：超越 {actual_vs_spy:+.1f}%")
else:
    print(f"  📌 實際組合 vs SPY：落後 {actual_vs_spy:.1f}%")

print()
print("  ⚠️  注意事項：")
print("  1. Survivorship bias：快照含 SHOP/UEC 等已停損標的，B&H 結果是反事實")
print("  2. System-perfect 只校正已知 skipped exit（TSLA 延遲 4 天）")
print("  3. 快照為 2026-01-02，2/12 前的交易未被系統記錄")
print("=" * 65)
