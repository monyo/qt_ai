"""
_deposit_sim_backtest.py

假設：
  - 依照實際入金時間與金額，但完全不買 VOO
  - 使用現行動能系統（TOP5，21天再平衡，固定-15%/追蹤-25%停損）
  - 每個再平衡點全倉切換至新 TOP5（等權重）
  - 入金直接加入現金池，下次再平衡時投入

比較節點：
  - 2021/06/07 首次入金
  - 2026/02/11 系統打造初期
  - 2026/02/27 歷史資料終點
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
from datetime import date

OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
symbols  = list(df_close.columns)
dates    = list(df_close.index)
print(f"資料：{len(symbols)} 支  {dates[0].date()} ~ {dates[-1].date()}")

# ── 入金排程 ─────────────────────────────────────────────────────────────────
DEPOSITS = [
    (pd.Timestamp('2021-06-07'), 3600),
    (pd.Timestamp('2021-07-01'), 10500),
    (pd.Timestamp('2021-08-04'), 11000),
    (pd.Timestamp('2022-01-28'), 7200),
    (pd.Timestamp('2022-09-01'), 10000),
    (pd.Timestamp('2023-05-09'), 4070),
    (pd.Timestamp('2025-08-08'), 29400),
]
TOTAL_DEPOSITS = sum(v for _, v in DEPOSITS)

# ── 參數 ─────────────────────────────────────────────────────────────────────
MOM_SHORT  = 21
MOM_LONG   = 252
REBAL_FREQ = 21
TOP_N      = 5
FIXED_STOP = 0.15
TRAIL_STOP = 0.25

# ── 日期索引 ─────────────────────────────────────────────────────────────────
date_to_idx = {d: i for i, d in enumerate(dates)}
close_arr   = {s: df_close[s].values.astype(float) for s in symbols}

START_IDX = MOM_LONG + 10

# ── 找「第一個 >= target 的交易日索引」─────────────────────────────────────
def find_next_idx(target_ts):
    for i, d in enumerate(dates):
        if d >= target_ts:
            return i
    return len(dates) - 1

# ── 計算一個時間點的 TOP_N 動能股 ─────────────────────────────────────────
def get_top_n(t):
    scores = {}
    for s in symbols:
        c = close_arr[s]
        if t < MOM_LONG or t >= len(c): continue
        if np.isnan(c[t]) or np.isnan(c[t-MOM_SHORT]) or np.isnan(c[t-MOM_LONG]): continue
        m_s = c[t] / c[t-MOM_SHORT] - 1
        m_l = c[t] / c[t-MOM_LONG]  - 1
        scores[s] = 0.5*m_s + 0.5*m_l
    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    return ranked[:TOP_N]

# ── 模擬 ─────────────────────────────────────────────────────────────────────
cash      = 0.0
positions = {}   # sym -> {shares, entry_price, high_price}
deposit_q = list(DEPOSITS)

# 入金時間點 → 索引
deposit_schedule = {find_next_idx(ts): amt for ts, amt in DEPOSITS}

# 再平衡時間點
rebal_times = []
t = START_IDX
while t < len(dates):
    rebal_times.append(t)
    t += REBAL_FREQ

# 紀錄快照
snapshots = {}   # date -> total_value

def portfolio_value(t):
    v = cash
    for s, pos in positions.items():
        c = close_arr[s]
        if t < len(c) and not np.isnan(c[t]):
            v += pos['shares'] * c[t]
    return v

# 找首次有現金的再平衡點
first_deposit_idx = find_next_idx(DEPOSITS[0][0])

rebal_set = set(rebal_times)

for ti, d in enumerate(dates):
    if ti < START_IDX:
        continue

    # ── 入金 ─────────────────────────────────────────────
    for dep_t in sorted(list(deposit_schedule.keys())):
        if dep_t <= ti:
            amt = deposit_schedule.pop(dep_t)
            cash += amt

    # ── 每日停損檢查 ──────────────────────────────────────
    to_exit = []
    for s, pos in positions.items():
        c = close_arr[s]
        if ti >= len(c) or np.isnan(c[ti]):
            to_exit.append(s)
            continue
        px = c[ti]
        pos['high_price'] = max(pos['high_price'], px)
        trail_s = pos['high_price'] * (1 - TRAIL_STOP)
        fixed_s = pos['entry_price'] * (1 - FIXED_STOP)
        if px < max(trail_s, fixed_s):
            cash += pos['shares'] * px
            to_exit.append(s)
    for s in to_exit:
        del positions[s]

    # ── 再平衡點：切換至新 TOP_N ──────────────────────────
    if ti in rebal_set and (cash > 100 or positions):
        top = get_top_n(ti)
        if not top:
            snapshots[d] = portfolio_value(ti)
            continue

        # 賣出不在 TOP_N 的持倉
        to_sell = [s for s in list(positions.keys()) if s not in top]
        for s in to_sell:
            c = close_arr[s]
            if ti < len(c) and not np.isnan(c[ti]):
                cash += positions[s]['shares'] * c[ti]
            del positions[s]

        # 等權重分配（只用可用現金補倉，不强制賣出現有持倉）
        total_val = portfolio_value(ti)
        per_slot  = total_val / TOP_N

        for s in top:
            c = close_arr[s]
            if ti >= len(c) or np.isnan(c[ti]) or c[ti] <= 0:
                continue
            px = c[ti]
            current_val = positions[s]['shares'] * px if s in positions else 0
            diff = per_slot - current_val
            if diff > px * 0.5 and cash >= diff * 0.99:
                add_shares = diff / px
                cash -= add_shares * px
                if s in positions:
                    old = positions[s]
                    ns = old['shares'] + add_shares
                    positions[s] = {
                        'shares': ns,
                        'entry_price': (old['entry_price']*old['shares'] + px*add_shares) / ns,
                        'high_price': max(old['high_price'], px),
                    }
                else:
                    positions[s] = {'shares': add_shares, 'entry_price': px, 'high_price': px}
            elif diff < -px * 0.5:
                sell_shares = min(-diff / px, positions[s]['shares'])
                cash += sell_shares * px
                positions[s]['shares'] -= sell_shares
                if positions[s]['shares'] < 0.01:
                    del positions[s]

    # 紀錄快照
    snapshots[d] = portfolio_value(ti)

# ── 顯示關鍵節點 ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  假設：依實際入金 × 現行動能系統 × 完全不買 VOO")
print(f"  總投入：${TOTAL_DEPOSITS:,}")
print("=" * 65)

milestones = [
    (pd.Timestamp('2021-06-07'), '首次入金'),
    (pd.Timestamp('2022-01-01'), '2022年初'),
    (pd.Timestamp('2023-01-01'), '2023年初'),
    (pd.Timestamp('2024-01-01'), '2024年初'),
    (pd.Timestamp('2025-01-01'), '2025年初'),
    (pd.Timestamp('2025-08-08'), '第7筆入金'),
    (pd.Timestamp('2026-01-02'), '2026年初'),
    (pd.Timestamp('2026-02-11'), '系統打造初'),
    (pd.Timestamp('2026-02-27'), '資料終點'),
]

for ts, label in milestones:
    # 找最近的快照
    closest = min(snapshots.keys(), key=lambda d: abs((d - ts).days))
    val = snapshots[closest]
    invested_by = sum(amt for dep_ts, amt in DEPOSITS if dep_ts <= ts)
    ret = (val / invested_by - 1) * 100 if invested_by > 0 else 0
    print(f"  {label:12}  {closest.date()}  ${val:>12,.0f}  "
          f"(投入${invested_by:,}  {ret:+.1f}%)")

print("=" * 65)
print()

# 2026/02/11 vs 實際
sim_0211 = min(snapshots.keys(), key=lambda d: abs((d - pd.Timestamp('2026-02-11')).days))
sim_end  = min(snapshots.keys(), key=lambda d: abs((d - pd.Timestamp('2026-02-27')).days))
actual_0211 = 125948   # 實際系統打造初期
actual_end  = 128587   # 實際 2026-02-26

print(f"  對比（系統打造初 2/11）：")
print(f"    模擬（無VOO）：   ${snapshots[sim_0211]:>12,.0f}")
print(f"    實際：            ${actual_0211:>12,.0f}")
print(f"    差距：            ${snapshots[sim_0211]-actual_0211:>+12,.0f}")
print()
print(f"  對比（2/26~2/27）：")
print(f"    模擬（無VOO）：   ${snapshots[sim_end]:>12,.0f}")
print(f"    實際：            ${actual_end:>12,.0f}")
print(f"    差距：            ${snapshots[sim_end]-actual_end:>+12,.0f}")
