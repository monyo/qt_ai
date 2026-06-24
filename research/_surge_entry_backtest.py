"""
_surge_entry_backtest.py

比較「爆衝當天收盤進場（T日）」vs「隔天進場（T+1日，系統確認後）」的報酬差異

事件定義：
  1. T 日單日漲幅 > SURGE_THR
  2. T+1 日動能分數進入全市場前 TOP_N_CONFIRM（系統隔天會看到）

策略：
  A：T 日收盤進場
  B：T+1 日收盤進場（模擬現行系統行為）

出場：固定停損 -15% + 追蹤停損 -25%，最長持有 HOLD_DAYS 天
基準：SPY B&H（相同持有期）
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd

OHLCV_PATH  = "data/_protection_bt_ohlcv.pkl"
ohlcv       = pd.read_pickle(OHLCV_PATH)
df_close    = ohlcv["Close"]
symbols     = list(df_close.columns)
dates       = list(df_close.index)
close_arr   = {s: df_close[s].values.astype(float) for s in symbols}
spy_c       = close_arr.get("SPY")
print(f"資料：{len(symbols)} 支  {dates[0].date()} ~ {dates[-1].date()}")

# ── 參數 ─────────────────────────────────────────────────────────────────────
SURGE_THR     = 0.15    # T 日漲幅門檻（15%）
TOP_N_CONFIRM = 50      # T+1 動能需進入全市場前幾名才算「系統確認」
MOM_SHORT     = 21
MOM_LONG      = 252
FIXED_STOP    = 0.15
TRAIL_STOP    = 0.25
HOLD_DAYS     = 42      # 持有天數

START_T = MOM_LONG + 10

# ── 預計算每日動能分數與排名 ──────────────────────────────────────────────────
print("預計算每日動能分數...")
daily_rank = {}   # date_idx -> {sym: rank}

for t in range(START_T, len(dates)):
    sc = {}
    for s in symbols:
        c = close_arr[s]
        if t >= len(c) or np.isnan(c[t]) or np.isnan(c[t-MOM_SHORT]) or np.isnan(c[t-MOM_LONG]):
            continue
        m_s = c[t] / c[t-MOM_SHORT] - 1
        m_l = c[t] / c[t-MOM_LONG]  - 1
        sc[s] = 0.5*m_s + 0.5*m_l
    sorted_syms = sorted(sc, key=lambda s: sc[s], reverse=True)
    daily_rank[t] = {s: (i+1) for i, s in enumerate(sorted_syms)}

print(f"  完成，{len(daily_rank)} 個交易日")

# ── 找「爆衝 + 隔天系統確認」事件 ────────────────────────────────────────────
print(f"\n掃描爆衝事件（T日漲幅 > {SURGE_THR*100:.0f}% + T+1進前{TOP_N_CONFIRM}名）...")

events = []   # (sym, t_surge, t_entry_A, t_entry_B)

for s in symbols:
    c = close_arr[s]
    for t in range(START_T + 1, len(dates) - HOLD_DAYS - 2):
        if np.isnan(c[t]) or np.isnan(c[t-1]) or c[t-1] <= 0:
            continue
        daily_ret = c[t] / c[t-1] - 1
        if daily_ret < SURGE_THR:
            continue
        # T+1 動能確認
        t1 = t + 1
        if t1 not in daily_rank:
            continue
        rank_t1 = daily_rank[t1].get(s)
        if rank_t1 is None or rank_t1 > TOP_N_CONFIRM:
            continue
        events.append((s, t, t, t1))   # (sym, surge_day, entry_A=T, entry_B=T+1)

print(f"  找到 {len(events)} 個事件")

# ── 回測單一事件 ──────────────────────────────────────────────────────────────
def sim_trade(sym, entry_t):
    c = close_arr[sym]
    entry = c[entry_t]
    if np.isnan(entry) or entry <= 0:
        return None
    high_px = entry
    fixed_s = entry * (1 - FIXED_STOP)
    exit_r  = None
    for dt in range(1, HOLD_DAYS + 1):
        ti = entry_t + dt
        if ti >= len(dates): break
        px = c[ti]
        if np.isnan(px): break
        high_px  = max(high_px, px)
        trail_s  = high_px * (1 - TRAIL_STOP)
        eff_stop = max(fixed_s, trail_s)
        if px < eff_stop:
            exit_r = px / entry - 1
            break
    if exit_r is None:
        exit_r = c[min(entry_t + HOLD_DAYS, len(dates)-1)] / entry - 1
    return exit_r

# ── 執行回測 ──────────────────────────────────────────────────────────────────
print("回測中...")
rets_A, rets_B = [], []
entry_diffs = []   # B 比 A 貴多少 %

for sym, t_surge, t_A, t_B in events:
    c = close_arr[sym]
    if np.isnan(c[t_A]) or np.isnan(c[t_B]) or c[t_A] <= 0:
        continue
    r_A = sim_trade(sym, t_A)
    r_B = sim_trade(sym, t_B)
    if r_A is None or r_B is None:
        continue
    rets_A.append(r_A)
    rets_B.append(r_B)
    entry_diffs.append(c[t_B] / c[t_A] - 1)   # T+1 比 T 貴多少

rets_A = np.array(rets_A)
rets_B = np.array(rets_B)
entry_diffs = np.array(entry_diffs)

# SPY 基準
spy_pts = []
for _, _, t_A, _ in events:
    if t_A + HOLD_DAYS < len(dates) and spy_c is not None:
        r = spy_c[t_A + HOLD_DAYS] / spy_c[t_A] - 1
        if not np.isnan(r):
            spy_pts.append(r)
spy_mean = np.mean(spy_pts) * 100 if spy_pts else 0

# ── 結果輸出 ──────────────────────────────────────────────────────────────────
def stats(arr, label):
    if len(arr) == 0:
        print(f"  {label}: 無資料")
        return
    mean   = arr.mean() * 100
    median = np.median(arr) * 100
    win    = (arr > 0).mean() * 100
    p25    = np.percentile(arr, 25) * 100
    p75    = np.percentile(arr, 75) * 100
    alpha  = mean - spy_mean
    print(f"  {label:<38} 平均:{mean:>+6.2f}%  中位:{median:>+6.2f}%  "
          f"勝率:{win:>5.1f}%  vs SPY:{alpha:>+6.2f}%  "
          f"P25={p25:+.1f}%  P75={p75:+.1f}%  n={len(arr)}")

print()
print("=" * 105)
print(f"  爆衝門檻 >{SURGE_THR*100:.0f}%  |  系統確認前{TOP_N_CONFIRM}名  |  持有{HOLD_DAYS}天  |  停損 -{FIXED_STOP*100:.0f}% / -{TRAIL_STOP*100:.0f}%")
print("-" * 105)
stats(rets_A, f"A  T日收盤進場（爆衝當天）")
stats(rets_B, f"B  T+1收盤進場（隔天系統確認）")
print(f"  {'SPY B&H 基準':<38} 平均:{spy_mean:>+6.2f}%")
print("-" * 105)

# 進場價差
pos_diff = (entry_diffs > 0).mean() * 100
mean_diff = entry_diffs.mean() * 100
med_diff  = np.median(entry_diffs) * 100
print(f"  T+1 比 T 貴：平均 {mean_diff:+.2f}%  中位 {med_diff:+.2f}%  "
      f"（{pos_diff:.1f}% 的事件 T+1 更貴）")

# A vs B 直接比較
diff_AB = rets_A - rets_B   # A 比 B 多賺多少
print(f"  A 比 B 多賺：平均 {diff_AB.mean()*100:+.2f}%  中位 {np.median(diff_AB)*100:+.2f}%  "
      f"A 更好的比例 {(diff_AB > 0).mean()*100:.1f}%")
print("=" * 105)

# ── 不同爆衝門檻敏感度 ───────────────────────────────────────────────────────
print()
print("── 爆衝門檻敏感度（TOP_N_CONFIRM=50，持有42天）────────────────────────────────")
print(f"  {'門檻':>6}  {'事件數':>6}  {'A平均':>8}  {'B平均':>8}  {'A-B差':>8}  {'進場價差（中位）':>14}")
for thr in [0.08, 0.10, 0.15, 0.20, 0.25]:
    ev, ra, rb, ed = [], [], [], []
    for s in symbols:
        c = close_arr[s]
        for t in range(START_T + 1, len(dates) - HOLD_DAYS - 2):
            if np.isnan(c[t]) or np.isnan(c[t-1]) or c[t-1] <= 0: continue
            if c[t]/c[t-1]-1 < thr: continue
            t1 = t + 1
            if t1 not in daily_rank: continue
            rk = daily_rank[t1].get(s)
            if rk is None or rk > TOP_N_CONFIRM: continue
            rA = sim_trade(s, t)
            rB = sim_trade(s, t1)
            if rA is None or rB is None: continue
            ra.append(rA); rb.append(rB)
            ed.append(c[t1]/c[t]-1)
            ev.append(1)
    if not ra: continue
    ra, rb, ed = np.array(ra), np.array(rb), np.array(ed)
    print(f"  >{thr*100:>4.0f}%  {len(ev):>6}  "
          f"{ra.mean()*100:>+7.2f}%  {rb.mean()*100:>+7.2f}%  "
          f"{(ra-rb).mean()*100:>+7.2f}%  {np.median(ed)*100:>+12.2f}%")

# ── 持有天數敏感度 ───────────────────────────────────────────────────────────
print()
print("── 持有天數敏感度（爆衝>15%，TOP_N_CONFIRM=50）───────────────────────────────")
print(f"  {'持有':>6}  {'A平均':>8}  {'B平均':>8}  {'A-B差':>8}")
for hd in [21, 42, 63, 84]:
    ra, rb = [], []
    for sym, t_surge, t_A, t_B in events:
        c = close_arr[sym]
        if np.isnan(c[t_A]) or np.isnan(c[t_B]) or c[t_A] <= 0: continue
        # 用指定 hold_days 重跑
        def sim2(s, et, hold):
            cc = close_arr[s]
            en = cc[et]
            if np.isnan(en) or en <= 0: return None
            hp = en; fs = en*(1-FIXED_STOP); er = None
            for dt in range(1, hold+1):
                ti = et+dt
                if ti >= len(dates): break
                px = cc[ti]
                if np.isnan(px): break
                hp = max(hp, px)
                if px < max(fs, hp*(1-TRAIL_STOP)): er = px/en-1; break
            return er if er is not None else cc[min(et+hold, len(dates)-1)]/en-1
        rA = sim2(sym, t_A, hd)
        rB = sim2(sym, t_B, hd)
        if rA is None or rB is None: continue
        ra.append(rA); rb.append(rB)
    if not ra: continue
    ra, rb = np.array(ra), np.array(rb)
    print(f"  {hd:>5}天  {ra.mean()*100:>+7.2f}%  {rb.mean()*100:>+7.2f}%  {(ra-rb).mean()*100:>+7.2f}%")
