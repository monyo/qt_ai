"""
_oscillator_dip_backtest.py

回測「震盪型股逢低買入」策略

震盪型股識別（再平衡點動態判斷）：
  1. 近 6 個再平衡點排名標準差 > UNSTABLE_THR（確認真正震盪）
  2. 252 日動能 > 0（長期趨勢未破）
  3. 近 4 期曾進入 top 50（基本面有被市場認可）

逢低買入條件：
  - 目前動能排名 > WEAK_RANK（目前在低谷）
  - 距近 42 日高點跌幅 > DIP_MIN（確認有跌）

持有 / 出場：
  - 持滿 HOLD_MAX 天後出場（最長持有）
  - 252 日動能轉負 → 長期趨勢破壞，提前出場
  - 固定停損 -15%

比較：
  A  標準動能（現行系統，TOP5，63 天）
  B1 逢低買入，持 63 天（對等比較）
  B2 逢低買入，持 126 天（讓利潤跑更久）
  C  SPY B&H
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
symbols  = list(df_close.columns)
dates    = df_close.index
print(f"  {len(symbols)} 支  /  {len(dates)} 日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 參數 ───────────────────────────────────────────────────────────────────
MOM_SHORT     = 21
MOM_LONG      = 252
REBAL_FREQ    = 21
TOP_N         = 5
FIXED_STOP    = 0.15
TRAIL_STOP    = 0.25
HOLD_A        = 63
HOLD_B1       = 63
HOLD_B2       = 126
STABILITY_LB  = 6      # 近 6 期（約 3 個月）計算排名 std
UNSTABLE_THR  = 50     # 排名 std > 此值 = 震盪型
RECENT_LB     = 4      # 「近期曾強」的判斷期數
RECENT_STRONG = 50     # 近期曾進入此名次以內
WEAK_RANK     = 100    # 逢低買入：目前排名需 > 此值
DIP_MIN       = 0.08   # 距近 42 日高點跌幅需 > 8%
DIP_WINDOW    = 42     # 計算「近期高點」的天數

START_T = MOM_LONG + REBAL_FREQ * (STABILITY_LB + 2) + DIP_WINDOW + 10

close_arr = {s: df_close[s].values.astype(float) for s in symbols}
spy_c     = close_arr.get("SPY")

# ── 預計算動能排名 ───────────────────────────────────────────────────────────
print("預計算動能排名...")
rebal_times = []
t = MOM_LONG + 10
while t < len(dates):
    rebal_times.append(t)
    t += REBAL_FREQ

all_scores, all_ranks = {}, {}
for t in rebal_times:
    sc = {}
    for s in symbols:
        c = close_arr[s]
        if t >= len(c) or np.isnan(c[t]) or np.isnan(c[t-MOM_SHORT]) or np.isnan(c[t-MOM_LONG]):
            continue
        m_s = c[t] / c[t-MOM_SHORT] - 1
        m_l = c[t] / c[t-MOM_LONG]  - 1
        sc[s] = (0.5*m_s + 0.5*m_l, m_s, m_l)
    all_scores[t] = sc
    sorted_syms = sorted(sc, key=lambda s: sc[s][0], reverse=True)
    all_ranks[t] = {s: (i+1) for i, s in enumerate(sorted_syms)}

rebal_idx = {t: i for i, t in enumerate(rebal_times)}
print(f"  完成，{len(rebal_times)} 個再平衡點")

# ── 輔助：排名穩定性 ────────────────────────────────────────────────────────
def rank_std(sym, ri):
    start = max(0, ri - STABILITY_LB)
    ranks = [all_ranks[rebal_times[i]][sym]
             for i in range(start, ri)
             if sym in all_ranks[rebal_times[i]]]
    return np.std(ranks) if len(ranks) >= STABILITY_LB // 2 else None

def recently_strong(sym, ri):
    start = max(0, ri - RECENT_LB)
    return any(
        sym in all_ranks[rebal_times[i]] and
        all_ranks[rebal_times[i]][sym] <= RECENT_STRONG
        for i in range(start, ri)
    )

# ── 回測引擎 ─────────────────────────────────────────────────────────────────
def simulate(strategy):
    rets, n_rebal_empty = [], 0
    hold_days = {"A": HOLD_A, "B1": HOLD_B1, "B2": HOLD_B2}[strategy]
    valid = [t for t in rebal_times if t >= START_T and t + hold_days < len(dates)]

    for t0 in valid:
        ri     = rebal_idx[t0]
        sc_now = all_scores.get(t0, {})
        rk_now = all_ranks.get(t0, {})

        if strategy == "A":
            picks = [s for s, _ in
                     sorted(sc_now.items(), key=lambda x: x[1][0], reverse=True)[:TOP_N]]

        else:  # B1 / B2
            cands = []
            for sym, (composite, m_s, m_l) in sc_now.items():
                rank_now = rk_now.get(sym)
                if rank_now is None or rank_now < WEAK_RANK:
                    continue
                if m_l <= 0:                         # 長期趨勢需正
                    continue
                rs = rank_std(sym, ri)
                if rs is None or rs < UNSTABLE_THR:  # 需為震盪型
                    continue
                if not recently_strong(sym, ri):     # 近期需曾強勢
                    continue

                c = close_arr[sym]
                if t0 < DIP_WINDOW or np.isnan(c[t0]):
                    continue
                high_42 = np.nanmax(c[t0-DIP_WINDOW:t0+1])
                dip = (c[t0] - high_42) / high_42    # 負值
                if dip > -DIP_MIN:                   # 跌幅不夠
                    continue

                score = rs * 0.5 + rank_now * 0.3 + (-m_s) * 100 * 0.2
                cands.append((score, sym))

            cands.sort(reverse=True)
            picks = [s for _, s in cands[:TOP_N]]

        if not picks:
            n_rebal_empty += 1
            continue

        for sym in picks:
            c = close_arr[sym]
            entry = c[t0]
            if np.isnan(entry) or entry <= 0:
                continue

            high_px = entry
            fixed_s = entry * (1 - FIXED_STOP)
            exit_r  = None

            for dt in range(1, hold_days + 1):
                ti = t0 + dt
                if ti >= len(dates): break
                px = c[ti]
                if np.isnan(px): break

                high_px  = max(high_px, px)
                trail_s  = high_px * (1 - TRAIL_STOP)
                eff_stop = max(fixed_s, trail_s)

                if px < eff_stop:
                    exit_r = px / entry - 1
                    break

                # B2：長期趨勢破壞則提前出場
                if strategy == "B2" and ti >= MOM_LONG:
                    m_l_now = px / c[ti-MOM_LONG] - 1
                    if m_l_now < 0:
                        exit_r = px / entry - 1
                        break

            if exit_r is None:
                exit_r = c[min(t0 + hold_days, len(dates)-1)] / entry - 1
            rets.append(exit_r)

    return np.array(rets), n_rebal_empty


print("\n回測中...")
rets_A,  e_A  = simulate("A")
print(f"  A  標準動能完成（{len(rets_A)} 筆）")
rets_B1, e_B1 = simulate("B1")
print(f"  B1 逢低63天完成（{len(rets_B1)} 筆，{e_B1} 個再平衡點無候選）")
rets_B2, e_B2 = simulate("B2")
print(f"  B2 逢低126天完成（{len(rets_B2)} 筆，{e_B2} 個再平衡點無候選）")

# SPY B&H（以 HOLD_A 為單位）
spy_pts  = [spy_c[t+HOLD_A]/spy_c[t]-1
            for t in rebal_times
            if t >= START_T and t+HOLD_A < len(dates) and spy_c is not None]
spy_mean = np.mean(spy_pts)*100 if spy_pts else 0

def stats(arr, label, hold_label=""):
    if len(arr) == 0:
        print(f"  {label}: 無資料")
        return
    mean   = arr.mean()*100
    median = np.median(arr)*100
    win    = (arr > 0).mean()*100
    p10    = np.percentile(arr, 10)*100
    p25    = np.percentile(arr, 25)*100
    p75    = np.percentile(arr, 75)*100
    p90    = np.percentile(arr, 90)*100
    alpha  = mean - spy_mean
    print(f"  {label:<34} 平均:{mean:>+6.2f}% 中位:{median:>+6.2f}% "
          f"勝率:{win:>5.1f}% vs SPY:{alpha:>+6.2f}% "
          f"P25={p25:+.1f}% P75={p75:+.1f}% P90={p90:+.1f}% n={len(arr)}")

print()
print("=" * 115)
print(f"  震盪門檻 std>{UNSTABLE_THR}（近{STABILITY_LB}期）  /  跌幅門檻 -{DIP_MIN*100:.0f}% 距{DIP_WINDOW}日高  /  長期動能>0")
print("-" * 115)
stats(rets_A,  f"A  標準動能（TOP{TOP_N}，{HOLD_A}天）")
stats(rets_B1, f"B1 逢低買入（震盪股，{HOLD_B1}天持有）")
stats(rets_B2, f"B2 逢低買入（震盪股，{HOLD_B2}天持有）")
print(f"  {'C  SPY B&H':<34} 平均:{spy_mean:>+6.2f}%（基準）")
print("=" * 115)

# ── 當下掃描：震盪型股 + 逢低訊號 ─────────────────────────────────────────
print()
print("── 當下「震盪型股」掃描 ─────────────────────────────────────────────")
last_t  = rebal_times[-1]
last_ri = rebal_idx[last_t]
sc_last = all_scores.get(last_t, {})
rk_last = all_ranks.get(last_t, {})

rows = []
for sym, (composite, m_s, m_l) in sc_last.items():
    rank_now = rk_last.get(sym)
    if rank_now is None: continue
    rs = rank_std(sym, last_ri)
    if rs is None or rs < UNSTABLE_THR: continue
    if m_l <= 0: continue

    c = close_arr[sym]
    t0 = last_t
    high_42 = np.nanmax(c[t0-DIP_WINDOW:t0+1]) if t0 >= DIP_WINDOW else np.nan
    dip = (c[t0] - high_42) / high_42 if not np.isnan(high_42) else 0

    dip_signal = (rank_now > WEAK_RANK and dip < -DIP_MIN and recently_strong(sym, last_ri))
    rows.append({
        "sym": sym, "rank": rank_now, "std": round(rs,1),
        "mom_21": round(m_s*100,1), "mom_252": round(m_l*100,1),
        "dip": round(dip*100,1), "signal": dip_signal,
    })

rows.sort(key=lambda x: (0 if x["signal"] else 1, -x["std"]))
print(f"  {'標的':6}  {'排名':>5}  {'std':>6}  {'21d動能':>8}  {'252d動能':>9}  {'距42高':>7}  {'訊號':>8}")
print(f"  {'-'*70}")
for r in rows[:20]:
    sig = "⬇ 逢低買" if r["signal"] else ""
    print(f"  {r['sym']:6}  #{r['rank']:>4}  std={r['std']:>5.1f}"
          f"  {r['mom_21']:>+7.1f}%  {r['mom_252']:>+8.1f}%"
          f"  {r['dip']:>+6.1f}%  {sig}")
