"""
_contrarian_momentum_backtest.py

回測「動能不穩股逆向操作」vs 標準動能 vs SPY B&H

策略 B（逆向）識別邏輯：
  1. 過去 12 個再平衡點，動能排名標準差 > UNSTABLE_THR（排名波動大）
  2. 目前排名 > CTR_WEAK_RANK（目前處於弱勢）
  3. 近 4 個再平衡點曾進入 top CTR_RECENT_STRONG（近期曾強勢）
  4. 252 日動能 > 0（長期趨勢未壞）

買入：滿足上述條件，按「排名波動幅度 × 目前超賣程度」排序取 TOP_N
賣出：21 日動能 > +CTR_SELL_MOM（衝高出場）或持滿 HOLD_DAYS_B 或停損 -15%
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
MOM_SHORT         = 21
MOM_LONG          = 252
REBAL_FREQ        = 21
TOP_N             = 5
HOLD_DAYS_A       = 63     # 標準動能持有天數
HOLD_DAYS_B       = 42     # 逆向持有天數（均值回歸較快）
FIXED_STOP        = 0.15
TRAIL_STOP        = 0.25
STABILITY_LB      = 6      # 排名穩定性計算用再平衡期數（只看近 3 個月）
UNSTABLE_THR      = 50     # 排名 std > 此值 = 動能不穩
CTR_WEAK_RANK     = 100    # 逆向買入：目前排名需 > 此值
CTR_RECENT_STRONG = 50     # 近期曾進入此名次以內
CTR_RECENT_LB     = 4      # 「近期」的定義（再平衡期數）
CTR_SELL_MOM      = 0.12   # 逆向出場：21d 動能 > +12%
CTR_MIN_LONG_MOM  = 0.0    # 長期動能需 > 0

START_T = MOM_LONG + REBAL_FREQ * (STABILITY_LB + 2) + 10

close_arr = {s: df_close[s].values.astype(float) for s in symbols}
spy_c     = close_arr.get("SPY")

# ── 預計算每個再平衡點的動能分數與排名 ─────────────────────────────────────
print("預計算動能排名（所有時間點）...")

rebal_times = []
t = MOM_LONG + 10
while t < len(dates):
    rebal_times.append(t)
    t += REBAL_FREQ

all_scores = {}   # t -> {sym: (composite, m_short, m_long)}
all_ranks  = {}   # t -> {sym: rank}

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

print(f"  完成，{len(rebal_times)} 個再平衡點")

# ── 排名穩定性查詢 ──────────────────────────────────────────────────────────
rebal_idx_map = {t: i for i, t in enumerate(rebal_times)}

def get_rank_std(sym, ri):
    start = max(0, ri - STABILITY_LB)
    ranks = [all_ranks[rebal_times[i]][sym]
             for i in range(start, ri)
             if sym in all_ranks[rebal_times[i]]]
    if len(ranks) < STABILITY_LB // 2:
        return None
    return np.std(ranks)

def was_recently_strong(sym, ri):
    start = max(0, ri - CTR_RECENT_LB)
    return any(
        sym in all_ranks[rebal_times[i]] and
        all_ranks[rebal_times[i]][sym] <= CTR_RECENT_STRONG
        for i in range(start, ri)
    )

# ── 回測引擎 ─────────────────────────────────────────────────────────────────
def sim_strategy(strategy):
    rets = []
    hold_days = HOLD_DAYS_A if strategy == "A" else HOLD_DAYS_B
    valid = [t for t in rebal_times if t >= START_T and t + hold_days < len(dates)]

    for t0 in valid:
        ri      = rebal_idx_map[t0]
        sc_now  = all_scores.get(t0, {})
        rk_now  = all_ranks.get(t0, {})

        if strategy == "A":
            picks = [s for s, _ in
                     sorted(sc_now.items(), key=lambda x: x[1][0], reverse=True)[:TOP_N]]

        else:  # B: contrarian
            cands = []
            for sym, (composite, m_s, m_l) in sc_now.items():
                rank_now = rk_now.get(sym)
                if rank_now is None or rank_now < CTR_WEAK_RANK:
                    continue
                if m_l < CTR_MIN_LONG_MOM:
                    continue
                rank_std = get_rank_std(sym, ri)
                if rank_std is None or rank_std < UNSTABLE_THR:
                    continue
                if not was_recently_strong(sym, ri):
                    continue
                # 分數：動能越不穩 + 目前越超賣 + 短期動能越弱 → 越優先
                score = rank_std * 0.5 + rank_now * 0.3 + (-m_s) * 100 * 0.2
                cands.append((score, sym))
            cands.sort(reverse=True)
            picks = [s for _, s in cands[:TOP_N]]

        if not picks:
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
                if ti >= len(dates):
                    break
                px = c[ti]
                if np.isnan(px):
                    break
                high_px  = max(high_px, px)
                trail_s  = high_px * (1 - TRAIL_STOP)
                eff_stop = max(fixed_s, trail_s)

                if px < eff_stop:
                    exit_r = px / entry - 1
                    break

                if strategy == "B" and ti - t0 >= MOM_SHORT:
                    m_s_now = px / c[ti - MOM_SHORT] - 1
                    if m_s_now > CTR_SELL_MOM:
                        exit_r = px / entry - 1
                        break

            if exit_r is None:
                exit_r = c[min(t0 + hold_days, len(dates) - 1)] / entry - 1
            rets.append(exit_r)

    return np.array(rets)


print("\n回測中...")
rets_A = sim_strategy("A")
print(f"  A 標準動能完成（{len(rets_A)} 筆）")
rets_B = sim_strategy("B")
print(f"  B 逆向操作完成（{len(rets_B)} 筆）")

# SPY B&H
spy_pts = [spy_c[t + HOLD_DAYS_A] / spy_c[t] - 1
           for t in rebal_times
           if t >= START_T and t + HOLD_DAYS_A < len(dates) and spy_c is not None]
spy_mean = np.mean(spy_pts) * 100 if spy_pts else 0

def stats(arr, label):
    if len(arr) == 0:
        print(f"  {label}: 無資料（候選股不足）")
        return
    mean   = arr.mean() * 100
    median = np.median(arr) * 100
    win    = (arr > 0).mean() * 100
    p10    = np.percentile(arr, 10) * 100
    p90    = np.percentile(arr, 90) * 100
    alpha  = mean - spy_mean
    print(f"  {label:<32}  平均:{mean:>+6.2f}%  中位:{median:>+6.2f}%  "
          f"勝率:{win:>5.1f}%  vs SPY:{alpha:>+6.2f}%  P10={p10:+.1f}%  P90={p90:+.1f}%  n={len(arr)}")

print()
print("=" * 108)
print(f"  排名不穩門檻 std>{UNSTABLE_THR}  /  弱勢進場 rank>{CTR_WEAK_RANK}"
      f"  /  逆向出場 21d動能>{CTR_SELL_MOM*100:.0f}% 或持{HOLD_DAYS_B}天")
print("-" * 108)
stats(rets_A, f"A 標準動能（TOP{TOP_N}，{HOLD_DAYS_A}天持有）")
stats(rets_B, f"B 逆向操作（動能不穩股，{HOLD_DAYS_B}天持有）")
print(f"  {'C SPY B&H':<32}  平均:{spy_mean:>+6.2f}%  （基準）")
print("=" * 108)

print()
print("分布比較（A vs B）：")
for label, arr in [("A 標準動能", rets_A), ("B 逆向操作", rets_B)]:
    if len(arr) == 0:
        continue
    p10 = np.percentile(arr, 10) * 100
    p25 = np.percentile(arr, 25) * 100
    p75 = np.percentile(arr, 75) * 100
    p90 = np.percentile(arr, 90) * 100
    print(f"  {label}:  P10={p10:+.1f}%  P25={p25:+.1f}%  P75={p75:+.1f}%  P90={p90:+.1f}%")

# ── 顯示目前「動能不穩」股清單（當下診斷）─────────────────────────────────
print()
print("── 當下「動能不穩」股掃描（最新一個再平衡點）─────────────────────")
last_t  = rebal_times[-1]
last_ri = rebal_idx_map[last_t]
sc_last = all_scores.get(last_t, {})
rk_last = all_ranks.get(last_t, {})

unstable_now = []
for sym, (composite, m_s, m_l) in sc_last.items():
    rank_now = rk_last.get(sym)
    if rank_now is None:
        continue
    rank_std = get_rank_std(sym, last_ri)
    if rank_std is None or rank_std < UNSTABLE_THR:
        continue
    recently_strong = was_recently_strong(sym, last_ri)
    unstable_now.append({
        "symbol":   sym,
        "rank_now": rank_now,
        "rank_std": round(rank_std, 1),
        "mom_21d":  round(m_s * 100, 1),
        "mom_252d": round(m_l * 100, 1),
        "long_ok":  m_l > CTR_MIN_LONG_MOM,
        "was_strong": recently_strong,
        "buy_signal": rank_now > CTR_WEAK_RANK and m_l > CTR_MIN_LONG_MOM and recently_strong,
    })

unstable_now.sort(key=lambda x: x["rank_std"], reverse=True)
print(f"  {'標的':6}  {'現排名':>6}  {'排名std':>7}  {'21日動能':>8}  {'252日動能':>9}  {'近期強過':>8}  {'買入訊號':>8}")
print(f"  {'-'*70}")
for r in unstable_now[:20]:
    signal = "⬆ 逆向買" if r["buy_signal"] else ""
    strong = "✓" if r["was_strong"] else "✗"
    print(f"  {r['symbol']:6}  #{r['rank_now']:>4}    std={r['rank_std']:>5.1f}"
          f"  {r['mom_21d']:>+7.1f}%  {r['mom_252d']:>+8.1f}%"
          f"  {strong:>8}  {signal}")
