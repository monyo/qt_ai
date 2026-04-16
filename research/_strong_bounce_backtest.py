"""
_strong_bounce_backtest.py

回測問題：當股票距40日高點 -15%~-25%（轉弱格局），
         且單日出現強彈（+8% 以上），
         往後能回到前高的機率有多高？需要多久？

使用：
    conda run -n qt_env python _strong_bounce_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pickle
from datetime import timedelta

# ── 載入價格快取 ──────────────────────────────────────────────────────────────
CACHE = "data/_protection_bt_prices.pkl"
print("載入價格快取...")
with open(CACHE, "rb") as f:
    data = pickle.load(f)

# data 格式：{symbol: pd.Series(Close, index=DatetimeIndex)}
# 轉成 numpy array 方便計算
symbols = list(data.keys())
dates   = data[symbols[0]].index
n_dates = len(dates)
print(f"  {len(symbols)} 支股票  /  {n_dates} 個交易日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 參數 ──────────────────────────────────────────────────────────────────────
HIGH_WINDOW    = 40       # 前高計算窗口（天）
FROM_HIGH_MIN  = -0.25    # 距前高下限（更深）
FROM_HIGH_MAX  = -0.15    # 距前高上限（較淺）
BOUNCE_THRESH  = 0.08     # 強彈門檻（單日漲幅）
FORWARD_DAYS   = [5, 10, 21, 42, 63]   # 往後追蹤天數
RECOVERY_PCT   = 0.00     # 回到前高的定義（0% = 觸及前高，-0.05 = 到前高的 95%）

# ── 掃描觸發事件 ──────────────────────────────────────────────────────────────
events = []          # 觸發強彈事件
controls = []        # 對照組：轉弱但無強彈

print("\n掃描觸發事件（距前高 -15%~-25% + 單日 ≥+8%）...")

for sym in symbols:
    closes = data[sym].values.astype(float)
    n = len(closes)

    for t in range(HIGH_WINDOW + 1, n - max(FORWARD_DAYS) - 1):
        p0   = closes[t]
        p1   = closes[t - 1]
        if p0 <= 0 or p1 <= 0:
            continue

        day_ret   = p0 / p1 - 1
        high_40d  = np.max(closes[t - HIGH_WINDOW: t])

        if high_40d <= 0:
            continue

        from_high = p0 / high_40d - 1

        # 必須在轉弱區間
        if not (FROM_HIGH_MIN <= from_high <= FROM_HIGH_MAX):
            continue

        # 對照組：轉弱 + 平靜（無強彈，單日漲跌在 -2% ~ +2%）
        if abs(day_ret) < 0.02:
            # 降低採樣頻率，避免對照組過多（每5天取一個）
            if t % 5 == 0:
                controls.append({
                    "sym": sym, "t": t,
                    "p0": p0, "high_40d": high_40d,
                    "from_high": from_high,
                    "day_ret": day_ret,
                    "closes_fwd": closes[t: t + max(FORWARD_DAYS) + 1],
                })
            continue

        # 主要組：強彈
        if day_ret >= BOUNCE_THRESH:
            events.append({
                "sym": sym, "t": t,
                "date": dates[t],
                "p0": p0, "high_40d": high_40d,
                "from_high": from_high,
                "day_ret": day_ret,
                "closes_fwd": closes[t: t + max(FORWARD_DAYS) + 1],
            })

print(f"  強彈事件: {len(events)} 筆")
print(f"  對照組:   {len(controls)} 筆")

# ── 分析：往後各時間點的報酬 & 回到前高機率 ────────────────────────────────────
def analyze(group, label):
    if not group:
        return
    print(f"\n{'='*62}")
    print(f"  {label}  (N={len(group)})")
    print(f"{'='*62}")

    # 往後 N 天報酬分佈
    print(f"\n  往後平均報酬（以觸發日收盤為基準）：")
    print(f"  {'天數':>6}  {'平均':>8}  {'中位':>8}  {'>0%機率':>9}  {'>+5%機率':>9}")
    print(f"  {'-'*50}")
    for fwd in FORWARD_DAYS:
        rets = []
        for ev in group:
            cf = ev["closes_fwd"]
            if len(cf) > fwd:
                r = cf[fwd] / cf[0] - 1
                rets.append(r)
        if not rets:
            continue
        rets = np.array(rets)
        print(f"  {fwd:>6}天  {np.mean(rets)*100:>+7.1f}%  "
              f"{np.median(rets)*100:>+7.1f}%  "
              f"{np.mean(rets>0)*100:>8.1f}%  "
              f"{np.mean(rets>0.05)*100:>8.1f}%")

    # 回到前高的機率與時間
    print(f"\n  回到前高（{RECOVERY_PCT*100:.0f}%）的統計：")
    reach_days = []
    never_count = 0
    for ev in group:
        cf   = ev["closes_fwd"]
        tgt  = ev["high_40d"] * (1 + RECOVERY_PCT)
        days_to = None
        for d in range(1, len(cf)):
            if cf[d] >= tgt:
                days_to = d
                break
        if days_to is not None:
            reach_days.append(days_to)
        else:
            never_count += 1

    total   = len(group)
    reached = len(reach_days)
    print(f"  回到前高機率（{max(FORWARD_DAYS)}交易日內）: {reached/total*100:.1f}%  ({reached}/{total})")
    print(f"  從未回到前高（觀察期內）:                {never_count/total*100:.1f}%  ({never_count}/{total})")
    if reach_days:
        rd = np.array(reach_days)
        print(f"  回到前高所需天數：")
        print(f"    中位數: {np.median(rd):.0f} 交易日  "
              f"平均: {np.mean(rd):.0f} 交易日")
        print(f"    25%分位: {np.percentile(rd,25):.0f}天  "
              f"75%分位: {np.percentile(rd,75):.0f}天")

        # 時間分佈
        buckets = [(0,5),(6,10),(11,21),(22,42),(43,63)]
        print(f"\n  時間分佈（回到前高的那些事件）：")
        for lo, hi in buckets:
            cnt = np.sum((rd >= lo) & (rd <= hi))
            print(f"    {lo:>2}~{hi:>2} 天: {cnt:>4} 件  ({cnt/reached*100:.1f}%)")

    # 依強彈幅度分層
    if label.startswith("強彈"):
        print(f"\n  依強彈幅度分層（回到前高機率）：")
        buckets_ret = [(0.08,0.12),(0.12,0.20),(0.20,1.0)]
        labels_ret  = ["+8~12%","+12~20%","+20%以上"]
        for (lo, hi), lb in zip(buckets_ret, labels_ret):
            sub = [ev for ev in group if lo <= ev["day_ret"] < hi]
            if not sub:
                continue
            sub_reach = sum(
                1 for ev in sub
                if any(ev["closes_fwd"][d] >= ev["high_40d"]*(1+RECOVERY_PCT)
                       for d in range(1, len(ev["closes_fwd"])))
            )
            avg_ret_21 = np.mean([
                ev["closes_fwd"][21]/ev["closes_fwd"][0]-1
                for ev in sub if len(ev["closes_fwd"]) > 21
            ])
            print(f"    {lb}: N={len(sub):>4}  回前高率={sub_reach/len(sub)*100:.1f}%  "
                  f"21天平均報酬={avg_ret_21*100:+.1f}%")

    # 依觸發時距前高深度分層
    print(f"\n  依觸發時距前高深度分層（回到前高機率）：")
    depth_buckets = [(-0.17,-0.15),(-0.20,-0.17),(-0.25,-0.20)]
    depth_labels  = ["-15~-17%","-17~-20%","-20~-25%"]
    for (lo, hi), lb in zip(depth_buckets, depth_labels):
        sub = [ev for ev in group if lo <= ev["from_high"] < hi]
        if not sub:
            continue
        sub_reach = sum(
            1 for ev in sub
            if any(ev["closes_fwd"][d] >= ev["high_40d"]*(1+RECOVERY_PCT)
                   for d in range(1, len(ev["closes_fwd"])))
        )
        avg_ret_21 = np.mean([
            ev["closes_fwd"][21]/ev["closes_fwd"][0]-1
            for ev in sub if len(ev["closes_fwd"]) > 21
        ])
        print(f"    {lb}: N={len(sub):>4}  回前高率={sub_reach/len(sub)*100:.1f}%  "
              f"21天平均報酬={avg_ret_21*100:+.1f}%")


analyze(events,   "強彈組（距前高 -15~-25% + 單日 ≥+8%）")
analyze(controls, "對照組（距前高 -15~-25% + 平靜日 ±2%）")

# ── 總結 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  結論摘要")
print(f"{'='*62}")
print(f"  觀測期: {dates[0].date()} ~ {dates[-1].date()}")
print(f"  強彈定義: 距前高 -{abs(FROM_HIGH_MIN)*100:.0f}%~-{abs(FROM_HIGH_MAX)*100:.0f}% + 單日 ≥+{BOUNCE_THRESH*100:.0f}%")
print(f"  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股")
