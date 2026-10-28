"""
A頭（倒V）型態偵測與回測
=========================
問題：高動能股出現 A頭（短期急漲後急跌）後，後續表現如何？
      這種型態是否預測繼續下跌，還是只是正常回調後繼續上漲？

A頭定義：
  - 過去 WINDOW 天內，出現峰值
  - 峰值 = 窗口內最高點，且不在最後5天（否則還沒反轉）
  - 峰值距窗口起點漲幅 ≥ SPIKE_PCT（有一個明顯的拉升）
  - 當前價格距峰值跌幅 ≥ REVERSAL_PCT（已明顯回落）
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 參數 ──────────────────────────────────────────────────
WINDOW        = 21    # 偵測窗口（交易日）= 約1個月
SPIKE_PCT     = 0.12  # 峰值距窗口起點 ≥ +12%
REVERSAL_PCT  = 0.12  # 當前價格距峰值 ≤ -12%
PEAK_BUFFER   = 5     # 峰值不能在窗口最後幾天（確保已反轉）

# 也用於高動能篩選（比較有無A頭的差異）
MOMENTUM_WINDOW = 63  # 動能計算窗口

# 前瞻回報觀察區間
FORWARD_DAYS = [5, 10, 21, 42]

DATA_PATH = Path("data/_protection_bt_prices.pkl")
# ──────────────────────────────────────────────────────────


def detect_a_top(prices: pd.Series,
                 window=WINDOW,
                 spike_pct=SPIKE_PCT,
                 reversal_pct=REVERSAL_PCT,
                 peak_buffer=PEAK_BUFFER) -> bool:
    if len(prices) < window:
        return False
    recent = prices.iloc[-window:]
    peak_idx = int(recent.values.argmax())
    # 峰值不能在窗口末尾（還沒確認反轉）
    if peak_idx >= window - peak_buffer:
        return False
    peak_price  = float(recent.iloc[peak_idx])
    start_price = float(recent.iloc[0])
    curr_price  = float(recent.iloc[-1])
    spike     = (peak_price - start_price) / start_price
    reversal  = (curr_price  - peak_price)  / peak_price
    return spike >= spike_pct and reversal <= -reversal_pct


def main():
    print(f"載入價格資料...")
    with open(DATA_PATH, "rb") as f:
        prices_dict = pickle.load(f)

    # 轉成 DataFrame
    prices = pd.DataFrame(prices_dict).sort_index()
    print(f"  {prices.shape[1]} 支股票，{len(prices)} 個交易日")
    print(f"  範圍：{prices.index[0].date()} ～ {prices.index[-1].date()}")

    # ── 逐股、逐日掃描 ────────────────────────────────────
    print(f"\nA頭定義：窗口{WINDOW}日，漲幅≥{SPIKE_PCT*100:.0f}%，回落≥{REVERSAL_PCT*100:.0f}%")
    print("掃描中...")

    records = []
    tickers = prices.columns.tolist()
    n = len(tickers)

    for i, sym in enumerate(tickers):
        if i % 100 == 0:
            print(f"  {i}/{n}...", end="\r")
        s = prices[sym].dropna()
        if len(s) < WINDOW + max(FORWARD_DAYS):
            continue

        for t in range(WINDOW + MOMENTUM_WINDOW, len(s) - max(FORWARD_DAYS)):
            window_prices = s.iloc[t - WINDOW : t]
            is_a = detect_a_top(window_prices)

            # 動能：過去 MOMENTUM_WINDOW 天報酬
            mom_start = s.iloc[t - MOMENTUM_WINDOW]
            curr      = s.iloc[t]
            momentum  = (curr - mom_start) / mom_start if mom_start > 0 else np.nan

            rec = {
                "sym":       sym,
                "date":      s.index[t],
                "is_a_top":  is_a,
                "momentum":  momentum,
                "curr_price": curr,
            }
            for fd in FORWARD_DAYS:
                fwd = s.iloc[t + fd]
                rec[f"fwd_{fd}d"] = (fwd - curr) / curr
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"\n總樣本數：{len(df):,}  其中 A頭：{df['is_a_top'].sum():,} ({df['is_a_top'].mean()*100:.1f}%)")

    # ── 分析 1：整體 A頭 vs 非A頭 ──────────────────────────
    print("\n" + "="*60)
    print("分析 1：整體 A頭 vs 非A頭 後續報酬")
    print("="*60)
    for fd in FORWARD_DAYS:
        col = f"fwd_{fd}d"
        a   = df[df["is_a_top"]][col].dropna()
        na  = df[~df["is_a_top"]][col].dropna()
        print(f"\n  {fd}日後：")
        print(f"    A頭  中位 {a.median()*100:+.2f}%  均值 {a.mean()*100:+.2f}%  勝率 {(a>0).mean()*100:.1f}%  n={len(a):,}")
        print(f"    非A頭 中位 {na.median()*100:+.2f}%  均值 {na.mean()*100:+.2f}%  勝率 {(na>0).mean()*100:.1f}%  n={len(na):,}")
        print(f"    差距  {(a.median()-na.median())*100:+.2f}%（A頭 - 非A頭，中位）")

    # ── 分析 2：高動能股中，A頭 vs 非A頭 ───────────────────
    print("\n" + "="*60)
    print(f"分析 2：高動能股（動能前25%）中 A頭 vs 非A頭")
    print("="*60)
    mom_thresh = df["momentum"].quantile(0.75)
    high_mom = df[df["momentum"] >= mom_thresh]
    print(f"  高動能門檻：動能 ≥ {mom_thresh*100:.1f}%  n={len(high_mom):,}")
    for fd in FORWARD_DAYS:
        col = f"fwd_{fd}d"
        a   = high_mom[high_mom["is_a_top"]][col].dropna()
        na  = high_mom[~high_mom["is_a_top"]][col].dropna()
        print(f"\n  {fd}日後：")
        print(f"    高動能+A頭  中位 {a.median()*100:+.2f}%  勝率 {(a>0).mean()*100:.1f}%  n={len(a):,}")
        print(f"    高動能+無A頭 中位 {na.median()*100:+.2f}%  勝率 {(na>0).mean()*100:.1f}%  n={len(na):,}")
        print(f"    差距  {(a.median()-na.median())*100:+.2f}%")

    # ── 分析 3：A頭深度分析（按回落幅度分組）───────────────
    print("\n" + "="*60)
    print("分析 3：A頭回落深度 vs 後續21日報酬")
    print("="*60)
    a_df = df[df["is_a_top"]].copy()

    # 計算 A頭回落幅度
    def calc_reversal(row, prices):
        sym = row["sym"]
        s = prices[sym].dropna()
        idx = s.index.get_loc(row["date"])
        if idx < WINDOW:
            return np.nan
        recent = s.iloc[idx - WINDOW : idx]
        peak = float(recent.max())
        curr = float(recent.iloc[-1])
        return (curr - peak) / peak

    print("  計算各 A頭回落幅度...")
    a_df["reversal"] = a_df.apply(lambda r: calc_reversal(r, prices), axis=1)

    bins = [-1, -0.20, -0.15, -0.12]
    labels = ["-20%以上回落", "-15~-20%回落", "-12~-15%回落"]
    a_df["reversal_bin"] = pd.cut(a_df["reversal"], bins=bins, labels=labels)

    for grp, gdf in a_df.groupby("reversal_bin", observed=True):
        col = "fwd_21d"
        r = gdf[col].dropna()
        if len(r) < 10:
            continue
        print(f"\n  {grp}（n={len(r):,}）")
        print(f"    21日後  中位 {r.median()*100:+.2f}%  均值 {r.mean()*100:+.2f}%  勝率 {(r>0).mean()*100:.1f}%")

    # ── 分析 4：板塊分層——A頭效果是否跨板塊一致 ──────────────
    print("\n" + "="*60)
    print("分析 4：各板塊高動能股中，A頭 vs 無A頭（21日後）")
    print("（驗證效果是否只集中在特定板塊）")
    print("="*60)
    try:
        from src.data_loader import get_sp500_sector_map
        sector_map = get_sp500_sector_map()
        df["sector"] = df["sym"].map(sector_map).fillna("其他")

        high_mom = df[df["momentum"] >= df["momentum"].quantile(0.75)]
        sector_results = []
        for sector, gdf in high_mom.groupby("sector"):
            a   = gdf[gdf["is_a_top"]]["fwd_21d"].dropna()
            na  = gdf[~gdf["is_a_top"]]["fwd_21d"].dropna()
            if len(a) < 30 or len(na) < 100:
                continue
            gap = (a.median() - na.median()) * 100
            a_wr  = (a > 0).mean() * 100
            na_wr = (na > 0).mean() * 100
            sector_results.append((sector, gap, a.median()*100, na.median()*100, a_wr, na_wr, len(a)))

        sector_results.sort(key=lambda x: x[1])
        print(f"\n  {'板塊':<10} {'A頭中位':>8} {'無A頭中位':>9} {'差距':>7} {'A頭勝率':>8} {'無A頭勝率':>9} {'n(A)':>6}")
        print("  " + "-"*65)
        for s, gap, a_med, na_med, a_wr, na_wr, n in sector_results:
            mark = " ✅" if gap < -0.5 else (" ❌" if gap > 0.5 else "")
            print(f"  {s:<10} {a_med:>7.2f}%  {na_med:>8.2f}%  {gap:>+6.2f}%  {a_wr:>7.1f}%  {na_wr:>8.1f}%  {n:>5}{mark}")

        consistent = sum(1 for _, gap, *_ in sector_results if gap < -0.5)
        total = len(sector_results)
        print(f"\n  A頭拖累（差距 < -0.5%）的板塊：{consistent}/{total}")
        if consistent >= total * 0.6:
            print("  ✅ 效果跨板塊一致，可信度高")
        elif consistent >= total * 0.4:
            print("  🟡 部分板塊有效，建議搭配板塊條件使用")
        else:
            print("  ❌ 效果不跨板塊，可能有選擇偏誤")
    except Exception as e:
        print(f"  板塊分析失敗：{e}")

    # ── 分析 5：A頭 時間動態（低點 + 恢復時間）────────────────
    print("\n" + "="*60)
    print("分析 5：A頭後 時間動態")
    print("  低點幾天後出現？恢復到 A頭峰值要多久？")
    print("="*60)

    MAX_SCAN = 126  # 掃描最多 126 交易日 ≈ 6 個月
    mom_thresh_5 = df["momentum"].quantile(0.75)

    results5 = []
    for sym, sym_df in a_df.groupby("sym"):
        s    = prices[sym].dropna()
        vals = s.values
        slen = len(s)

        for _, row in sym_df.iterrows():
            date = row["date"]
            try:
                t = s.index.get_loc(date)
            except KeyError:
                continue
            if t < WINDOW or t + 2 >= slen:
                continue

            curr  = float(vals[t])
            peak  = float(vals[t - WINDOW : t].max())
            end   = min(t + MAX_SCAN + 1, slen)
            fwd   = vals[t : end]

            trough_idx   = int(np.argmin(fwd))
            trough_price = float(fwd[trough_idx])

            above = np.where(fwd >= peak)[0]
            rec_days = int(above[0]) if len(above) > 0 else np.nan

            results5.append({
                "days_to_trough": trough_idx,
                "add_drop":       (trough_price - curr) / curr,
                "days_to_rec":    rec_days,
                "high_mom":       row["momentum"] >= mom_thresh_5,
            })

    r5 = pd.DataFrame(results5)

    def show_time_stats(sub, label):
        ta  = sub["days_to_trough"].values
        da  = sub["add_drop"].values
        ra  = sub["days_to_rec"].values.astype(float)
        rv  = ra[~np.isnan(ra)]
        rrate = len(rv) / len(ra) * 100 if len(ra) > 0 else 0

        print(f"\n  【{label}，n={len(ta):,}】")
        print(f"  低點出現時間（交易日）：中位 {np.median(ta):.0f}天  平均 {np.mean(ta):.0f}天  "
              f"P25={np.percentile(ta,25):.0f}  P75={np.percentile(ta,75):.0f}")

        buckets = [(0,5),(5,10),(10,21),(21,42),(42,63),(63,126)]
        dist = "  分佈："
        for lo, hi in buckets:
            pct = ((ta >= lo) & (ta < hi)).mean() * 100
            dist += f"  {lo}~{hi}日:{pct:.0f}%"
        print(dist)

        print(f"  信號日後再跌幅：中位 {np.median(da)*100:+.1f}%  "
              f"P25={np.percentile(da,25)*100:+.1f}%  P75={np.percentile(da,75)*100:+.1f}%")

        print(f"  恢復到峰值（{MAX_SCAN}日內）：{rrate:.1f}%", end="")
        if len(rv) > 0:
            print(f"  中位 {np.median(rv):.0f}天  P25={np.percentile(rv,25):.0f}  P75={np.percentile(rv,75):.0f}")
        else:
            print()

    show_time_stats(r5, "全部 A頭")
    show_time_stats(r5[r5["high_mom"]], "高動能 A頭")
    show_time_stats(r5[~r5["high_mom"]], "非高動能 A頭")

    # ── 結論 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("結論摘要")
    print("="*60)
    a_21  = df[df["is_a_top"]]["fwd_21d"].dropna()
    na_21 = df[~df["is_a_top"]]["fwd_21d"].dropna()
    gap   = (a_21.median() - na_21.median()) * 100
    print(f"  A頭21日後中位報酬：{a_21.median()*100:+.2f}%")
    print(f"  非A頭21日後中位報酬：{na_21.median()*100:+.2f}%")
    print(f"  A頭拖累：{gap:+.2f}%")
    if gap < -1.0:
        print("  ✅ A頭型態有統計意義的拖累效果，建議加入過濾條件")
    elif gap < -0.5:
        print("  🟡 A頭型態有輕微拖累，可考慮作為質化參考")
    else:
        print("  ❌ A頭型態效果不顯著，動能策略可忽略此型態")


if __name__ == "__main__":
    main()
