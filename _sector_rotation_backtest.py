"""
_sector_rotation_backtest.py  (向量化重寫，預期 < 2 分鐘)

三個問題：
A. 板塊 ETF 出現相對強度訊號後，板塊內個股的 forward alpha 如何？
B. 月度選股：A 純個股動能 vs B 最強板塊內的 top-5 vs C 混合
C. Lag 分析：板塊 ETF 轉強後，個股幾天才進 top-N？

執行：
    conda run -n qt_env python _sector_rotation_backtest.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("載入資料...")
prices     = pd.read_pickle("data/_protection_bt_prices.pkl")
etf_prices = pd.read_pickle("data/_ml_sector_etf_prices.pkl")
sector_map = pd.read_pickle("data/_ml_sector_map.pkl")

for df in [prices, etf_prices]:
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

common = prices.index.intersection(etf_prices.index).sort_values()
prices     = prices.loc[common].astype(float)
etf_prices = etf_prices.loc[common].astype(float)

n_dates = len(common)
symbols = list(prices.columns)
print(f"  個股 {len(symbols)} 支 / 共同日期 {n_dates} 天 ({common[0].date()}~{common[-1].date()})")

ETF_TO_SECTOR = {
    "XLK":"Technology", "XLC":"Communication Services", "XLY":"Consumer Cyclical",
    "XLF":"Financial Services", "XLV":"Healthcare", "XLI":"Industrials",
    "XLE":"Energy", "XLB":"Basic Materials", "XLP":"Consumer Defensive",
    "XLRE":"Real Estate", "XLU":"Utilities",
}
SECTOR_ETFS = [e for e in ETF_TO_SECTOR if e in etf_prices.columns]

# 板塊→個股清單（含別名）
_alias = {"Basic Materials": ["Basic Materials","Materials"],
          "Financial Services": ["Financial Services","Financial"],
          "Communication Services": ["Communication Services","Communication"]}
etf_to_syms = {}
for etf, sec in ETF_TO_SECTOR.items():
    valid = [sec] + _alias.get(sec, [])
    syms_in = [s for s in symbols if sector_map.get(s) in valid]
    etf_to_syms[etf] = syms_in

print(f"  板塊對應完成：{sum(len(v) for v in etf_to_syms.values())} 支有板塊資料\n")

# ── 向量化預計算 ───────────────────────────────────────────────────────────────
print("預計算動能矩陣...")
px = prices.values                                # shape (n_dates, n_sym)

# mom_mixed = 0.5*21日動能 + 0.5*252日動能
with np.errstate(divide='ignore', invalid='ignore'):
    mom21  = np.where(px[:-21]  > 0, px[21:]  / px[:-21]  - 1, np.nan)
    mom252 = np.where(px[:-252] > 0, px[252:] / px[:-252] - 1, np.nan)

# 補齊長度（mom21 短 21 列，mom252 短 252 列）
pad21  = np.full((21,  len(symbols)), np.nan)
pad252 = np.full((252, len(symbols)), np.nan)
mom21_full  = np.vstack([pad21,  mom21])    # (n_dates, n_sym)
mom252_full = np.vstack([pad252, mom252])   # (n_dates, n_sym)
mom_mat = 0.5 * mom21_full + 0.5 * mom252_full   # NaN if either missing

# 板塊 ETF 相對動能（10日 vs SPY）
spy_px  = etf_prices["SPY"].values
etf_mat = {}   # etf → (n_dates,) relative 10d momentum
for etf in SECTOR_ETFS:
    ep = etf_prices[etf].values
    with np.errstate(divide='ignore', invalid='ignore'):
        ret_etf = np.where(ep[:-10] > 0, ep[10:]/ep[:-10] - 1, np.nan)
        ret_spy = np.where(spy_px[:-10] > 0, spy_px[10:]/spy_px[:-10] - 1, np.nan)
    rel = ret_etf - ret_spy
    etf_mat[etf] = np.concatenate([np.full(10, np.nan), rel])   # (n_dates,)

# Forward return vs SPY（21日）
with np.errstate(divide='ignore', invalid='ignore'):
    fwd21_stock = np.where(px[:-21] > 0, px[21:]/px[:-21] - 1, np.nan)
    spy_fwd21   = np.where(spy_px[:-21] > 0, spy_px[21:]/spy_px[:-21] - 1, np.nan)
fwd21_alpha_full = np.full_like(mom_mat, np.nan)
for i in range(n_dates - 21):
    fwd21_alpha_full[i] = fwd21_stock[i] - spy_fwd21[i]

# 42日 alpha
with np.errstate(divide='ignore', invalid='ignore'):
    fwd42_stock = np.where(px[:-42] > 0, px[42:]/px[:-42] - 1, np.nan)
    spy_fwd42   = np.where(spy_px[:-42] > 0, spy_px[42:]/spy_px[:-42] - 1, np.nan)
fwd42_alpha_full = np.full_like(mom_mat, np.nan)
for i in range(n_dates - 42):
    fwd42_alpha_full[i] = fwd42_stock[i] - spy_fwd42[i]

sym_idx = {s: i for i, s in enumerate(symbols)}
MOM_LONG   = 252
START_TI   = MOM_LONG + 10
REBAL_FREQ = 21
ETF_THRESH = 0.02
TOP_N      = 5
rebal_tis  = list(range(START_TI, n_dates - 65, REBAL_FREQ))
print(f"  預計算完成，再平衡點 {len(rebal_tis)} 個\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 問題 A：板塊訊號預測力
# ═══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  【問題 A】板塊 ETF 相對強度 → 個股 forward alpha")
print("="*70)

a_strong_21, a_strong_42 = [], []
a_weak_21,   a_weak_42   = [], []
by_etf_21 = {e: [] for e in SECTOR_ETFS}

for ti in rebal_tis:
    for etf in SECTOR_ETFS:
        rel = etf_mat[etf][ti]
        if np.isnan(rel):
            continue
        strong = rel > ETF_THRESH
        sym_list = etf_to_syms.get(etf, [])
        for sym in sym_list:
            si = sym_idx.get(sym)
            if si is None:
                continue
            m = mom_mat[ti, si]
            if np.isnan(m) or m <= 0:
                continue
            a21 = fwd21_alpha_full[ti, si]
            a42 = fwd42_alpha_full[ti, si]
            if not np.isnan(a21):
                (a_strong_21 if strong else a_weak_21).append(a21)
                if strong:
                    by_etf_21[etf].append(a21)
            if not np.isnan(a42):
                (a_strong_42 if strong else a_weak_42).append(a42)

def stats(arr):
    a = np.array(arr)
    if len(a) == 0: return "N/A"
    return (f"平均 {np.mean(a)*100:>+5.2f}%  中位 {np.median(a)*100:>+5.2f}%  "
            f"勝率 {np.mean(a>0)*100:.0f}%  N={len(a)}")

print(f"\n  板塊 ETF 訊號強（>+2%）後：")
print(f"    21日 alpha：{stats(a_strong_21)}")
print(f"    42日 alpha：{stats(a_strong_42)}")
print(f"\n  板塊 ETF 訊號弱（≤+2%）後：")
print(f"    21日 alpha：{stats(a_weak_21)}")
print(f"    42日 alpha：{stats(a_weak_42)}")

print(f"\n  各板塊 ETF 訊號強後 21 日 alpha：")
print(f"  {'ETF':<6}  {'板塊':<24}  {'21日平均':>8}  {'21日中位':>8}  {'勝率':>6}  {'N':>5}")
print(f"  {'─'*62}")
for etf in SECTOR_ETFS:
    arr = np.array(by_etf_21[etf])
    if len(arr) < 10: continue
    sec = ETF_TO_SECTOR[etf]
    print(f"  {etf:<6}  {sec:<24}  {np.mean(arr)*100:>+7.2f}%  "
          f"{np.median(arr)*100:>+7.2f}%  {np.mean(arr>0)*100:>5.0f}%  {len(arr):>5}")

# ═══════════════════════════════════════════════════════════════════════════════
# 問題 B：策略比較
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  【問題 B】月度選股策略比較（持有 21 天）")
print(f"{'='*70}")

res_a, res_b, res_c = [], [], []

for ti in rebal_tis:
    row_mom = mom_mat[ti]          # (n_sym,)
    row_a21 = fwd21_alpha_full[ti]

    # 全市場正動能排名
    valid_mask = (~np.isnan(row_mom)) & (row_mom > 0)
    if valid_mask.sum() < TOP_N:
        continue
    order = np.argsort(-np.where(valid_mask, row_mom, -np.inf))

    # 策略 A
    picks_a_idx = order[:TOP_N]
    alpha_a = row_a21[picks_a_idx]
    alpha_a = alpha_a[~np.isnan(alpha_a)]
    if len(alpha_a) >= 3:
        res_a.append(float(np.mean(alpha_a)))

    # 板塊 ETF 相對動能排名
    etf_rels = [(etf, etf_mat[etf][ti]) for etf in SECTOR_ETFS
                if not np.isnan(etf_mat[etf][ti])]
    if not etf_rels:
        continue
    etf_rels.sort(key=lambda x: -x[1])

    # 策略 B：最強板塊內的 top-5
    top_etf   = etf_rels[0][0]
    b_pool    = [sym_idx[s] for s in etf_to_syms.get(top_etf, []) if s in sym_idx]
    if len(b_pool) < 3:
        res_b.append(res_a[-1] if res_a else 0)
        res_c.append(res_a[-1] if res_a else 0)
        continue
    b_pool_mask = np.zeros(len(symbols), dtype=bool)
    b_pool_mask[b_pool] = True
    combined_b = valid_mask & b_pool_mask
    if combined_b.sum() < 3:
        res_b.append(res_a[-1] if res_a else 0)
    else:
        order_b    = np.argsort(-np.where(combined_b, row_mom, -np.inf))
        picks_b    = order_b[:TOP_N]
        alpha_b    = row_a21[picks_b]
        alpha_b    = alpha_b[~np.isnan(alpha_b)]
        res_b.append(float(np.mean(alpha_b)) if len(alpha_b) >= 2 else res_a[-1])

    # 策略 C：前 2 強板塊各出 top-3
    picks_c_idx = []
    for etf, _ in etf_rels[:2]:
        c_pool = [sym_idx[s] for s in etf_to_syms.get(etf, []) if s in sym_idx]
        c_mask = np.zeros(len(symbols), dtype=bool)
        c_mask[c_pool] = True
        combined_c = valid_mask & c_mask
        if combined_c.sum() < 1:
            continue
        order_c = np.argsort(-np.where(combined_c, row_mom, -np.inf))
        picks_c_idx += list(order_c[:3])
    picks_c_idx = list(dict.fromkeys(picks_c_idx))
    if len(picks_c_idx) < 3:
        res_c.append(res_a[-1] if res_a else 0)
    else:
        alpha_c = row_a21[picks_c_idx]
        alpha_c = alpha_c[~np.isnan(alpha_c)]
        res_c.append(float(np.mean(alpha_c)) if len(alpha_c) >= 2 else res_a[-1])

n = min(len(res_a), len(res_b), len(res_c))
a_arr = np.array(res_a[:n]); b_arr = np.array(res_b[:n]); c_arr = np.array(res_c[:n])

print(f"\n  {'策略':<28}  {'月均alpha':>8}  {'月中位':>8}  {'α>0%':>7}  {'勝策略A%':>9}")
print(f"  {'─'*64}")
for name, arr, base in [
    ("A 純個股 top-5（現行）", a_arr, None),
    ("B 最強板塊 top-5",       b_arr, a_arr),
    ("C 混合（2板塊×top3）",   c_arr, a_arr),
]:
    vs = f"{np.mean(arr>base)*100:>7.1f}%" if base is not None else f"{'—':>8}"
    print(f"  {name:<28}  {np.mean(arr)*100:>+7.2f}%  {np.median(arr)*100:>+7.2f}%  "
          f"{np.mean(arr>0)*100:>6.0f}%  {vs}")

# ═══════════════════════════════════════════════════════════════════════════════
# 問題 C：Lag 分析（向量化）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  【問題 C】板塊 ETF 轉強後，個股進 top-50 的 lag（天）")
print(f"{'='*70}\n")

# 預計算每天各股全市場排名（只在 rebal_tis 上）
# 但 lag 需要每天的排名，所以取每天的快照
# 為節省時間，每 5 天取一次
step = 5
sample_tis = list(range(START_TI, n_dates - 65, step))

# 建立 lag 資料：遍歷 ETF 訊號轉折點
lags_by_etf = {etf: [] for etf in SECTOR_ETFS}

for etf in SECTOR_ETFS:
    rel_series = etf_mat[etf]
    sym_list   = etf_to_syms.get(etf, [])
    si_list    = [sym_idx[s] for s in sym_list if s in sym_idx]
    if not si_list:
        continue

    prev_strong = False
    for ti in sample_tis:
        rel = rel_series[ti]
        if np.isnan(rel):
            prev_strong = False
            continue
        now_strong = rel > ETF_THRESH

        if now_strong and not prev_strong:
            # 信號剛出現 → 計算板塊內個股的全市場動能排名
            row = mom_mat[ti]
            valid = ~np.isnan(row)
            if valid.sum() < 50:
                prev_strong = now_strong
                continue
            # 計算全市場排名
            rank_arr = np.full(len(symbols), 9999)
            sorted_idx = np.argsort(-np.where(valid, row, -np.inf))
            for rank, si in enumerate(sorted_idx):
                if valid[si]:
                    rank_arr[si] = rank + 1
                else:
                    break

            # 對板塊內個股，記錄當下排名
            for si in si_list:
                m = mom_mat[ti, si]
                if np.isnan(m) or m <= 0:
                    continue
                rank_now = rank_arr[si]
                if rank_now <= 50:
                    lags_by_etf[etf].append(0)
                    continue
                # 往後最多 42 天找進 top-50
                found = False
                for lag in range(step, 42, step):
                    t2 = ti + lag
                    if t2 >= n_dates:
                        break
                    row2 = mom_mat[t2]
                    v2   = ~np.isnan(row2)
                    if v2.sum() < 50:
                        continue
                    sorted2 = np.argsort(-np.where(v2, row2, -np.inf))
                    rank2 = 9999
                    for r, s2 in enumerate(sorted2[:60]):
                        if s2 == si:
                            rank2 = r + 1
                            break
                    if rank2 <= 50:
                        lags_by_etf[etf].append(lag)
                        found = True
                        break
                if not found:
                    lags_by_etf[etf].append(42)

        prev_strong = now_strong

all_lags = [l for lgs in lags_by_etf.values() for l in lgs]
if all_lags:
    lags = np.array(all_lags)
    print(f"  整體 lag 分佈：")
    for p in [10, 25, 50, 75, 90]:
        print(f"    P{p:<3}: {np.percentile(lags, p):>5.1f} 天")
    print(f"    平均:  {np.mean(lags):>5.1f} 天  (≥42天: {np.mean(lags>=42)*100:.0f}%)\n")

    print(f"  {'ETF':<6}  {'板塊':<24}  {'中位lag':>7}  {'平均lag':>7}  {'≤14天%':>7}  N")
    print(f"  {'─'*60}")
    for etf in SECTOR_ETFS:
        lgs = lags_by_etf.get(etf, [])
        if len(lgs) < 5: continue
        arr = np.array(lgs)
        sec = ETF_TO_SECTOR[etf]
        print(f"  {etf:<6}  {sec:<24}  {np.median(arr):>6.0f}天  "
              f"{np.mean(arr):>6.0f}天  {np.mean(arr<=14)*100:>6.0f}%  {len(arr)}")

# ── 結論 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  綜合結論")
print(f"{'='*70}")

diff_ab = np.mean(b_arr)*100 - np.mean(a_arr)*100
diff_ac = np.mean(c_arr)*100 - np.mean(a_arr)*100
print(f"  A 純個股動能：月 alpha {np.mean(a_arr)*100:>+.2f}%")
print(f"  B 板塊優先：  月 alpha {np.mean(b_arr)*100:>+.2f}%  ({diff_ab:>+.2f}% vs A)")
print(f"  C 混合：      月 alpha {np.mean(c_arr)*100:>+.2f}%  ({diff_ac:>+.2f}% vs A)")

if all_lags:
    med = np.median(lags)
    print(f"\n  板塊 ETF 轉強後，個股進 top-50 中位 lag = {med:.0f} 天")
    if med >= 14:
        print(f"  → lag 達 {med:.0f} 天，提早依板塊訊號進場有實際優勢")
    else:
        print(f"  → lag 僅 {med:.0f} 天，個股動能能快速反映，提早進場優勢有限")

print(f"\n  ⚠️  Survivorship bias：僅含 S&P500 現有成份股")
