"""
保護期動態調整回測

核心問題：買入後第10天若出現加速度減速信號，
在第15天出場（提前解除保護）vs 等到第30天（現行），差距多大？

測試三種分組門檻：
  保守：acc10_mean < -10%pts 且 21d_return < 0%
  中等：acc10_mean < -5%pts  且 21d_return < -2%
  寬鬆：acc10_mean < -5%pts（不看 21d 方向）

第二層：C 組在 T+15 換入當時最強候選股，再投資報酬 vs 原股續持
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from src.data_loader import get_sp500_tickers

# ── 參數 ──────────────────────────────────────────────────────────────
CACHE_PATH    = "data/_protection_bt_prices.pkl"
DATA_START    = "2019-06-01"
DATA_END      = "2026-03-01"
BATCH_SIZE    = 50
MOM_SHORT     = 21
MOM_LONG      = 252
TOP_N         = 5
ACC_WINDOW    = 10    # 買入後前 10 個交易日的加速度均值
PROTECT_CURR  = 30    # 現行保護期
PROTECT_EARLY = 15    # 提前出場日
REINVEST_FWD  = 45    # 再投資追蹤：T+15 → T+45

THRESHOLDS = {
    "保守 (acc<-10% & ret<0%)": {"acc": -10.0, "ret":  0.0},
    "中等 (acc<-5%  & ret<-2%)": {"acc":  -5.0, "ret": -2.0},
    "寬鬆 (acc<-5%)":            {"acc":  -5.0, "ret": None},
}

# ── 下載資料（分批，有快取）─────────────────────────────────────────
if os.path.exists(CACHE_PATH):
    print(f"讀取快取：{CACHE_PATH}")
    prices = pd.read_pickle(CACHE_PATH)
else:
    tickers = get_sp500_tickers()
    print(f"分批下載 {len(tickers)} 檔（每批 {BATCH_SIZE}）...")
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_dfs = []
    for i, batch in enumerate(batches):
        print(f"  批次 {i+1}/{len(batches)}...", end=" ", flush=True)
        try:
            raw = yf.download(batch, start=DATA_START, end=DATA_END,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print("空")
                continue
            close = raw["Close"]
            if isinstance(close.columns, pd.MultiIndex):
                close.columns = close.columns.droplevel(0)
            all_dfs.append(close)
            print(f"OK ({close.shape[1]} 檔)")
        except Exception as e:
            print(f"ERROR: {e}")

    prices = pd.concat(all_dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all")
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    os.makedirs("data", exist_ok=True)
    prices.to_pickle(CACHE_PATH)
    print(f"已快取至 {CACHE_PATH}")

prices.index = pd.to_datetime(prices.index).tz_localize(None) \
    if prices.index.tz is None else pd.to_datetime(prices.index).tz_convert(None)

# 共用交易日曆
trading_days = prices.index.sort_values()
print(f"資料：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日  "
      f"({trading_days[0].date()} ~ {trading_days[-1].date()})\n")


# ── 工具函式 ──────────────────────────────────────────────────────────
def trading_idx(ref_date: str) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(ref_date)))


def price_at(sym: str, tidx: int) -> float | None:
    if tidx < 0 or tidx >= len(trading_days):
        return None
    if sym not in prices.columns:
        return None
    val = prices.iloc[tidx][sym]
    return float(val) if pd.notna(val) else None


def fwd_return(sym: str, entry_tidx: int, offset: int) -> float | None:
    p0 = price_at(sym, entry_tidx)
    p1 = price_at(sym, entry_tidx + offset)
    if p0 and p1 and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None


def mom_mixed_at(sym: str, tidx: int) -> float | None:
    if sym not in prices.columns:
        return None
    if tidx < MOM_LONG or tidx >= len(trading_days):
        return None
    p0   = price_at(sym, tidx)
    p21  = price_at(sym, tidx - MOM_SHORT)
    p252 = price_at(sym, tidx - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    r21  = (p0 / p21 - 1) * 100
    r252 = (p0 / p252 - 1) * 100
    return 0.5 * r21 + 0.5 * r252


def acc10_mean_at(sym: str, entry_tidx: int, window: int = ACC_WINDOW) -> float | None:
    """entry_tidx 起算往後 window 天，計算每日加速度均值"""
    if sym not in prices.columns:
        return None
    accels = []
    for d in range(window):
        idx = entry_tidx + d
        if idx < MOM_SHORT * 2 or idx >= len(trading_days):
            continue
        p_now  = price_at(sym, idx)
        p_21   = price_at(sym, idx - MOM_SHORT)
        p_42   = price_at(sym, idx - MOM_SHORT * 2)
        if not (p_now and p_21 and p_42 and p_21 > 0 and p_42 > 0):
            continue
        r21_now  = (p_now / p_21  - 1) * 100
        r21_prev = (p_21  / p_42  - 1) * 100
        accels.append(r21_now - r21_prev)
    return float(np.mean(accels)) if len(accels) >= window // 2 else None


def ret21_at(sym: str, tidx: int) -> float | None:
    p0 = price_at(sym, tidx - MOM_SHORT)
    p1 = price_at(sym, tidx)
    if p0 and p1 and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None


# ── 月份回測循環 ──────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")

events = []   # 每筆買入事件

print("回測中...")
for ref in rebalance_dates:
    tidx = trading_idx(str(ref.date()))
    if tidx < MOM_LONG:
        continue

    # 計算全市場動能，選前 TOP_N
    moms = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, tidx)
        if m is not None and m > 0:
            moms.append((sym, m))

    if len(moms) < TOP_N:
        continue

    top5 = sorted(moms, key=lambda x: -x[1])[:TOP_N]

    # 記錄每個 pick 的事件
    for sym, entry_mom in top5:
        # 前向報酬
        r = {
            "date":      str(ref.date()),
            "symbol":    sym,
            "entry_mom": entry_mom,
        }
        for d in [15, 30, 45, 60]:
            r[f"r{d}"] = fwd_return(sym, tidx, d)

        # T+10 狀態
        r["acc10"] = acc10_mean_at(sym, tidx, ACC_WINDOW)
        r["ret21_t10"] = ret21_at(sym, tidx + ACC_WINDOW)

        # T+15 時的最佳替代股（用於再投資分析）
        tidx_15 = tidx + PROTECT_EARLY
        if tidx_15 < len(trading_days):
            alt_moms = []
            for s in prices.columns:
                if s == sym:
                    continue
                m2 = mom_mixed_at(s, tidx_15)
                if m2 is not None and m2 > 0:
                    alt_moms.append((s, m2))
            if alt_moms:
                best_alt = max(alt_moms, key=lambda x: x[1])[0]
                r["best_alt"] = best_alt
                r["alt_r15_45"] = fwd_return(best_alt, tidx_15,
                                             REINVEST_FWD - PROTECT_EARLY)
                r["orig_r15_45"] = fwd_return(sym, tidx_15,
                                              REINVEST_FWD - PROTECT_EARLY)
            else:
                r["best_alt"] = None
                r["alt_r15_45"] = None
                r["orig_r15_45"] = None
        else:
            r["best_alt"] = r["alt_r15_45"] = r["orig_r15_45"] = None

        events.append(r)

df_all = pd.DataFrame(events)
df_all = df_all.dropna(subset=["acc10", "ret21_t10", "r15", "r30"])
print(f"有效買入事件：{len(df_all)} 筆\n")


# ── 分析輸出 ──────────────────────────────────────────────────────────
def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A "
    return f"{v:+.2f}%"


print("=" * 72)
print("  保護期動態調整回測  2021-2025  S&P500 月度動能前5名")
print("=" * 72)

summary_rows = []

for tname, tparam in THRESHOLDS.items():
    acc_thr = tparam["acc"]
    ret_thr = tparam["ret"]

    # 分組：C = 觸發提前解除；A = 正常
    if ret_thr is not None:
        mask_c = (df_all["acc10"] < acc_thr) & (df_all["ret21_t10"] < ret_thr)
    else:
        mask_c = (df_all["acc10"] < acc_thr)

    df_c = df_all[mask_c]
    df_a = df_all[~mask_c]

    print(f"\n{'─'*72}")
    print(f"  門檻：{tname}")
    print(f"  A組（正常持有）：{len(df_a):3d} 筆  |  C組（觸發提前出場）：{len(df_c):3d} 筆")
    print(f"  {'分組':<20} {'T+15':>8} {'T+30':>8} {'T+45':>8} {'T+60':>8}  n")
    print(f"  {'-'*60}")

    for label, sub in [("A 正常", df_a), ("C 減速觸發", df_c)]:
        n = len(sub)
        if n == 0:
            print(f"  {label:<20} {'─':>8} {'─':>8} {'─':>8} {'─':>8}  {n}")
            continue
        r15 = sub["r15"].mean()
        r30 = sub["r30"].mean()
        r45 = sub["r45"].dropna().mean()
        r60 = sub["r60"].dropna().mean()
        print(f"  {label:<20} {fmt(r15):>8} {fmt(r30):>8} {fmt(r45):>8} {fmt(r60):>8}  {n}")

    # 關鍵數字：T+15 出場 vs T+30 才能動
    if len(df_c) > 0:
        saved = df_c["r15"].mean() - df_c["r30"].mean()
        print(f"\n  ★ C組：T+15 出場 vs T+30 出場，每筆節省損失 {saved:+.2f}%pts")

    # 再投資分析
    df_c_ri = df_c.dropna(subset=["alt_r15_45", "orig_r15_45"])
    if len(df_c_ri) > 0:
        orig_mean = df_c_ri["orig_r15_45"].mean()
        alt_mean  = df_c_ri["alt_r15_45"].mean()
        gain      = alt_mean - orig_mean
        print(f"  ★ C組再投資 T+15→T+45：原股 {fmt(orig_mean)}  →  換股 {fmt(alt_mean)}"
              f"  （差距 {gain:+.2f}%pts，n={len(df_c_ri)}）")

    summary_rows.append({
        "門檻": tname,
        "n_A": len(df_a), "n_C": len(df_c),
        "A_r30": df_a["r30"].mean() if len(df_a) else None,
        "C_r15": df_c["r15"].mean() if len(df_c) else None,
        "C_r30": df_c["r30"].mean() if len(df_c) else None,
        "saved": (df_c["r15"].mean() - df_c["r30"].mean()) if len(df_c) else None,
    })


# ── 彙總表 ────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  彙總：C組提前出場可節省損失（T+15 vs T+30）")
print(f"  {'門檻':<32} {'C組 T+15':>10} {'C組 T+30':>10} {'節省':>8}  n_C")
print(f"  {'-'*64}")
for r in summary_rows:
    print(f"  {r['門檻']:<32} {fmt(r['C_r15']):>10} {fmt(r['C_r30']):>10} "
          f"{fmt(r['saved']):>8}  {r['n_C']}")

# ── 加速度分佈直方圖（文字版）────────────────────────────────────────
print(f"\n{'='*72}")
print("  acc10 分佈（全部事件，bins of 5%pts）")
bins = list(range(-60, 31, 5))
hist, edges = np.histogram(df_all["acc10"].dropna(), bins=bins)
for i, cnt in enumerate(hist):
    bar = "█" * (cnt // 5)
    lo, hi = edges[i], edges[i+1]
    print(f"  [{lo:+4.0f}% ~ {hi:+4.0f}%)  {bar} {cnt}")
