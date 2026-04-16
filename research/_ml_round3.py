"""
ML 選股 Round 3：個股超額動能 + 加速度 + 一致性 + 恐慌指數

前一輪最佳：ALL_NEW（30 個特徵）AUROC = 0.714

本輪新增四組：
  Group A（個股 vs 板塊）：stock_vs_sector_21d/63d
                           = 個股動能 - 所屬板塊 ETF 動能（板塊內 alpha）
  Group B（動能加速度）  ：accel_21_63 = mom_21d - mom_63d
                           accel_63_126 = mom_63d - mom_126d
  Group C（動能一致性）  ：up_months_12（過去 12 個月中正報酬月數）
  Group D（市場恐慌）    ：vix_level, vix_ma_ratio = vix / vix_63d_avg

驗證：conda run -n qt_env python _ml_round3.py
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    HAS_XGB = False

CACHE_PATH   = "data/_protection_bt_prices.pkl"
FEATURE_CACHE = "data/_ml_features.pkl"
SECTOR_CACHE  = "data/_ml_sector_map.pkl"
ETF_CACHE     = "data/_ml_sector_etf_prices.pkl"

RANDOM_STATE = 42
TRAIN_FRAC   = 0.80

SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Information Technology": "XLK",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))


# ── 載入快取 ───────────────────────────────────────────────────────────
for p in [CACHE_PATH, FEATURE_CACHE, SECTOR_CACHE, ETF_CACHE]:
    if not os.path.exists(p):
        print(f"找不到：{p}，請先執行前置腳本")
        raise SystemExit(1)

print("讀取快取...")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()

df_base = pd.read_pickle(FEATURE_CACHE)

sector_map = pd.read_pickle(SECTOR_CACHE)
if isinstance(sector_map, pd.Series):
    sector_map = sector_map.to_dict()

etf_prices = pd.read_pickle(ETF_CACHE)
etf_prices.index = (pd.to_datetime(etf_prices.index).tz_localize(None)
                    if etf_prices.index.tz is None
                    else etf_prices.index.tz_convert(None))
etf_prices = etf_prices.reindex(trading_days)

print(f"基礎特徵：{len(df_base):,} 個樣本\n")


def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

def sym_to_etf(sym):
    s = sector_map.get(sym)
    return SECTOR_ETF_MAP.get(s) if s else None


# ═══════════════════════════════════════════════════════════════════════
# Group A：個股 vs 板塊超額動能
# ═══════════════════════════════════════════════════════════════════════
print("計算 Group A：個股 vs 板塊超額動能...")

# 板塊 ETF 絕對動能（非相對 SPY）
etf_abs_21 = {}
etf_abs_63 = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    ep = etf_prices[etf]
    etf_abs_21[etf] = (ep / ep.shift(21) - 1) * 100
    etf_abs_63[etf] = (ep / ep.shift(63) - 1) * 100

df = df_base.copy()
df["_etf"] = df["symbol"].map(sym_to_etf)

date_ti_map = {str(ref.date()): tidx(str(ref.date()))
               for ref in pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")}

for feat_name, etf_dict, stock_col in [
    ("stock_vs_sector_21d", etf_abs_21, "mom_21d"),
    ("stock_vs_sector_63d", etf_abs_63, "mom_63d"),
]:
    col_vals = np.full(len(df), np.nan)
    for etf in SECTOR_ETFS:
        mask = df["_etf"] == etf
        if not mask.any() or etf not in etf_dict:
            continue
        series = etf_dict[etf]
        for pos, (idx, row) in enumerate(df[mask].iterrows()):
            ti = date_ti_map.get(row["date"])
            if ti is None:
                continue
            etf_ret = series.iloc[ti] if ti < len(series) else np.nan
            stock_ret = row[stock_col]
            if pd.notna(etf_ret) and pd.notna(stock_ret):
                col_vals[df.index.get_loc(idx)] = stock_ret - etf_ret
    df[feat_name] = col_vals

df = df.drop(columns=["_etf"])
print(f"  有效：stock_vs_sector_21d {df['stock_vs_sector_21d'].notna().sum():,}")


# ═══════════════════════════════════════════════════════════════════════
# Group B：動能加速度（直接用 df_base 已有的動能欄位計算）
# ═══════════════════════════════════════════════════════════════════════
print("計算 Group B：動能加速度...")

df["accel_21_63"]  = df["mom_21d"]  - df["mom_63d"]
df["accel_63_126"] = df["mom_63d"]  - df["mom_126d"]
df["accel_126_252"]= df["mom_126d"] - df["mom_252d"]

print(f"  有效：accel_21_63 {df['accel_21_63'].notna().sum():,}")


# ═══════════════════════════════════════════════════════════════════════
# Group C：動能一致性（過去 12 個月中有幾個月正報酬）
# ═══════════════════════════════════════════════════════════════════════
print("計算 Group C：動能一致性（月度正報酬計數）...")

rebalance_dates = pd.bdate_range("2020-01-01", "2025-09-01", freq="BMS")

# 預計算每支股票在每個月度再平衡點的月報酬（向量化）
monthly_ret_rows = {}
prev_ti = None
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if prev_ti is None:
        prev_ti = ti
        continue
    cur  = prices.iloc[ti]
    prev = prices.iloc[prev_ti]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(prev > 0, cur / prev - 1, np.nan)
    monthly_ret_rows[str(ref.date())] = pd.Series(ret, index=prices.columns)
    prev_ti = ti

monthly_df = pd.DataFrame(monthly_ret_rows).T   # index=date_str, columns=symbols

# 對每個日期，往前滾動計算連續 12 個月的正報酬月數
dates_sorted = sorted(monthly_df.index)
up12_dict = {}   # date_str → Series(sym → count)
for i, d in enumerate(dates_sorted):
    start = max(0, i - 12)
    window = monthly_df.iloc[start:i]    # 往前 12 期（不含當期，避免前瞻）
    if len(window) < 3:
        continue
    up12_dict[d] = (window > 0).sum(axis=0)   # 每支股票的正報酬月數

# 將 up_months_12 填回 df
up_months_vals = np.full(len(df), np.nan)
for i, (idx, row) in enumerate(df.iterrows()):
    d = row["date"]
    if d in up12_dict:
        v = up12_dict[d].get(row["symbol"])
        if v is not None and pd.notna(v):
            up_months_vals[i] = float(v)
df["up_months_12"] = up_months_vals

print(f"  有效：up_months_12 {df['up_months_12'].notna().sum():,}")


# ═══════════════════════════════════════════════════════════════════════
# Group D：VIX 市場恐慌
# ═══════════════════════════════════════════════════════════════════════
print("計算 Group D：VIX 恐慌指數...")

vix_cache = "data/_ml_vix.pkl"
if os.path.exists(vix_cache):
    vix = pd.read_pickle(vix_cache)
else:
    vix = yf.Ticker("^VIX").history(start="2020-01-01", end="2026-03-01",
                                     auto_adjust=True)["Close"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.to_pickle(vix_cache)

vix = vix.reindex(trading_days, method="ffill")
vix_ma63 = vix.rolling(63, min_periods=30).mean()

vix_level_vals  = np.full(len(df), np.nan)
vix_ratio_vals  = np.full(len(df), np.nan)
for i, (idx, row) in enumerate(df.iterrows()):
    ti = date_ti_map.get(row["date"])
    if ti is None or ti >= len(vix):
        continue
    vl = vix.iloc[ti]
    vm = vix_ma63.iloc[ti]
    if pd.notna(vl):
        vix_level_vals[i]  = float(vl)
    if pd.notna(vl) and pd.notna(vm) and vm > 0:
        vix_ratio_vals[i]  = float(vl) / float(vm)

df["vix_level"]    = vix_level_vals
df["vix_ma_ratio"] = vix_ratio_vals

print(f"  有效：vix_level {df['vix_level'].notna().sum():,}\n")


# ═══════════════════════════════════════════════════════════════════════
# 附加：Sharpe 動能
# ═══════════════════════════════════════════════════════════════════════
df["sharpe_mom"] = df["mom_252d"] / df["vol_63d"].replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════
# 季節性 + 板塊特徵（從 _ml_sector_temporal.py 重算，確保資料齊全）
# ═══════════════════════════════════════════════════════════════════════
spy_etf = etf_prices["SPY"]

months = pd.to_datetime(df["date"]).dt.month
df["month_sin"] = np.sin(2 * np.pi * months / 12)
df["month_cos"] = np.cos(2 * np.pi * months / 12)

df["_etf"] = df["symbol"].map(sym_to_etf)

for etf in SECTOR_ETFS:
    df[f"sec_{etf}"] = (df["_etf"] == etf).astype(float)

# sector_rel_21/63/126d
for win, col in [(21, "sector_rel_21d"), (63, "sector_rel_63d"), (126, "sector_rel_126d")]:
    rel_dict = {}
    for etf in SECTOR_ETFS:
        if etf not in etf_prices.columns:
            continue
        rel = etf_prices[etf] / spy_etf
        rel_dict[etf] = (rel / rel.shift(win) - 1) * 100

    col_vals = np.full(len(df), np.nan)
    for etf in SECTOR_ETFS:
        mask = df["_etf"] == etf
        if not mask.any() or etf not in rel_dict:
            continue
        series = rel_dict[etf]
        for idx in df.index[mask]:
            ti = date_ti_map.get(df.at[idx, "date"])
            if ti is None or ti >= len(series):
                continue
            v = series.iloc[ti]
            if pd.notna(v):
                col_vals[df.index.get_loc(idx)] = float(v)
    df[col] = col_vals

# sector_streak
etf_monthly_rel = {}
rebal_streak = pd.bdate_range("2020-12-01", "2025-09-01", freq="BMS")
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    ep = etf_prices[etf]
    monthly = {}
    prev_ref = None
    for ref in rebal_streak:
        ti = tidx(str(ref.date()))
        if prev_ref is None:
            prev_ref = ref
            continue
        ti_p = tidx(str(prev_ref.date()))
        sp_n, sp_p = spy_etf.iloc[ti], spy_etf.iloc[ti_p]
        ep_n, ep_p = ep.iloc[ti], ep.iloc[ti_p]
        if all(pd.notna(x) and x > 0 for x in [ep_n, ep_p, sp_n, sp_p]):
            monthly[str(ref.date())] = (ep_n/ep_p - 1) - (sp_n/sp_p - 1)
        prev_ref = ref
    etf_monthly_rel[etf] = pd.Series(monthly)

etf_streak_map = {}
for etf, series in etf_monthly_rel.items():
    dates = sorted(series.index)
    streak = {}
    for i, d in enumerate(dates):
        cnt = 0
        for j in range(i - 1, max(i - 13, -1), -1):
            if series[dates[j]] > 0:
                cnt += 1
            else:
                break
        streak[d] = cnt
    etf_streak_map[etf] = pd.Series(streak)

col_streak = np.zeros(len(df), dtype=float)
for etf, series in etf_streak_map.items():
    mask = df["_etf"] == etf
    for idx in df.index[mask]:
        v = series.get(df.at[idx, "date"], np.nan)
        col_streak[df.index.get_loc(idx)] = v
df["sector_streak"] = col_streak

df = df.drop(columns=["_etf"])
print(f"全部特徵計算完成，df 共 {len(df.columns)} 欄\n")


# ═══════════════════════════════════════════════════════════════════════
# Train/Test 切分（同 seed）
# ═══════════════════════════════════════════════════════════════════════
all_syms = df["symbol"].unique()
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_syms)
n_train    = int(len(all_syms) * TRAIN_FRAC)
train_syms = set(all_syms[:n_train])
test_syms  = set(all_syms[n_train:])

df_tr = df[df["symbol"].isin(train_syms)]
df_te = df[df["symbol"].isin(test_syms)]
y_tr  = df_tr["y"].values
y_te  = df_te["y"].values


# ═══════════════════════════════════════════════════════════════════════
# 特徵組合
# ═══════════════════════════════════════════════════════════════════════
BASE = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct", "breadth",
]
PREV_BEST = BASE + [
    "sector_rel_21d", "sector_rel_63d", "sector_rel_126d", "sector_streak",
    "month_sin", "month_cos",
] + [f"sec_{e}" for e in SECTOR_ETFS]

GRP_A  = PREV_BEST + ["stock_vs_sector_21d", "stock_vs_sector_63d"]
GRP_B  = PREV_BEST + ["accel_21_63", "accel_63_126", "accel_126_252"]
GRP_C  = PREV_BEST + ["up_months_12"]
GRP_D  = PREV_BEST + ["vix_level", "vix_ma_ratio"]
GRP_SH = PREV_BEST + ["sharpe_mom"]
ALL_R3 = PREV_BEST + [
    "stock_vs_sector_21d", "stock_vs_sector_63d",
    "accel_21_63", "accel_63_126", "accel_126_252",
    "up_months_12",
    "vix_level", "vix_ma_ratio",
    "sharpe_mom",
]


# ═══════════════════════════════════════════════════════════════════════
# 訓練與評估
# ═══════════════════════════════════════════════════════════════════════
def run_xgb(feat_cols, label):
    X_tr = df_tr[feat_cols].copy()
    X_te = df_te[feat_cols].copy()
    med  = X_tr.median()
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)

    model.fit(X_tr, y_tr)
    prob  = model.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, prob)
    return auroc, model


print("=== AUROC 比較（XGBoost，Test set）===\n")

auroc_prev, _    = run_xgb(PREV_BEST, "prev_best")
auroc_a,    _    = run_xgb(GRP_A,     "+個股vs板塊")
auroc_b,    _    = run_xgb(GRP_B,     "+加速度")
auroc_c,    _    = run_xgb(GRP_C,     "+一致性")
auroc_d,    _    = run_xgb(GRP_D,     "+VIX")
auroc_sh,   _    = run_xgb(GRP_SH,    "+Sharpe動能")
auroc_all,  mdl  = run_xgb(ALL_R3,    "全部R3")

rows = [
    ("上一輪最佳（30 特徵）",              auroc_prev, len(PREV_BEST), 0),
    ("+ 個股 vs 板塊 (A)",               auroc_a,    len(GRP_A),     auroc_a   - auroc_prev),
    ("+ 動能加速度 (B)",                  auroc_b,    len(GRP_B),     auroc_b   - auroc_prev),
    ("+ 一致性 up_months_12 (C)",        auroc_c,    len(GRP_C),     auroc_c   - auroc_prev),
    ("+ VIX 恐慌 (D)",                   auroc_d,    len(GRP_D),     auroc_d   - auroc_prev),
    ("+ Sharpe 動能",                    auroc_sh,   len(GRP_SH),    auroc_sh  - auroc_prev),
    ("全部 Round 3",                     auroc_all,  len(ALL_R3),    auroc_all - auroc_prev),
]

print(f"  {'組合':<30s}  {'AUROC':>7s}  {'特徵數':>5s}  {'增量':>8s}")
print("  " + "-" * 60)
for name, auroc, n, delta in rows:
    mark = "✅" if delta > 0.005 else ("➖" if abs(delta) <= 0.002 else "⚠️ ")
    delta_str = f"{delta:+.4f}" if delta != 0 else "  基準"
    flag = f" {mark}" if delta != 0 else ""
    print(f"  {name:<30s}  {auroc:.4f}  {n:>5d}  {delta_str}{flag}")

# 特徵重要性
if HAS_XGB:
    print("\n=== 特徵重要性 Top 20（全部 Round 3 模型）===")
    tags = {
        "stock_vs_sector_21d": "← 板塊內 alpha",
        "stock_vs_sector_63d": "← 板塊內 alpha",
        "accel_21_63":         "← 動能加速度",
        "accel_63_126":        "← 動能加速度",
        "accel_126_252":       "← 動能加速度",
        "up_months_12":        "← 一致性",
        "vix_level":           "← 恐慌",
        "vix_ma_ratio":        "← 恐慌",
        "sharpe_mom":          "← Sharpe動能",
        "month_sin":           "← 季節",
        "month_cos":           "← 季節",
        "sector_streak":       "← 板塊持續",
        "sector_rel_21d":      "← 板塊強弱",
        "sector_rel_63d":      "← 板塊強弱",
        "sector_rel_126d":     "← 板塊強弱",
        "breadth":             "← 市場廣度",
    }
    imp = sorted(zip(ALL_R3, mdl.feature_importances_), key=lambda x: -x[1])
    for rank, (feat, v) in enumerate(imp[:20], 1):
        tag = tags.get(feat, "")
        print(f"  {rank:2d}. {feat:<25s} {v:.4f}  {tag}")

# 新特徵 delta 摘要
print("\n=== 新特徵增量摘要 ===")
new_feats = [
    ("個股 vs 板塊",    auroc_a  - auroc_prev),
    ("動能加速度",      auroc_b  - auroc_prev),
    ("一致性",          auroc_c  - auroc_prev),
    ("VIX 恐慌",       auroc_d  - auroc_prev),
    ("Sharpe 動能",    auroc_sh - auroc_prev),
]
for name, delta in sorted(new_feats, key=lambda x: -x[1]):
    bar = "█" * max(0, int(delta * 200)) + ("░" * max(0, int(-delta * 200)))
    print(f"  {name:<12s}  {delta:+.4f}  {bar}")

print(f"\n  全部合併增量：{auroc_all - auroc_prev:+.4f}")
if auroc_all > auroc_prev:
    best_single = max(new_feats, key=lambda x: x[1])
    print(f"  單一最佳組：{best_single[0]}（{best_single[1]:+.4f}）")
