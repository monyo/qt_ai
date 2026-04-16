"""
ML 選股：季節性 × 板塊身分 × 持續強度實驗

假設：模型能學到「某板塊在一年中的某時段持續強勢時，能大幅打敗大盤」

上一輪缺口：
  - 缺口1：無時間特徵（模型不知道現在是幾月）
  - 缺口2：板塊只有「現況」（無法區分 XLK vs XLE）
  - 缺口3：板塊只有截面強度（不知道已強勢幾個月）

本輪新增：
  month_sin/cos          ← 季節性（循環編碼）
  sector_XLK/XLF/...     ← 板塊身分（11 個 one-hot）
  sector_streak          ← 連續幾個月板塊跑贏 SPY（最多看回 12 個月）
  sector_rel_126d        ← 6 個月板塊相對強弱（更長窗口）

比較基準：base13 + sector_rel_21d/63d → AUROC 0.6735

驗證：conda run -n qt_env python _ml_sector_temporal.py
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

CACHE_PATH    = "data/_protection_bt_prices.pkl"
FEATURE_CACHE = "data/_ml_features.pkl"
SECTOR_CACHE  = "data/_ml_sector_map.pkl"
ETF_CACHE     = "data/_ml_sector_etf_prices.pkl"

RANDOM_STATE = 42
TRAIN_FRAC   = 0.80
FWD_OFFSET   = 21

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
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))   # 11 個 ETF


# ── 載入快取 ───────────────────────────────────────────────────────────
for p in [CACHE_PATH, FEATURE_CACHE, SECTOR_CACHE, ETF_CACHE]:
    if not os.path.exists(p):
        print(f"找不到：{p}，請先執行 _ml_stock_selector.py 和 _ml_feature_augment.py")
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
etf_prices.index = pd.to_datetime(etf_prices.index).tz_localize(None) \
    if etf_prices.index.tz is None else etf_prices.index.tz_convert(None)
etf_prices = etf_prices.reindex(trading_days)

print(f"基礎特徵：{len(df_base):,} 個樣本\n")


# ── tidx ─────────────────────────────────────────────────────────────
def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))


# ═══════════════════════════════════════════════════════════════════════
# A. 月度板塊相對報酬（預計算，效率高）
# ═══════════════════════════════════════════════════════════════════════
rebalance_dates = pd.bdate_range("2020-12-01", "2025-09-01", freq="BMS")

spy_etf = etf_prices["SPY"]

print("預計算月度板塊相對報酬...")

# etf_monthly_rel[etf] = Series, index=date_str, value=月度相對報酬
etf_monthly_rel = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    ep = etf_prices[etf]
    monthly = {}
    prev_date = None
    for ref in rebalance_dates:
        ti = tidx(str(ref.date()))
        if prev_date is None:
            prev_date = ref
            continue
        ti_prev = tidx(str(prev_date.date()))
        ep_now  = ep.iloc[ti]
        ep_prev = ep.iloc[ti_prev]
        sp_now  = spy_etf.iloc[ti]
        sp_prev = spy_etf.iloc[ti_prev]
        if all(pd.notna(x) and x > 0 for x in [ep_now, ep_prev, sp_now, sp_prev]):
            monthly[str(ref.date())] = (ep_now/ep_prev - 1) - (sp_now/sp_prev - 1)
        prev_date = ref
    etf_monthly_rel[etf] = pd.Series(monthly)

# 板塊 streak：每個月往前數連續跑贏（>0）的月數，最多 12 個月
etf_streak = {}
for etf, series in etf_monthly_rel.items():
    dates  = sorted(series.index)
    streak = {}
    for i, d in enumerate(dates):
        cnt = 0
        for j in range(i - 1, max(i - 13, -1), -1):
            if series[dates[j]] > 0:
                cnt += 1
            else:
                break
        streak[d] = cnt
    etf_streak[etf] = pd.Series(streak)

# 板塊 rel_126d（6 個月相對報酬）：用 etf_prices 直接算
etf_rel_126 = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    rel = etf_prices[etf] / spy_etf
    etf_rel_126[etf] = (rel / rel.shift(126) - 1) * 100

# 板塊 rel_21d / rel_63d（與上一輪相同，重算供組合用）
etf_rel_21 = {}
etf_rel_63 = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    rel = etf_prices[etf] / spy_etf
    etf_rel_21[etf] = (rel / rel.shift(21)  - 1) * 100
    etf_rel_63[etf] = (rel / rel.shift(63)  - 1) * 100

print("板塊預計算完成\n")


# ═══════════════════════════════════════════════════════════════════════
# B. 建立擴充特徵矩陣
# ═══════════════════════════════════════════════════════════════════════
def sym_to_etf(sym):
    s = sector_map.get(sym)
    return SECTOR_ETF_MAP.get(s) if s else None

print("建立擴充特徵矩陣...")
df = df_base.copy()

# ── 時間特徵 ──────────────────────────────────────────────────────────
months = pd.to_datetime(df["date"]).dt.month
df["month_sin"] = np.sin(2 * np.pi * months / 12)
df["month_cos"] = np.cos(2 * np.pi * months / 12)

# ── 板塊特徵（向量化：避免逐行 loop）─────────────────────────────────
# 先建 date → ti 的快取
date_ti_map = {str(ref.date()): tidx(str(ref.date()))
               for ref in pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")}

# 每個樣本對應的 ETF
df["_etf"] = df["symbol"].map(sym_to_etf)

# 逐 ETF 批次填充（比逐行快很多）
for feat_name, feat_dict in [
    ("sector_rel_21d",  etf_rel_21),
    ("sector_rel_63d",  etf_rel_63),
    ("sector_rel_126d", etf_rel_126),
]:
    col_vals = np.full(len(df), np.nan)
    for etf, series in feat_dict.items():
        mask = df["_etf"] == etf
        if not mask.any():
            continue
        idx_list = df.index[mask]
        for idx in idx_list:
            ti = date_ti_map.get(df.at[idx, "date"])
            if ti is None:
                continue
            v = series.iloc[ti] if ti < len(series) else np.nan
            col_vals[df.index.get_loc(idx)] = float(v) if pd.notna(v) else np.nan
    df[feat_name] = col_vals

# sector_streak（月度，用 date_str 查）
col_streak = np.zeros(len(df), dtype=float)
for etf, series in etf_streak.items():
    mask = df["_etf"] == etf
    if not mask.any():
        continue
    for pos, (idx, row) in enumerate(df[mask].iterrows()):
        v = series.get(row["date"], np.nan)
        col_streak[df.index.get_loc(idx)] = v
df["sector_streak"] = col_streak

# ── 板塊 one-hot ───────────────────────────────────────────────────────
for etf in SECTOR_ETFS:
    df[f"sec_{etf}"] = (df["_etf"] == etf).astype(float)

df = df.drop(columns=["_etf"])

print(f"擴充完成，共 {len(df.columns)} 欄\n")


# ═══════════════════════════════════════════════════════════════════════
# C. Train/Test 切分（與前兩輪相同 seed）
# ═══════════════════════════════════════════════════════════════════════
all_syms = df["symbol"].unique()
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_syms)
n_train   = int(len(all_syms) * TRAIN_FRAC)
train_syms = set(all_syms[:n_train])
test_syms  = set(all_syms[n_train:])

df_tr = df[df["symbol"].isin(train_syms)]
df_te = df[df["symbol"].isin(test_syms)]
y_tr  = df_tr["y"].values
y_te  = df_te["y"].values


# ═══════════════════════════════════════════════════════════════════════
# D. 特徵組合定義
# ═══════════════════════════════════════════════════════════════════════
BASE = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct", "breadth",
]
SECTOR_BASIC = BASE + ["sector_rel_21d", "sector_rel_63d"]          # 上一輪最佳
SEASON       = BASE + ["sector_rel_21d", "sector_rel_63d",
                        "month_sin", "month_cos"]
SECTOR_ID    = BASE + ["sector_rel_21d", "sector_rel_63d"] + \
               [f"sec_{e}" for e in SECTOR_ETFS]
STREAK       = BASE + ["sector_rel_21d", "sector_rel_63d",
                        "sector_rel_126d", "sector_streak"]
ALL_NEW      = BASE + ["sector_rel_21d", "sector_rel_63d",
                        "sector_rel_126d", "sector_streak",
                        "month_sin", "month_cos"] + \
               [f"sec_{e}" for e in SECTOR_ETFS]


# ═══════════════════════════════════════════════════════════════════════
# E. 訓練與評估
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

auroc_base,   _   = run_xgb(BASE,         "base13")
auroc_prev,   _   = run_xgb(SECTOR_BASIC, "base+sector_21/63d")
auroc_season, _   = run_xgb(SEASON,       "+季節(sin/cos)")
auroc_id,     _   = run_xgb(SECTOR_ID,    "+板塊身分(one-hot)")
auroc_streak, _   = run_xgb(STREAK,       "+streak+126d")
auroc_all,    mdl = run_xgb(ALL_NEW,      "全部新特徵")

rows = [
    ("基準（原 13 特徵）",               auroc_base,   len(BASE),         0),
    ("+ 板塊相對強弱 21/63d",           auroc_prev,   len(SECTOR_BASIC), auroc_prev   - auroc_base),
    ("+ 季節性 (month_sin/cos)",        auroc_season, len(SEASON),       auroc_season - auroc_prev),
    ("+ 板塊身分 (one-hot 11)",         auroc_id,     len(SECTOR_ID),    auroc_id     - auroc_prev),
    ("+ 板塊 streak + 126d",            auroc_streak, len(STREAK),       auroc_streak - auroc_prev),
    ("全部新特徵",                       auroc_all,    len(ALL_NEW),      auroc_all    - auroc_base),
]

print(f"  {'組合':<30s}  {'AUROC':>7s}  {'特徵數':>5s}  {'增量':>8s}")
print("  " + "-" * 60)
for name, auroc, n, delta in rows:
    arrow = "✅" if delta > 0.005 else ("➖" if abs(delta) < 0.002 else "⚠️ ")
    prefix = "  " if delta == 0 else f"{arrow}"
    delta_str = f"{delta:+.4f}" if delta != 0 else "  基準"
    print(f"  {name:<30s}  {auroc:.4f}  {n:>5d}  {delta_str}")

# ── 特徵重要性（全部新特徵模型）──────────────────────────────────────
if HAS_XGB:
    print("\n=== 特徵重要性（全部新特徵模型）===")
    imp = sorted(zip(ALL_NEW, mdl.feature_importances_), key=lambda x: -x[1])
    for rank, (feat, v) in enumerate(imp[:15], 1):
        tag = ""
        if feat in ("month_sin", "month_cos"):
            tag = " ← 季節"
        elif feat.startswith("sec_"):
            tag = " ← 板塊身分"
        elif feat in ("sector_streak",):
            tag = " ← 持續強度"
        elif "sector_rel" in feat:
            tag = " ← 板塊強弱"
        print(f"  {rank:2d}. {feat:<22s} {v:.4f}{tag}")

# ── 季節性深挖：每個月的平均 y label ─────────────────────────────────
print("\n=== 月份 × 板塊打敗大盤比例（Train set，前 5 名板塊）===")
df_tr_copy = df_tr.copy()
df_tr_copy["month"] = pd.to_datetime(df_tr_copy["date"]).dt.month
df_tr_copy["etf"] = df_tr_copy["symbol"].map(sym_to_etf)

top_etfs = df_tr_copy["etf"].value_counts().head(5).index.tolist()
pivot = df_tr_copy[df_tr_copy["etf"].isin(top_etfs)].pivot_table(
    index="etf", columns="month", values="y", aggfunc="mean"
)
print(pivot.round(2).to_string())

print("\n=== 解讀 ===")
best_gain = max(auroc_season - auroc_prev, auroc_id - auroc_prev,
                auroc_streak - auroc_prev)
if auroc_season - auroc_prev > 0.005:
    print("  ✅ 季節性特徵有效：模型確實學到時間軸上的板塊規律")
else:
    print("  ➖ 季節性特徵效果有限（板塊輪動季節性在此資料中不顯著）")
if auroc_id - auroc_prev > 0.005:
    print("  ✅ 板塊身分有效：不同板塊有不同的持續性，非同質")
else:
    print("  ➖ 板塊身分效果有限（板塊差異已被相對強弱特徵捕捉）")
if auroc_streak - auroc_prev > 0.005:
    print("  ✅ streak 有效：持續強勢的板塊未來繼續跑贏的機率更高")
else:
    print("  ➖ streak 效果有限（短期強弱已足夠，持續時間不提供額外訊號）")
