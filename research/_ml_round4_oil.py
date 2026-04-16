"""
ML 選股 Round 4：石油價格特徵

動機：股癌提到看 VIX 之外也看石油價格。
      推測邏輯：低VIX + 平穩石油 = 健康風險偏好；高VIX + 飆升石油 = 滯脹恐慌。

油價來源：WTI 原油期貨 "CL=F"（yfinance），備用 USO ETF

新增特徵：
  oil_ret_21d     ← 21 日油價報酬
  oil_ret_63d     ← 63 日油價報酬
  oil_ma_ratio    ← 油價 / 200MA（油價 regime：牛/熊）
  oil_vs_spy_21d  ← 油價相對 SPY（商品 vs 股票輪動訊號）

驗證：conda run -n qt_env python _ml_round4_oil.py
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
VIX_CACHE     = "data/_ml_vix.pkl"
OIL_CACHE     = "data/_ml_oil.pkl"

RANDOM_STATE = 42
TRAIN_FRAC   = 0.80

SECTOR_ETF_MAP = {
    "Technology":             "XLK", "Information Technology": "XLK",
    "Financial Services":     "XLF", "Financials":             "XLF",
    "Healthcare":             "XLV", "Health Care":            "XLV",
    "Energy":                 "XLE", "Utilities":              "XLU",
    "Industrials":            "XLI", "Basic Materials":        "XLB",
    "Materials":              "XLB", "Real Estate":            "XLRE",
    "Consumer Cyclical":      "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP", "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))


# ── 載入快取 ───────────────────────────────────────────────────────────
for p in [CACHE_PATH, FEATURE_CACHE, SECTOR_CACHE, ETF_CACHE]:
    if not os.path.exists(p):
        print(f"找不到：{p}"); raise SystemExit(1)

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


# ── 下載石油價格 ───────────────────────────────────────────────────────
def load_oil():
    if os.path.exists(OIL_CACHE):
        oil = pd.read_pickle(OIL_CACHE)
        print(f"讀取石油快取：{OIL_CACHE}（{len(oil)} 筆）")
        return oil

    print("下載石油價格（WTI CL=F）...")
    oil = yf.Ticker("CL=F").history(
        start="2020-01-01", end="2026-03-01", auto_adjust=True
    )["Close"]
    oil.index = pd.to_datetime(oil.index).tz_localize(None)

    # 若 CL=F 資料過少，改用 USO ETF 作代理
    if len(oil.dropna()) < 200:
        print("  CL=F 資料不足，改用 USO ETF 作代理...")
        oil = yf.Ticker("USO").history(
            start="2020-01-01", end="2026-03-01", auto_adjust=True
        )["Close"]
        oil.index = pd.to_datetime(oil.index).tz_localize(None)
        print(f"  USO 資料：{len(oil.dropna())} 筆")
    else:
        print(f"  CL=F 資料：{len(oil.dropna())} 筆")

    oil.to_pickle(OIL_CACHE)
    print(f"已快取至 {OIL_CACHE}\n")
    return oil

oil = load_oil()
oil_aligned = oil.reindex(trading_days, method="ffill")

# 石油指標（向量化預計算）
oil_ma200   = oil_aligned.rolling(200, min_periods=100).mean()
spy_etf     = etf_prices["SPY"]
oil_ret_21  = (oil_aligned / oil_aligned.shift(21)  - 1) * 100
oil_ret_63  = (oil_aligned / oil_aligned.shift(63)  - 1) * 100
oil_ma_r    = oil_aligned / oil_ma200                        # >1 = 油價牛市
oil_spy_21  = oil_ret_21 - (spy_etf / spy_etf.shift(21) - 1) * 100  # 油 vs 股

# VIX
if os.path.exists(VIX_CACHE):
    vix = pd.read_pickle(VIX_CACHE)
else:
    vix = yf.Ticker("^VIX").history(start="2020-01-01", end="2026-03-01",
                                     auto_adjust=True)["Close"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.to_pickle(VIX_CACHE)
vix = vix.reindex(trading_days, method="ffill")
vix_ma63    = vix.rolling(63, min_periods=30).mean()

print()


def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

date_ti_map = {str(ref.date()): tidx(str(ref.date()))
               for ref in pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")}


# ── 建立擴充特徵矩陣 ───────────────────────────────────────────────────
print("計算石油 + VIX 特徵（市場環境，所有股票共用）...")

df = df_base.copy()

oil_r21_vals = np.full(len(df), np.nan)
oil_r63_vals = np.full(len(df), np.nan)
oil_mar_vals = np.full(len(df), np.nan)
oil_spy_vals = np.full(len(df), np.nan)
vix_lv_vals  = np.full(len(df), np.nan)
vix_mr_vals  = np.full(len(df), np.nan)

for i, (idx, row) in enumerate(df.iterrows()):
    ti = date_ti_map.get(row["date"])
    if ti is None:
        continue

    def safe(series, ti):
        v = series.iloc[ti] if ti < len(series) else np.nan
        return float(v) if pd.notna(v) else np.nan

    oil_r21_vals[i] = safe(oil_ret_21, ti)
    oil_r63_vals[i] = safe(oil_ret_63, ti)
    oil_mar_vals[i] = safe(oil_ma_r,   ti)
    oil_spy_vals[i] = safe(oil_spy_21, ti)
    vl = safe(vix, ti);  vm = safe(vix_ma63, ti)
    vix_lv_vals[i]  = vl
    vix_mr_vals[i]  = vl / vm if (pd.notna(vl) and pd.notna(vm) and vm > 0) else np.nan

df["oil_ret_21d"]    = oil_r21_vals
df["oil_ret_63d"]    = oil_r63_vals
df["oil_ma_ratio"]   = oil_mar_vals
df["oil_vs_spy_21d"] = oil_spy_vals
df["vix_level"]      = vix_lv_vals
df["vix_ma_ratio"]   = vix_mr_vals

print(f"  油價有效：{df['oil_ret_21d'].notna().sum():,}  VIX 有效：{df['vix_level'].notna().sum():,}")

# 從 _ml_sector_temporal 重建板塊 + 季節特徵（複用邏輯）
months = pd.to_datetime(df["date"]).dt.month
df["month_sin"] = np.sin(2 * np.pi * months / 12)
df["month_cos"] = np.cos(2 * np.pi * months / 12)

def sym_to_etf(sym):
    s = sector_map.get(sym)
    return SECTOR_ETF_MAP.get(s) if s else None

df["_etf"] = df["symbol"].map(sym_to_etf)
for etf in SECTOR_ETFS:
    df[f"sec_{etf}"] = (df["_etf"] == etf).astype(float)

spy_ref = etf_prices["SPY"]
for win, col in [(21, "sector_rel_21d"), (63, "sector_rel_63d"), (126, "sector_rel_126d")]:
    rel_dict = {}
    for etf in SECTOR_ETFS:
        if etf not in etf_prices.columns:
            continue
        rel = etf_prices[etf] / spy_ref
        rel_dict[etf] = (rel / rel.shift(win) - 1) * 100
    col_vals = np.full(len(df), np.nan)
    for etf in SECTOR_ETFS:
        mask = df["_etf"] == etf
        if not mask.any() or etf not in rel_dict:
            continue
        series = rel_dict[etf]
        for idx2 in df.index[mask]:
            ti = date_ti_map.get(df.at[idx2, "date"])
            if ti is not None and ti < len(series):
                v = series.iloc[ti]
                if pd.notna(v):
                    col_vals[df.index.get_loc(idx2)] = float(v)
    df[col] = col_vals

# sector_streak
rebal_all = pd.bdate_range("2020-12-01", "2025-09-01", freq="BMS")
etf_monthly_rel = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    ep = etf_prices[etf]
    monthly = {}
    prev_ref = None
    for ref in rebal_all:
        ti = tidx(str(ref.date()))
        if prev_ref is None:
            prev_ref = ref; continue
        tp = tidx(str(prev_ref.date()))
        sp_n, sp_p = spy_ref.iloc[ti], spy_ref.iloc[tp]
        ep_n, ep_p = ep.iloc[ti], ep.iloc[tp]
        if all(pd.notna(x) and x > 0 for x in [ep_n, ep_p, sp_n, sp_p]):
            monthly[str(ref.date())] = (ep_n/ep_p - 1) - (sp_n/sp_p - 1)
        prev_ref = ref
    ms = pd.Series(monthly)
    dates = sorted(ms.index)
    streak = {}
    for i, d in enumerate(dates):
        cnt = 0
        for j in range(i-1, max(i-13, -1), -1):
            if ms[dates[j]] > 0: cnt += 1
            else: break
        streak[d] = cnt
    etf_monthly_rel[etf] = pd.Series(streak)

col_streak = np.zeros(len(df), dtype=float)
for etf, series in etf_monthly_rel.items():
    mask = df["_etf"] == etf
    for idx2 in df.index[mask]:
        v = series.get(df.at[idx2, "date"], np.nan)
        col_streak[df.index.get_loc(idx2)] = v
df["sector_streak"] = col_streak
df = df.drop(columns=["_etf"])

print(f"特徵計算完成，df 共 {len(df.columns)} 欄\n")


# ── Train/Test 切分 ────────────────────────────────────────────────────
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


# ── 特徵組合 ───────────────────────────────────────────────────────────
PREV_BEST = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct", "breadth",
    "sector_rel_21d", "sector_rel_63d", "sector_rel_126d", "sector_streak",
    "month_sin", "month_cos",
] + [f"sec_{e}" for e in SECTOR_ETFS]     # Round 3 最佳基礎（含 VIX 之前）

WITH_VIX  = PREV_BEST + ["vix_level", "vix_ma_ratio"]
WITH_OIL  = PREV_BEST + ["oil_ret_21d", "oil_ret_63d", "oil_ma_ratio", "oil_vs_spy_21d"]
WITH_BOTH = PREV_BEST + ["vix_level", "vix_ma_ratio",
                          "oil_ret_21d", "oil_ret_63d", "oil_ma_ratio", "oil_vs_spy_21d"]


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
    auroc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return auroc, model


# ── 比較 ───────────────────────────────────────────────────────────────
print("=== AUROC 比較（XGBoost，Test set）===\n")

auroc_prev, _     = run_xgb(PREV_BEST, "prev_best")
auroc_vix,  _     = run_xgb(WITH_VIX,  "+VIX")
auroc_oil,  _     = run_xgb(WITH_OIL,  "+石油")
auroc_both, mdl   = run_xgb(WITH_BOTH, "+VIX+石油")

rows = [
    ("上一輪基礎（30 特徵）",             auroc_prev, len(PREV_BEST), 0),
    ("+ VIX（已驗證 +0.0055）",          auroc_vix,  len(WITH_VIX),  auroc_vix  - auroc_prev),
    ("+ 石油 4 特徵",                    auroc_oil,  len(WITH_OIL),  auroc_oil  - auroc_prev),
    ("+ VIX + 石油（組合）",              auroc_both, len(WITH_BOTH), auroc_both - auroc_prev),
]

print(f"  {'組合':<30s}  {'AUROC':>7s}  {'特徵數':>5s}  {'增量':>8s}")
print("  " + "-" * 58)
for name, auroc, n, delta in rows:
    mark = "✅" if delta > 0.005 else ("➖" if abs(delta) <= 0.002 else "⚠️ ")
    delta_str = f"{delta:+.4f}" if delta != 0 else "  基準"
    flag = f" {mark}" if delta != 0 else ""
    print(f"  {name:<30s}  {auroc:.4f}  {n:>5d}  {delta_str}{flag}")

# 特徵重要性（含VIX+石油的模型）
if HAS_XGB:
    print("\n=== 特徵重要性 Top 15（VIX + 石油組合模型）===")
    env_feats = {"vix_level", "vix_ma_ratio",
                 "oil_ret_21d", "oil_ret_63d", "oil_ma_ratio", "oil_vs_spy_21d"}
    imp = sorted(zip(WITH_BOTH, mdl.feature_importances_), key=lambda x: -x[1])
    for rank, (feat, v) in enumerate(imp[:15], 1):
        tag = " ← 市場環境" if feat in env_feats else ""
        tag = " ← VIX" if "vix" in feat else tag
        tag = " ← 石油" if "oil" in feat else tag
        print(f"  {rank:2d}. {feat:<22s} {v:.4f}{tag}")

# VIX × 石油交叉分析
print("\n=== VIX × 石油環境分析（Train set 打敗大盤比例）===\n")
df_tr_copy = df_tr.copy()
# 分組：VIX 高/低 × 油價漲/跌
df_tr_copy["vix_high"] = df_tr_copy["vix_level"] >= 20
df_tr_copy["oil_up"]   = df_tr_copy["oil_ret_21d"] >= 0

groups = {
    "低VIX + 油價平穩/下跌（最佳動能環境）": (~df_tr_copy["vix_high"]) & (~df_tr_copy["oil_up"]),
    "低VIX + 油價上漲（健康風險偏好）":       (~df_tr_copy["vix_high"]) & df_tr_copy["oil_up"],
    "高VIX + 油價下跌（恐慌但通縮）":        df_tr_copy["vix_high"]  & (~df_tr_copy["oil_up"]),
    "高VIX + 油價上漲（滯脹恐慌）":          df_tr_copy["vix_high"]  & df_tr_copy["oil_up"],
}

for desc, mask in groups.items():
    n    = mask.sum()
    rate = df_tr_copy.loc[mask, "y"].mean() * 100 if n > 0 else 0
    avg_alpha = df_tr_copy.loc[mask, "fwd_alpha"].mean() if n > 0 else 0
    print(f"  {desc}")
    print(f"    樣本 {n:,}  打敗大盤 {rate:.1f}%  平均 alpha {avg_alpha:+.2f}%\n")
