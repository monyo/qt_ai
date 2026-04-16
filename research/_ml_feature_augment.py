"""
ML 選股特徵擴充實驗

基礎：_ml_stock_selector.py（13 個特徵，XGBoost AUROC 0.645）
新增三組特徵，逐組比較 AUROC：
  Group V（交易量）：vol_ratio_21_63（量能趨勢）, rel_vol_5d（異常量能）
  Group S（板塊）  ：sector_rel_21d, sector_rel_63d（板塊輪動 vs SPY）
  Group M（市值）  ：log_mktcap（規模因子，靜態近似，非歷史精確）

驗證：conda run -n qt_env python _ml_feature_augment.py
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import warnings
import time
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
    HAS_XGB = False

CACHE_PATH    = "data/_protection_bt_prices.pkl"
FEATURE_CACHE = "data/_ml_features.pkl"
VOL_CACHE     = "data/_ml_vol_cache.pkl"
SECTOR_CACHE  = "data/_ml_sector_map.pkl"
MKTCAP_CACHE  = "data/_ml_mktcap_map.pkl"

RANDOM_STATE  = 42
TRAIN_FRAC    = 0.80
FWD_OFFSET    = 21

# GICS 板塊 → 對應 ETF
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


# ── 載入基礎快取 ───────────────────────────────────────────────────────
if not os.path.exists(CACHE_PATH):
    print(f"找不到快取 {CACHE_PATH}"); raise SystemExit(1)
if not os.path.exists(FEATURE_CACHE):
    print(f"找不到特徵快取 {FEATURE_CACHE}，請先執行 _ml_stock_selector.py")
    raise SystemExit(1)

print(f"讀取價格快取：{CACHE_PATH}")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
symbols = list(prices.columns)
print(f"資料：{len(symbols)} 檔 / {len(trading_days)} 交易日\n")

print(f"讀取特徵快取：{FEATURE_CACHE}")
df_base = pd.read_pickle(FEATURE_CACHE)
print(f"基礎樣本：{len(df_base):,} 個\n")


# ═══════════════════════════════════════════════════════════════════════
# A. 交易量資料
# ═══════════════════════════════════════════════════════════════════════
def load_volume():
    if os.path.exists(VOL_CACHE):
        print(f"讀取交易量快取：{VOL_CACHE}")
        return pd.read_pickle(VOL_CACHE)

    print(f"下載交易量資料（{len(symbols)} 支，約需 2-3 分鐘）...")
    raw = yf.download(
        symbols, start="2020-01-01", end="2026-03-01",
        auto_adjust=True, progress=False, group_by="column"
    )
    if isinstance(raw.columns, pd.MultiIndex):
        vol = raw["Volume"]
    else:
        vol = raw[["Volume"]]

    vol.index = pd.to_datetime(vol.index).tz_localize(None)
    vol.to_pickle(VOL_CACHE)
    print(f"已快取至 {VOL_CACHE}\n")
    return vol

print("=== Group V：交易量特徵 ===")
vol_df = load_volume()
vol_df = vol_df.reindex(trading_days)   # 對齊價格索引

# 預計算 rolling 平均量（避免逐樣本計算）
vol_ma5  = vol_df.rolling(5,  min_periods=3).mean()
vol_ma21 = vol_df.rolling(21, min_periods=10).mean()
vol_ma63 = vol_df.rolling(63, min_periods=30).mean()

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """為每個樣本加入量能特徵"""
    df = df.copy()
    rows_ratio = []
    rows_spike = []
    for _, row in df.iterrows():
        sym  = row["symbol"]
        date = pd.Timestamp(row["date"])
        ti   = int(trading_days.searchsorted(date))

        if ti < 63 or sym not in vol_df.columns:
            rows_ratio.append(np.nan); rows_spike.append(np.nan)
            continue

        r21_63 = vol_ma21.iloc[ti].get(sym)
        r5_21  = vol_ma5.iloc[ti].get(sym)
        ma63   = vol_ma63.iloc[ti].get(sym)
        ma21   = vol_ma21.iloc[ti].get(sym)

        ratio = (float(r21_63) / float(ma63)) if (
            r21_63 and ma63 and pd.notna(r21_63) and pd.notna(ma63) and ma63 > 0
        ) else np.nan
        spike = (float(r5_21) / float(ma21)) if (
            r5_21 and ma21 and pd.notna(r5_21) and pd.notna(ma21) and ma21 > 0
        ) else np.nan

        rows_ratio.append(ratio)
        rows_spike.append(spike)

    df["vol_ratio_21_63"] = rows_ratio
    df["rel_vol_5d"]      = rows_spike
    return df

print("計算交易量特徵（逐樣本，需幾分鐘）...")
t0 = time.time()
df_vol = add_volume_features(df_base)
print(f"完成（{time.time()-t0:.0f}s），有效樣本：vol_ratio {df_vol['vol_ratio_21_63'].notna().sum():,} / rel_vol {df_vol['rel_vol_5d'].notna().sum():,}\n")


# ═══════════════════════════════════════════════════════════════════════
# B. 板塊相對強弱
# ═══════════════════════════════════════════════════════════════════════
print("=== Group S：板塊特徵 ===")

# B1. 板塊 ETF 歷史收盤價
etf_cache = "data/_ml_sector_etf_prices.pkl"
if os.path.exists(etf_cache):
    etf_prices = pd.read_pickle(etf_cache)
else:
    print(f"下載板塊 ETF 報價：{SECTOR_ETFS}")
    raw = yf.download(SECTOR_ETFS + ["SPY"],
                      start="2020-01-01", end="2026-03-01",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        etf_prices = raw["Close"]
    else:
        etf_prices = raw
    etf_prices.index = pd.to_datetime(etf_prices.index).tz_localize(None)
    etf_prices.to_pickle(etf_cache)

etf_prices = etf_prices.reindex(trading_days)

# 板塊 ETF rolling return（相對 SPY）
spy_etf = etf_prices["SPY"]
etf_rel_21 = {}
etf_rel_63 = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    # 相對報酬 = ETF / SPY 的 rolling ratio
    rel = etf_prices[etf] / spy_etf
    etf_rel_21[etf] = rel / rel.shift(21) - 1    # 21d 板塊 vs SPY
    etf_rel_63[etf] = rel / rel.shift(63) - 1    # 63d 板塊 vs SPY

# B2. 股票 → 板塊 mapping
if os.path.exists(SECTOR_CACHE):
    sector_map = pd.read_pickle(SECTOR_CACHE)
    print(f"讀取板塊 mapping：{len(sector_map)} 支")
else:
    print(f"抓取板塊資訊（{len(symbols)} 支，約需 3-5 分鐘）...")
    sector_map = {}
    for i, sym in enumerate(symbols):
        try:
            info = yf.Ticker(sym).fast_info
            # fast_info 沒有 sector，改用 info
            s = yf.Ticker(sym).info.get("sector", None)
            sector_map[sym] = s
        except Exception:
            sector_map[sym] = None
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)} 完成...")
            time.sleep(1)   # 避免被 rate limit

    pd.Series(sector_map).to_pickle(SECTOR_CACHE)
    print(f"已快取至 {SECTOR_CACHE}\n")

def sym_to_etf(sym):
    """股票代碼 → 板塊 ETF 代碼"""
    sector = sector_map.get(sym)
    if not sector:
        return None
    return SECTOR_ETF_MAP.get(sector)

def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows_21 = []
    rows_63 = []
    for _, row in df.iterrows():
        sym  = row["symbol"]
        date = pd.Timestamp(row["date"])
        ti   = int(trading_days.searchsorted(date))
        etf  = sym_to_etf(sym)
        if etf is None or ti < 63:
            rows_21.append(np.nan); rows_63.append(np.nan)
            continue
        v21 = etf_rel_21.get(etf)
        v63 = etf_rel_63.get(etf)
        r21 = float(v21.iloc[ti]) if v21 is not None and pd.notna(v21.iloc[ti]) else np.nan
        r63 = float(v63.iloc[ti]) if v63 is not None and pd.notna(v63.iloc[ti]) else np.nan
        rows_21.append(r21 * 100)
        rows_63.append(r63 * 100)

    df["sector_rel_21d"] = rows_21
    df["sector_rel_63d"] = rows_63
    return df

print("計算板塊相對強弱特徵...")
t0 = time.time()
df_sector = add_sector_features(df_vol)
print(f"完成（{time.time()-t0:.0f}s），有效：{df_sector['sector_rel_21d'].notna().sum():,}\n")


# ═══════════════════════════════════════════════════════════════════════
# C. 市值（靜態近似，非歷史精確）
# ═══════════════════════════════════════════════════════════════════════
print("=== Group M：市值特徵 ===")

if os.path.exists(MKTCAP_CACHE):
    mktcap_map = pd.read_pickle(MKTCAP_CACHE)
    print(f"讀取市值 mapping：{len(mktcap_map)} 支")
else:
    print(f"抓取市值資訊（{len(symbols)} 支）...")
    mktcap_map = {}
    for i, sym in enumerate(symbols):
        try:
            mc = yf.Ticker(sym).info.get("marketCap", None)
            mktcap_map[sym] = float(mc) if mc else np.nan
        except Exception:
            mktcap_map[sym] = np.nan
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)} 完成...")
            time.sleep(0.5)
    pd.Series(mktcap_map).to_pickle(MKTCAP_CACHE)
    print(f"已快取至 {MKTCAP_CACHE}\n")

def add_mktcap_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_caps = []
    for sym in df["symbol"]:
        mc = mktcap_map.get(sym, np.nan)
        log_caps.append(np.log10(mc) if mc and pd.notna(mc) and mc > 0 else np.nan)
    df["log_mktcap"] = log_caps
    return df

df_full = add_mktcap_feature(df_sector)
print(f"市值有效：{df_full['log_mktcap'].notna().sum():,}\n")


# ═══════════════════════════════════════════════════════════════════════
# 模型訓練工具
# ═══════════════════════════════════════════════════════════════════════
BASE_COLS = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct", "breadth",
]

# 按股票 80/20 切分（與 v1 相同 seed）
all_syms = df_full["symbol"].unique()
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_syms)
n_train = int(len(all_syms) * TRAIN_FRAC)
train_syms = set(all_syms[:n_train])
test_syms  = set(all_syms[n_train:])

df_tr = df_full[df_full["symbol"].isin(train_syms)]
df_te = df_full[df_full["symbol"].isin(test_syms)]
y_tr  = df_tr["y"].values
y_te  = df_te["y"].values


def run_xgb(feat_cols, df_train, df_test, y_train, y_test, label):
    X_tr = df_train[feat_cols].copy()
    X_te = df_test[feat_cols].copy()
    med  = X_tr.median()
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE)

    model.fit(X_tr, y_train)
    prob = model.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_test, prob)
    print(f"  {label:<30s}  AUROC = {auroc:.4f}  ({len(feat_cols)} 特徵)")
    return auroc, model, scaler, med


# ═══════════════════════════════════════════════════════════════════════
# 比較實驗
# ═══════════════════════════════════════════════════════════════════════
print("=== AUROC 比較（XGBoost，Test set）===\n")

# 基準
auroc_base, _, _, _ = run_xgb(BASE_COLS, df_tr, df_te, y_tr, y_te, "基準（原 13 特徵）")

# +交易量
vol_cols = BASE_COLS + ["vol_ratio_21_63", "rel_vol_5d"]
auroc_v, _, _, _ = run_xgb(vol_cols, df_tr, df_te, y_tr, y_te, "+交易量（15 特徵）")

# +板塊
sec_cols = BASE_COLS + ["sector_rel_21d", "sector_rel_63d"]
auroc_s, _, _, _ = run_xgb(sec_cols, df_tr, df_te, y_tr, y_te, "+板塊（15 特徵）")

# +市值
mc_cols = BASE_COLS + ["log_mktcap"]
auroc_m, _, _, _ = run_xgb(mc_cols, df_tr, df_te, y_tr, y_te, "+市值（14 特徵）")

# 全部
all_cols = BASE_COLS + ["vol_ratio_21_63", "rel_vol_5d",
                         "sector_rel_21d", "sector_rel_63d", "log_mktcap"]
auroc_all, best_model, best_scaler, best_med = run_xgb(
    all_cols, df_tr, df_te, y_tr, y_te, "全部（18 特徵）"
)

print()
print(f"  增量：+交易量 {auroc_v - auroc_base:+.4f}  |  +板塊 {auroc_s - auroc_base:+.4f}  |  +市值 {auroc_m - auroc_base:+.4f}  |  全部 {auroc_all - auroc_base:+.4f}")

# 特徵重要性（全部特徵模型）
if HAS_XGB:
    print("\n=== 特徵重要性（全部 18 特徵模型）===")
    imp = sorted(zip(all_cols, best_model.feature_importances_), key=lambda x: -x[1])
    for rank, (feat, v) in enumerate(imp, 1):
        print(f"  {rank:2d}. {feat:<20s} {v:+.4f}")

print("\n=== 結論 ===")
if auroc_s - auroc_base > 0.005:
    print("  板塊特徵有顯著貢獻（delta > 0.005）")
else:
    print("  板塊特徵貢獻有限（可能已隱含在動能特徵中）")
if auroc_v - auroc_base > 0.005:
    print("  交易量特徵有顯著貢獻")
else:
    print("  交易量特徵貢獻有限")
if auroc_m - auroc_base > 0.005:
    print("  市值特徵有顯著貢獻（注意：靜態市值有前瞻偏誤）")
else:
    print("  市值特徵貢獻有限（且靜態市值本身有前瞻偏誤，不建議採用）")
