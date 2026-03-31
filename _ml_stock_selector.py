"""
ML 選股實驗

核心假設：動能、波動率、技術位置等特徵的組合存在非線性交互效應，
是手動規則無法捕捉的隱藏選股訊號。

設計：(股票 × 月份) pair 隨機切分
- 每個樣本 = 某支股票在某月的特徵快照（滑動視窗，過去資料計算特徵）
- 按股票 80/20 切分（整支股票的所有月份一起進 train 或 test）
- 目標：打敗 SPY 1M alpha > 0（二元分類）

驗證方式：
  conda run -n qt_env python _ml_stock_selector.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── 嘗試 import xgboost ────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("提示：xgboost 未安裝，跳過 XGBoost（pip install xgboost）")

CACHE_PATH    = "data/_protection_bt_prices.pkl"
FEATURE_CACHE = "data/_ml_features.pkl"
MOM_SHORT     = 21
MOM_LONG      = 252
MA_WIN        = 50
TRAIN_FRAC    = 0.80
RANDOM_STATE  = 42
TOP_POOL      = 150   # 每月動能池
TOP_N         = 5     # 回測每月選幾支
FWD_OFFSET    = 21    # 預測窗口（1 個月≈21 交易日）

# ── 載入資料 ──────────────────────────────────────────────────────────
if not os.path.exists(CACHE_PATH):
    print(f"找不到快取 {CACHE_PATH}")
    raise SystemExit(1)

print(f"讀取快取：{CACHE_PATH}")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"資料：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日\n")

spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)

# ── 預計算全市場指標（向量化）────────────────────────────────────────
print("預計算技術指標...")
ma50        = prices.rolling(MA_WIN, min_periods=30).mean()
ma200       = prices.rolling(200,    min_periods=100).mean()
above_ma50  = (prices > ma50).astype(int)
valid_count = (prices.notna() & ma50.notna()).sum(axis=1)
above_count = above_ma50.sum(axis=1)
breadth_series = (above_count / valid_count).where(valid_count > 100)

rolling_min40  = prices.rolling(40, min_periods=20).min()
rolling_max40  = prices.rolling(40, min_periods=20).max()
rolling_max252 = prices.rolling(252, min_periods=100).max()
bounce_pct_df  = (prices / rolling_min40  - 1) * 100
from_high_df   = (prices / rolling_max40  - 1) * 100
near52_df      = prices / rolling_max252          # (0, 1]

# 多窗口 log-return 序列（用於波動率）
log_ret = np.log(prices / prices.shift(1))

print("預計算完成\n")


# ── 工具函式 ──────────────────────────────────────────────────────────
def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

def price_at(sym, ti):
    if ti < 0 or ti >= len(trading_days):
        return None
    v = prices.iloc[ti].get(sym)
    return float(v) if v is not None and pd.notna(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    p0   = price_at(sym, ti)
    p21  = price_at(sym, ti - MOM_SHORT)
    p252 = price_at(sym, ti - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1)*100 + 0.5 * (p0/p252 - 1)*100

def get_breadth(ti):
    b = breadth_series.iloc[ti] if ti < len(breadth_series) else None
    return float(b) if b is not None and pd.notna(b) else None

def fwd_return_sym(sym, ti, offset):
    p0 = price_at(sym, ti)
    p1 = price_at(sym, ti + offset)
    if p0 and p1 and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None

def spy_fwd(ref_date, offset):
    s  = spy.dropna()
    ti = s.index.searchsorted(pd.Timestamp(ref_date))
    if ti + offset >= len(s):
        return None
    return (float(s.iloc[ti + offset]) / float(s.iloc[ti]) - 1) * 100

def vol_at_win(sym, ti, win, min_obs=15):
    """任意窗口年化波動率"""
    if ti < win:
        return None
    seg = log_ret.iloc[ti - win: ti][sym].dropna()
    if len(seg) < min_obs:
        return None
    return float(seg.std() * np.sqrt(252))


# ── 特徵提取（單樣本）────────────────────────────────────────────────
def extract_features(sym, ti):
    """
    從 ti 時間點回望，提取 13 個特徵。
    若關鍵資料缺失回傳 None。
    """
    p0 = price_at(sym, ti)
    if p0 is None:
        return None

    # 動能
    def mom_n(n):
        p = price_at(sym, ti - n)
        if p is None or p <= 0:
            return None
        return (p0 / p - 1) * 100

    m5   = mom_n(5)
    m21  = mom_n(21)
    m63  = mom_n(63)
    m126 = mom_n(126)
    m252 = mom_n(252)
    if any(x is None for x in [m21, m252]):   # 最重要的兩個不能缺
        return None

    # 波動率
    v21 = vol_at_win(sym, ti, 21)
    v63 = vol_at_win(sym, ti, 63)

    # 52W 高點接近度
    n52_val = near52_df.iloc[ti].get(sym)
    near52  = float(n52_val) if n52_val is not None and pd.notna(n52_val) else None

    # 距 MA
    ma50_val  = ma50.iloc[ti].get(sym)
    ma200_val = ma200.iloc[ti].get(sym)
    dist_ma50  = (p0 / float(ma50_val)  - 1) * 100 if ma50_val  and pd.notna(ma50_val)  else None
    dist_ma200 = (p0 / float(ma200_val) - 1) * 100 if ma200_val and pd.notna(ma200_val) else None

    # 40 日高低反彈
    bounce_val   = bounce_pct_df.iloc[ti].get(sym)
    fromhigh_val = from_high_df.iloc[ti].get(sym)
    bounce     = float(bounce_val)   if bounce_val   is not None and pd.notna(bounce_val)   else None
    from_high  = float(fromhigh_val) if fromhigh_val is not None and pd.notna(fromhigh_val) else None

    # 市場環境（廣度）
    breadth = get_breadth(ti)

    return {
        "mom_5d":       m5,
        "mom_21d":      m21,
        "mom_63d":      m63,
        "mom_126d":     m126,
        "mom_252d":     m252,
        "vol_21d":      v21,
        "vol_63d":      v63,
        "near52":       near52,
        "dist_ma50":    dist_ma50,
        "dist_ma200":   dist_ma200,
        "bounce_pct":   bounce,
        "from_high_pct":from_high,
        "breadth":      breadth,
    }

FEATURE_COLS = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct", "breadth",
]


# ═══════════════════════════════════════════════════════════════════════
# 1. 建立特徵矩陣
# ═══════════════════════════════════════════════════════════════════════
if os.path.exists(FEATURE_CACHE):
    print(f"讀取特徵快取：{FEATURE_CACHE}")
    df_feat = pd.read_pickle(FEATURE_CACHE)
    print(f"快取：{len(df_feat)} 個樣本 × {len(FEATURE_COLS)} 個特徵\n")
else:
    rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
    symbols = list(prices.columns)
    records = []
    total_months = len(rebalance_dates)

    print(f"建立特徵矩陣（{len(symbols)} 支 × {total_months} 個月）...")
    for mi, ref in enumerate(rebalance_dates):
        ti = tidx(str(ref.date()))
        if ti < MOM_LONG + 40:
            continue

        spy_ret = spy_fwd(ref.date(), FWD_OFFSET)

        for sym in symbols:
            feats = extract_features(sym, ti)
            if feats is None:
                continue

            fwd_ret = fwd_return_sym(sym, ti, FWD_OFFSET)
            if fwd_ret is None or spy_ret is None:
                continue

            alpha = fwd_ret - spy_ret
            label = 1 if alpha > 0 else 0

            row = {"symbol": sym, "date": str(ref.date()), "fwd_alpha": alpha, "y": label}
            row.update(feats)
            records.append(row)

        if (mi + 1) % 10 == 0:
            print(f"  {mi+1}/{total_months} 個月完成，累計 {len(records)} 樣本...")

    df_feat = pd.DataFrame(records)
    df_feat.to_pickle(FEATURE_CACHE)
    print(f"已快取至 {FEATURE_CACHE}\n")


# ═══════════════════════════════════════════════════════════════════════
# 2. Train/Test 切分（按股票）
# ═══════════════════════════════════════════════════════════════════════
all_symbols = df_feat["symbol"].unique()
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_symbols)
n_train = int(len(all_symbols) * TRAIN_FRAC)
train_syms = set(all_symbols[:n_train])
test_syms  = set(all_symbols[n_train:])

df_train = df_feat[df_feat["symbol"].isin(train_syms)].copy()
df_test  = df_feat[df_feat["symbol"].isin(test_syms)].copy()

print("=== ML 選股實驗  S&P500 2021-2025 ===\n")
print(f"特徵矩陣：{len(df_feat):,} 個樣本 × {len(FEATURE_COLS)} 個特徵")
print(f"Train：{len(df_train):,} 樣本（{len(train_syms)} 支股票）")
print(f"Test ：{len(df_test):,} 樣本（{len(test_syms)} 支股票）")
print(f"標籤分布（全）：打敗大盤 {df_feat['y'].mean()*100:.1f}%\n")


# ═══════════════════════════════════════════════════════════════════════
# 3. 分群探索（Train set）
# ═══════════════════════════════════════════════════════════════════════
print("[分群探索（Train set）]")

X_train_raw = df_train[FEATURE_COLS].copy()
y_train     = df_train["y"].values

# 用中位數填充缺失（部分樣本個別特徵可能缺失）
col_medians = X_train_raw.median()
X_train_filled = X_train_raw.fillna(col_medians)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_filled)

# K-means
for k in [3, 4]:
    km    = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_train)
    parts = []
    for c in range(k):
        mask = labels == c
        n    = mask.sum()
        avg_alpha = df_train["fwd_alpha"].values[mask].mean()
        parts.append(f"Cluster {c} N={n} alpha {avg_alpha:+.1f}%")
    print(f"  K={k}：" + " / ".join(parts))

# PCA 2D（只印 variance 比例）
pca  = PCA(n_components=2, random_state=RANDOM_STATE)
pca.fit(X_train)
var_ratio = pca.explained_variance_ratio_
print(f"  PCA 2D 解釋變異：{var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}% = {sum(var_ratio)*100:.1f}%\n")


# ═══════════════════════════════════════════════════════════════════════
# 4. 監督式分類（Train → Test）
# ═══════════════════════════════════════════════════════════════════════
print("[分類預測（Test set）]")

X_test_raw    = df_test[FEATURE_COLS].copy()
y_test        = df_test["y"].values
X_test_filled = X_test_raw.fillna(col_medians)
X_test        = scaler.transform(X_test_filled)

# 4a. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.1)
lr.fit(X_train, y_train)
lr_prob_test = lr.predict_proba(X_test)[:, 1]
lr_auroc     = roc_auc_score(y_test, lr_prob_test)
print(f"  Logistic Regression AUROC：{lr_auroc:.4f}")

# 4b. XGBoost
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_prob_test = xgb.predict_proba(X_test)[:, 1]
    xgb_auroc     = roc_auc_score(y_test, xgb_prob_test)
    print(f"  XGBoost             AUROC：{xgb_auroc:.4f}  （0.50 = 隨機）")
    best_model      = xgb
    best_model_prob = xgb_prob_test
    best_label      = "XGBoost"
else:
    best_model      = lr
    best_model_prob = lr_prob_test
    best_label      = "LR"

print()


# ═══════════════════════════════════════════════════════════════════════
# 5. 策略回測（全部股票，月度）
#    - A：純動能 top5
#    - E：ML top5（用全資料重訓後預測，避免 train-test 資料洩漏）
# ═══════════════════════════════════════════════════════════════════════
print("[策略回測（全部股票池，月度）]")

# 先用全 train 資料重訓最終模型
X_all_raw    = df_feat[FEATURE_COLS].copy()
col_medians_all = X_all_raw.median()
X_all_filled = X_all_raw.fillna(col_medians_all)
X_all        = scaler.fit_transform(X_all_filled)   # 重新 fit scaler

if HAS_XGB:
    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_model.fit(X_all, df_feat["y"].values)
else:
    final_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.1)
    final_model.fit(X_all, df_feat["y"].values)

# 預先將機率附回 df_feat
df_feat["ml_prob"] = final_model.predict_proba(X_all)[:, 1]

rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
results_A    = []
results_E    = []
results_F    = []
win_E_over_A = 0
win_F_over_A = 0
total_months = 0

for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG + 40:
        continue

    spy_ret = spy_fwd(ref.date(), FWD_OFFSET)
    if spy_ret is None:
        continue

    # 取當月所有樣本
    month_df = df_feat[df_feat["date"] == str(ref.date())].copy()
    if len(month_df) < TOP_N:
        continue

    # ── 策略 A：純動能 top5 ──
    # 動態計算當月動能（利用 df_feat 的 mom_21d + mom_252d 等效替代 mom_mixed）
    month_df = month_df.copy()
    month_df["mom_mixed"] = (
        month_df["mom_21d"].fillna(0) * 0.5 +
        month_df["mom_252d"].fillna(0) * 0.5
    )
    top_A = month_df.nlargest(TOP_N, "mom_mixed")["symbol"].tolist()

    def avg_alpha(syms):
        vals = []
        for sym in syms:
            fwd = fwd_return_sym(sym, ti, FWD_OFFSET)
            if fwd is not None:
                vals.append(fwd - spy_ret)
        return float(np.median(vals)) if vals else None

    alpha_A = avg_alpha(top_A)

    # ── 策略 E：ML top5（從 top150 動能池中取 ML 機率最高 5 支）──
    top150 = month_df.nlargest(TOP_POOL, "mom_mixed")
    top_E  = top150.nlargest(TOP_N, "ml_prob")["symbol"].tolist()
    alpha_E = avg_alpha(top_E)

    # ── 策略 F：純 ML top5（全市場不限動能，直接取 ML 最高 5 支）──
    top_F   = month_df.nlargest(TOP_N, "ml_prob")["symbol"].tolist()
    alpha_F = avg_alpha(top_F)

    if alpha_A is not None and alpha_E is not None and alpha_F is not None:
        results_A.append(alpha_A)
        results_E.append(alpha_E)
        results_F.append(alpha_F)
        if alpha_E > alpha_A:
            win_E_over_A += 1
        if alpha_F > alpha_A:
            win_F_over_A += 1
        total_months += 1

med_A = float(np.median(results_A)) if results_A else 0
med_E = float(np.median(results_E)) if results_E else 0
med_F = float(np.median(results_F)) if results_F else 0

print(f"  A. 純動能 top5              1M alpha 中位 {med_A:+.2f}%")
print(f"  E. ML top5（動能前150篩選）  1M alpha 中位 {med_E:+.2f}%  勝A: {win_E_over_A}/{total_months}月")
print(f"  F. 純ML top5（全市場不限）   1M alpha 中位 {med_F:+.2f}%  勝A: {win_F_over_A}/{total_months}月")
print()


# ═══════════════════════════════════════════════════════════════════════
# 6. 特徵重要性
# ═══════════════════════════════════════════════════════════════════════
print("[特徵重要性]")

if HAS_XGB:
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        print(f"  {rank:2d}. {feat:<15s} {imp:+.4f}")
else:
    # Logistic Regression：用係數絕對值
    coefs = np.abs(final_model.coef_[0])
    feat_imp = sorted(zip(FEATURE_COLS, coefs), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        print(f"  {rank:2d}. {feat:<15s} {imp:+.4f}")

print()

# ── 誠實評估 ──────────────────────────────────────────────────────────
print("=" * 60)
print("[限制與評估]")
print("  • Survivorship bias：快取只有現有 S&P500 成份股")
print("  • 按股票切分：同一支股票所有月份都在同側，")
print("    避免了相鄰月份 autocorrelation 洩漏到 test set")
print("  • 回測模型用全部資料重訓（含 test 股票），僅供方向參考")
if lr_auroc < 0.52:
    print(f"\n  ⚠️  AUROC {lr_auroc:.3f} ≈ 0.50：特徵無顯著預測力，ML 方向待審查")
elif lr_auroc > 0.65:
    print(f"\n  ⚠️  AUROC {lr_auroc:.3f} > 0.65：請審查是否有前瞻偏誤")
else:
    print(f"\n  ℹ️  AUROC {lr_auroc:.3f}：微弱訊號，符合弱式效率市場預期")
print("=" * 60)
