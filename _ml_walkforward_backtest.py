"""
Walk-forward ML 選股回測（嚴謹版）

設計原則：
  - 每個預測月 T，只用 T 之前的資料訓練模型（無未來資訊洩漏）
  - Out-of-sample 期間：2023-01 ~ 2025-09（訓練期至少 2 年）
  - 對比三策略：A 純動能、E ML+動能篩選、F 純ML全市場
  - 同步標注 Survivorship bias 限制

已知不可消除的偏誤：
  - Survivorship bias：快取僅含現存 S&P500，已退市爛股缺席
    → 所有策略報酬均偏高，但三策略間的「相對差距」仍有參考價值

使用：
    conda run -n qt_env python _ml_walkforward_backtest.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.linear_model import LogisticRegression
    print("提示：xgboost 未安裝，改用 Logistic Regression")

# ── 參數 ─────────────────────────────────────────────────────────────────────
FEATURE_CACHE   = "data/_ml_features.pkl"
PRICE_CACHE     = "data/_protection_bt_prices.pkl"
TOP_N           = 5       # 每月選幾支
TOP_POOL        = 150     # 策略 E 的動能前N篩選池
FWD_OFFSETS     = {"1M": 21, "3M": 63, "6M": 126}  # 多窗口比較
TRAIN_START     = "2021-01-01"   # 訓練資料起點
OOS_START       = "2023-01-01"   # Out-of-sample 起點（給模型 2 年訓練期）
OOS_END         = "2025-09-01"
RETRAIN_FREQ    = 1       # 每幾個月重訓一次（1=每月，3=每季，節省時間）

FEATURE_COLS = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct", "breadth",
]

# ── 讀取特徵快取 ──────────────────────────────────────────────────────────────
if not os.path.exists(FEATURE_CACHE):
    print(f"找不到特徵快取 {FEATURE_CACHE}，請先執行 _ml_stock_selector.py")
    raise SystemExit(1)

print("讀取特徵快取...")
df_feat = pd.read_pickle(FEATURE_CACHE)
df_feat["date"] = pd.to_datetime(df_feat["date"])
df_feat = df_feat.sort_values("date").reset_index(drop=True)
print(f"  {len(df_feat):,} 筆樣本  /  {df_feat['date'].nunique()} 個月  /  {df_feat['symbol'].nunique()} 支標的")

# ── 讀取價格（計算前向報酬）────────────────────────────────────────────────────
print("讀取價格快取...")
prices = pd.read_pickle(PRICE_CACHE)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()

spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)

def tidx(date_str):
    td = pd.Timestamp(date_str)
    arr = np.searchsorted(trading_days, td)
    return min(arr, len(trading_days) - 1)

def fwd_return_sym(sym, ti, offset):
    if sym not in prices.columns:
        return None
    t0 = ti
    t1 = min(ti + offset, len(trading_days) - 1)
    p0 = prices[sym].iloc[t0]
    p1 = prices[sym].iloc[t1]
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    return float(p1 / p0 - 1) * 100

def spy_fwd(date, offset):
    td = pd.Timestamp(date)
    sub = spy[spy.index >= td]
    if len(sub) <= offset:
        return None
    return float(sub.iloc[offset] / sub.iloc[0] - 1) * 100

# ── Walk-forward 主迴圈 ───────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range(OOS_START, OOS_END, freq="BMS")

results = {label: {"A": [], "E": [], "F": [], "Q": []} for label in FWD_OFFSETS}
model_cache   = {}   # binary 模型快取（每 RETRAIN_FREQ 個月重訓）
model_cache_q = {}   # quintile 模型快取

print(f"\nWalk-forward 回測：{OOS_START} ~ {OOS_END}（{len(rebalance_dates)} 個月）")
print(f"訓練策略：每 {RETRAIN_FREQ} 個月重訓一次\n")

for i, ref in enumerate(rebalance_dates):
    ref_str = str(ref.date())
    ti = tidx(ref_str)

    # 確認最短窗口有資料（若沒有則跳過）
    if spy_fwd(ref.date(), max(FWD_OFFSETS.values())) is None:
        continue

    # ── 取當月所有特徵樣本 ─────────────────────────────────────────────
    month_df = df_feat[df_feat["date"] == ref].copy()
    if len(month_df) < TOP_N * 2:
        continue

    month_df["mom_mixed"] = (
        month_df["mom_21d"].fillna(0) * 0.5 +
        month_df["mom_252d"].fillna(0) * 0.5
    )

    # ── Walk-forward 訓練（只用 ref 之前的資料）─────────────────────────
    cache_key = i // RETRAIN_FREQ
    if cache_key not in model_cache:
        train_df = df_feat[df_feat["date"] < ref].copy()
        if len(train_df) < 500:   # 訓練樣本太少，跳過
            continue

        X_train = train_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].median())
        y_train = train_df["y"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        if HAS_XGB:
            model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
        else:
            model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)

        model.fit(X_scaled, y_train)
        col_medians = train_df[FEATURE_COLS].median()
        model_cache[cache_key] = (model, scaler, col_medians)

        # ── Quintile 模型：top/bottom 20% alpha 視為 1/0，中間 60% 丟棄 ──
        n_q = 0
        if "fwd_alpha" in train_df.columns:
            q_top = train_df.groupby("date")["fwd_alpha"].transform(lambda x: x.quantile(0.8))
            q_bot = train_df.groupby("date")["fwd_alpha"].transform(lambda x: x.quantile(0.2))
            train_df["y_q"] = np.where(
                train_df["fwd_alpha"] >= q_top, 1,
                np.where(train_df["fwd_alpha"] <= q_bot, 0, np.nan)
            )
            train_q = train_df[train_df["y_q"].notna()].copy()
            X_train_q = train_q[FEATURE_COLS].fillna(train_q[FEATURE_COLS].median())
            y_train_q = train_q["y_q"].values
            scaler_q = StandardScaler()
            X_scaled_q = scaler_q.fit_transform(X_train_q)
            if HAS_XGB:
                model_q = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42, verbosity=0,
                )
            else:
                model_q = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            model_q.fit(X_scaled_q, y_train_q)
            col_medians_q = train_q[FEATURE_COLS].median()
            model_cache_q[cache_key] = (model_q, scaler_q, col_medians_q)
            n_q = len(train_q)

        n_train = len(train_df)
        print(f"  [{ref_str}] 重訓：{n_train:,} 樣本（Q: {n_q} 樣本）", flush=True)

    model, scaler, col_medians = model_cache[cache_key]

    # 對當月所有股票預測 ML%（binary）
    X_month = month_df[FEATURE_COLS].fillna(col_medians)
    X_month_scaled = scaler.transform(X_month)
    month_df = month_df.copy()
    month_df["ml_prob"] = model.predict_proba(X_month_scaled)[:, 1]

    # 對當月所有股票預測 ML%（quintile）
    has_q = cache_key in model_cache_q
    if has_q:
        model_q, scaler_q, col_medians_q = model_cache_q[cache_key]
        X_month_q = month_df[FEATURE_COLS].fillna(col_medians_q)
        X_month_scaled_q = scaler_q.transform(X_month_q)
        month_df["ml_prob_q"] = model_q.predict_proba(X_month_scaled_q)[:, 1]

    # 股票選法（只算一次）
    top_A = month_df.nlargest(TOP_N, "mom_mixed")["symbol"].tolist()
    top_E = month_df.nlargest(TOP_POOL, "mom_mixed").nlargest(TOP_N, "ml_prob")["symbol"].tolist()
    top_F = month_df.nlargest(TOP_N, "ml_prob")["symbol"].tolist()
    top_Q = month_df.nlargest(TOP_N, "ml_prob_q")["symbol"].tolist() if has_q else []

    # ── 對每個時間窗口計算 alpha ───────────────────────────────────────
    all_ok = True
    for win_label, offset in FWD_OFFSETS.items():
        spy_ret_w = spy_fwd(ref.date(), offset)
        if spy_ret_w is None:
            all_ok = False
            break

        def avg_alpha(syms, off=offset, spy_r=spy_ret_w):
            vals = [fwd_return_sym(s, ti, off) - spy_r
                    for s in syms
                    if fwd_return_sym(s, ti, off) is not None]
            return float(np.median(vals)) if vals else None

        aA = avg_alpha(top_A)
        aE = avg_alpha(top_E)
        aF = avg_alpha(top_F)
        aQ = avg_alpha(top_Q) if top_Q else None

        if aA is None or aE is None or aF is None:
            all_ok = False
            break

        results[win_label]["A"].append((ref_str, aA, top_A))
        results[win_label]["E"].append((ref_str, aE, top_E))
        results[win_label]["F"].append((ref_str, aF, top_F))
        if aQ is not None:
            results[win_label]["Q"].append((ref_str, aQ, top_Q))

    if not all_ok:
        # 回滾已加入的結果（保持各窗口長度一致）
        for win_label in FWD_OFFSETS:
            for key in ["A", "E", "F", "Q"]:
                if results[win_label][key] and results[win_label][key][-1][0] == ref_str:
                    results[win_label][key].pop()


# ── 輸出結果 ──────────────────────────────────────────────────────────────────
from scipy import stats

n = len(results["1M"]["A"])
print(f"\n{'='*70}")
print(f"  Walk-forward 回測結果  OOS: {OOS_START[:7]} ~ {OOS_END[:7]}  ({n} 個月)")
print(f"{'='*70}")
print(f"  ⚠️  Survivorship bias：各策略絕對報酬偏高，相對差距仍有參考價值")
print(f"{'='*70}\n")

for win_label, offset in FWD_OFFSETS.items():
    res = results[win_label]
    n_w = len(res["A"])
    print(f"  ── 前向窗口 {win_label}（{offset} 交易日）─────────────────────────────")
    alphas_A = [r[1] for r in res["A"]]
    strategies = [
        ("A", "純動能 top5              "),
        ("E", "ML+動能篩選 top5(binary) "),
        ("F", "純ML全市場 top5(binary)  "),
        ("Q", "純ML全市場 top5(quintile)"),
    ]
    for name, label in strategies:
        data = res[name]
        if not data:
            print(f"    {name}. {label}  （無資料）")
            continue
        alphas = [r[1] for r in data]
        n_this = len(alphas)
        med  = float(np.median(alphas))
        mean = float(np.mean(alphas))
        pos  = sum(1 for a in alphas if a > 0)
        if name != "A":
            # 對齊月份（Q 可能月份數略少）
            paired_A = alphas_A[:n_this]
            win_vs_A = sum(1 for a, ra in zip(alphas, paired_A) if a > ra)
            if len(alphas) == len(paired_A) and len(alphas) > 2:
                t_stat, p_val = stats.ttest_rel(alphas, paired_A)
                sig = "✅" if p_val < 0.05 else ("🟡" if p_val < 0.10 else "  ")
                extra = f"  勝A: {win_vs_A}/{n_this}月  {sig}p={p_val:.3f}"
            else:
                extra = f"  勝A: {win_vs_A}/{n_this}月"
        else:
            extra = ""
        print(f"    {name}. {label}  中位{med:+.1f}%  均值{mean:+.1f}%  正alpha {pos}/{n_this}月{extra}")
    print()
