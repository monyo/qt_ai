"""
_pyramid_tranche_exit_backtest.py

回測：金字塔第2批停損出場後，第1批應該留著還是一起出？

策略 A：第2批觸停（-10%）→ 只出第2批，第1批繼續按自己停損（-15%/追蹤-25%）
策略 B：第2批觸停（-10%）→ 兩批一起全出

使用：
    conda run -n qt_env python _pyramid_tranche_exit_backtest.py
"""

import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── 載入 OHLCV 快取 ────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}")
    print("   請先執行：conda run -n qt_env python _build_ohlcv_cache.py")
    sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv       = pd.read_pickle(OHLCV_PATH)
df_close    = ohlcv["Close"]
symbols     = list(df_close.columns)
dates       = df_close.index
n_dates     = len(dates)
print(f"  {len(symbols)} 支股票  /  {n_dates} 個交易日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 參數 ──────────────────────────────────────────────────────────────────────
REBAL_FREQ        = 21    # 月度間隔（避免事件重疊）
T1_TO_T2          = 21    # 第1批到第2批間隔（模擬一個月後加碼）
PYRAMID_MIN_GAIN  = 0.05  # 第2批入場價至少比第1批高 5%（上漲金字塔）
MOM_WINDOW        = 63    # 動能窗口（確認 T1 時動能為正）

T2_STOP           = 0.10  # tight_2 固定停損 -10%
T1_FIXED_STOP     = 0.15  # standard 固定停損 -15%
T1_TRAIL_STOP     = 0.25  # standard 追蹤停損 -25%

HOLD_SEARCH_MAX   = 126   # 最多等多少天讓 T2 觸停（找不到就跳過）
T1_HOLD_MAX       = 63    # T2 觸停後，T1 最多再持有多少天

MIN_T = MOM_WINDOW + T1_TO_T2 + 5

# ── 掃描金字塔事件 ────────────────────────────────────────────────────────────
print(f"\n掃描金字塔事件（每 {REBAL_FREQ} 天取樣，第2批至少高 {PYRAMID_MIN_GAIN*100:.0f}%）...")

results = []   # 每筆事件記錄 (ret_a, ret_b, t1_ret_after_trigger)

for sym in symbols:
    c = df_close[sym].values.astype(float)

    for t1 in range(MIN_T, n_dates - T1_TO_T2 - HOLD_SEARCH_MAX - T1_HOLD_MAX - 5, REBAL_FREQ):
        p1 = c[t1]
        if np.isnan(p1) or p1 <= 0:
            continue

        # 動能確認（T1 時正動能）
        p_mom = c[t1 - MOM_WINDOW]
        if np.isnan(p_mom) or p_mom <= 0:
            continue
        if p1 / p_mom - 1 <= 0:
            continue

        # 第2批入場
        t2 = t1 + T1_TO_T2
        if t2 >= n_dates:
            continue
        p2 = c[t2]
        if np.isnan(p2) or p2 <= 0:
            continue

        # 確認上漲金字塔（第2批比第1批貴）
        if p2 < p1 * (1 + PYRAMID_MIN_GAIN):
            continue

        # 停損價位
        t2_stop_price = p2 * (1 - T2_STOP)
        t1_fixed_price = p1 * (1 - T1_FIXED_STOP)

        # ── 從 T2 入場後，追蹤高點，找 T2 觸停日 ────────────────────────────
        high_since_t1 = max(p1, p2)
        trigger_day   = None
        trigger_price = None

        for dt in range(1, HOLD_SEARCH_MAX + 1):
            t = t2 + dt
            if t >= n_dates:
                break
            ct = c[t]
            if np.isnan(ct) or ct <= 0:
                continue
            high_since_t1 = max(high_since_t1, ct)

            if ct <= t2_stop_price:
                trigger_day   = t
                trigger_price = ct
                break

        if trigger_day is None:
            continue   # 第2批未觸停，不列入比較

        # ── Strategy B：兩批一起在 trigger_price 出場 ─────────────────────────
        ret_b_t1 = (trigger_price - p1) / p1
        ret_b_t2 = (trigger_price - p2) / p2   # ≈ -10%
        ret_b    = (ret_b_t1 + ret_b_t2) / 2   # 等權重

        # ── Strategy A：第2批出場，第1批繼續 ─────────────────────────────────
        ret_a_t2 = ret_b_t2   # 第2批一樣 -10%

        # 第1批從 trigger_day 繼續持有，直到自己停損或 T1_HOLD_MAX
        high_t1 = high_since_t1   # 繼承之前的最高點
        t1_exit_price = None

        for dt2 in range(1, T1_HOLD_MAX + 1):
            t = trigger_day + dt2
            if t >= n_dates:
                break
            ct = c[t]
            if np.isnan(ct) or ct <= 0:
                continue
            high_t1 = max(high_t1, ct)

            # 固定停損
            if ct <= t1_fixed_price:
                t1_exit_price = ct
                break
            # 追蹤停損
            if ct <= high_t1 * (1 - T1_TRAIL_STOP):
                t1_exit_price = ct
                break

        if t1_exit_price is None:
            # 到期強制出場
            t_exit = min(trigger_day + T1_HOLD_MAX, n_dates - 1)
            t1_exit_price = c[t_exit]
            if np.isnan(t1_exit_price) or t1_exit_price <= 0:
                continue

        ret_a_t1 = (t1_exit_price - p1) / p1
        ret_a    = (ret_a_t1 + ret_a_t2) / 2   # 等權重

        results.append({
            "ret_a":    ret_a,
            "ret_b":    ret_b,
            "ret_a_t1": ret_a_t1,   # 第1批繼續持有的報酬
            "ret_b_t1": ret_b_t1,   # 第1批在觸停日出場的報酬
        })

print(f"找到 {len(results)} 個金字塔觸停事件")

if len(results) == 0:
    print("❌ 無事件，請調整參數")
    sys.exit(1)

df = pd.DataFrame(results)

# ── 輸出結果 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  金字塔第2批觸停後的出場策略比較")
print(f"{'='*62}")
print(f"  條件：第2批入場價 ≥ 第1批 +5%，第2批跌破 -10% 觸停")
print(f"  第1批停損：固定 -15%  /  追蹤 -25%（最多再持有 {T1_HOLD_MAX} 天）")
print(f"  樣本數：{len(df)} 個事件\n")

ra = df["ret_a"].values
rb = df["ret_b"].values

print(f"  {'指標':<18} {'A：只出第2批':>14} {'B：全部出場':>14}  {'差異(A-B)':>10}")
print(f"  {'-'*60}")

metrics = [
    ("平均報酬",    np.nanmean(ra),             np.nanmean(rb)),
    ("中位報酬",    np.nanmedian(ra),           np.nanmedian(rb)),
    ("勝率（>0）",  np.nanmean(ra > 0),         np.nanmean(rb > 0)),
    ("P10（壞）",   np.nanpercentile(ra, 10),   np.nanpercentile(rb, 10)),
    ("P25",         np.nanpercentile(ra, 25),   np.nanpercentile(rb, 25)),
    ("P75",         np.nanpercentile(ra, 75),   np.nanpercentile(rb, 75)),
    ("P90（好）",   np.nanpercentile(ra, 90),   np.nanpercentile(rb, 90)),
]

for label, va, vb in metrics:
    pct = "%" if label != "勝率（>0）" else " "
    if label == "勝率（>0）":
        print(f"  {label:<18} {va*100:>13.1f}%  {vb*100:>13.1f}%  {(va-vb)*100:>+9.1f}%")
    else:
        print(f"  {label:<18} {va*100:>+13.2f}%  {vb*100:>+13.2f}%  {(va-vb)*100:>+9.2f}%")

# ── 第1批「繼續持有 vs 當下出場」分析 ─────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  [核心問題] 第2批觸停後，第1批繼續持有 vs 當場出場")
print(f"{'='*62}")

t1_cont = df["ret_a_t1"].values    # 第1批繼續持有
t1_exit = df["ret_b_t1"].values    # 第1批當場出場

print(f"  第2批觸停時，第1批的狀態：")
print(f"    平均盈虧：{np.nanmean(t1_exit)*100:>+.2f}%（此時出場 = Strategy B 的選擇）")
print(f"    有獲利比率：{np.nanmean(t1_exit > 0)*100:.1f}%\n")

print(f"  {'指標':<18} {'繼續持有':>14} {'當場出場':>14}  {'差異':>10}")
print(f"  {'-'*60}")

t1_metrics = [
    ("平均報酬",    np.nanmean(t1_cont),            np.nanmean(t1_exit)),
    ("中位報酬",    np.nanmedian(t1_cont),          np.nanmedian(t1_exit)),
    ("勝率（>0）",  np.nanmean(t1_cont > 0),        np.nanmean(t1_exit > 0)),
    ("P10（壞）",   np.nanpercentile(t1_cont, 10),  np.nanpercentile(t1_exit, 10)),
    ("P25",         np.nanpercentile(t1_cont, 25),  np.nanpercentile(t1_exit, 25)),
    ("P75",         np.nanpercentile(t1_cont, 75),  np.nanpercentile(t1_exit, 75)),
    ("P90（好）",   np.nanpercentile(t1_cont, 90),  np.nanpercentile(t1_exit, 90)),
]

for label, va, vb in t1_metrics:
    if label == "勝率（>0）":
        print(f"  {label:<18} {va*100:>13.1f}%  {vb*100:>13.1f}%  {(va-vb)*100:>+9.1f}%")
    else:
        print(f"  {label:<18} {va*100:>+13.2f}%  {vb*100:>+13.2f}%  {(va-vb)*100:>+9.2f}%")

print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股")
print(f"  ⚠️  不含交易成本、滑點")
