"""
_alpha_quality_backtest.py

驗證 Alpha 品質過濾的有效性，並深入分析 B1 組的可辨識性。

分組（第一層）：
  A  : 1Y > 0 AND 3Y > 0  （雙優，主清單）
  B1 : 1Y > 0 but 3Y ≤ 0  （近期強 / 長期弱）
  B2 : 1Y ≤ 0 but 3Y > 0  （近期弱 / 長期強）
  C  : 1Y ≤ 0 AND 3Y ≤ 0  （雙差）

B1 細分（第二層，測試可辨識性）：
  B1-L : RSI < 75   → 還沒過熱，可能是真反轉
  B1-H : RSI ≥ 75   → 已過熱，可能是短線衝高後崩回
"""

import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from src.data_loader import get_sp500_tickers

# ── 參數 ──────────────────────────────────────────────────────────
UNIVERSE_SIZE   = 505   # 取 S&P500 前 N 檔
TOP_N_MOMENTUM  = 20    # 每期取動能前 N 名
SNAPSHOT_MONTHS = 24    # 回溯快照月數
FORWARD_WINDOWS = [63, 126]  # 3 個月、6 個月（交易日）
DATA_YEARS      = 5     # 下載幾年歷史（需夠長以計算 3Y alpha）
RSI_HOT         = 75    # B1 細分門檻


# ── 工具函數 ───────────────────────────────────────────────────────
def momentum_score(prices: pd.Series, idx: int) -> float:
    """混合動能：50% 21天 + 50% 252天"""
    if idx < 252 or pd.isna(prices.iloc[idx]):
        return np.nan
    r_short = prices.iloc[idx] / prices.iloc[idx - 21] - 1 if idx >= 21 else np.nan
    r_long  = prices.iloc[idx] / prices.iloc[idx - 252] - 1
    if pd.isna(r_short):
        return r_long
    return 0.5 * r_short + 0.5 * r_long


def calc_rsi(prices: pd.Series, idx: int, period: int = 14) -> float:
    """標準 Wilder RSI（14期）"""
    start = idx - period * 3   # 取多一點暖機
    if start < 0:
        return np.nan
    chunk = prices.iloc[start: idx + 1]
    delta = chunk.diff().dropna()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan


def alpha_vs_spy(stock: pd.Series, spy: pd.Series, idx: int, window: int):
    """相對 SPY 超額報酬（%），無資料回傳 None"""
    start = idx - window
    if start < 0 or idx >= len(stock):
        return None
    if pd.isna(stock.iloc[start]) or pd.isna(stock.iloc[idx]):
        return None
    s_ret   = stock.iloc[idx] / stock.iloc[start] - 1
    spy_ret = spy.iloc[idx]   / spy.iloc[start]   - 1
    return (s_ret - spy_ret) * 100


def classify(a1y, a3y, rsi) -> str:
    """
    第一層：A / B1 / B2 / C
    B1 再按 RSI 細分為 B1-L / B1-H
    """
    ok_1y = (a1y is None) or (a1y > 0)
    ok_3y = (a3y is None) or (a3y > 0)
    if ok_1y and ok_3y:
        return "A"
    elif ok_1y:
        if rsi is None or np.isnan(rsi):
            return "B1-L"
        return "B1-H" if rsi >= RSI_HOT else "B1-L"
    elif ok_3y:
        return "B2"
    else:
        return "C"


def print_table(sub: pd.DataFrame, groups: list, labels: dict):
    hdr = f"  {'組別':<44} {'n':>4}  {'平均報酬':>8}  {'中位數':>7}  {'超額':>7}  {'勝率':>6}  {'打敗SPY':>7}"
    print(hdr)
    print(f"  {'-'*67}")
    for grp in groups:
        g = sub[sub["group"] == grp]
        if len(g) == 0:
            continue
        n        = len(g)
        mean_ret = g["fwd_ret"].mean()
        med_ret  = g["fwd_ret"].median()
        mean_exc = g["fwd_excess"].mean()
        win_rate = (g["fwd_ret"] > 0).mean() * 100
        beat_spy = (g["fwd_excess"] > 0).mean() * 100
        tag      = labels[grp]
        print(f"  [{grp}] {tag:<42} {n:>4}  {mean_ret:>+7.1f}%  {med_ret:>+6.1f}%"
              f"  {mean_exc:>+6.1f}%  {win_rate:>5.0f}%  {beat_spy:>6.0f}%")


# ── 主程式 ────────────────────────────────────────────────────────
def main():
    print("=== Alpha 品質 + RSI 細分回測 ===")
    fwd_labels = " / ".join(f"{f//21}個月" for f in FORWARD_WINDOWS)
    print(f"宇宙: S&P500 前 {UNIVERSE_SIZE} 檔 | 動能前 {TOP_N_MOMENTUM} 名 | "
          f"快照 {SNAPSHOT_MONTHS} 個月 | 前向 {fwd_labels} | B1 RSI 門檻 {RSI_HOT}\n")

    labels = {
        "A"   : "雙優    1Y>0 AND 3Y>0",
        "B1-L": f"B1-冷   1Y>0, 3Y≤0, RSI<{RSI_HOT}  ← 潛在反轉",
        "B1-H": f"B1-熱   1Y>0, 3Y≤0, RSI≥{RSI_HOT}  ← 過熱疑慮",
        "B2"  : "B2混合  1Y≤0, 3Y>0",
        "C"   : "雙差    1Y≤0 AND 3Y≤0",
    }
    all_groups = ["A", "B1-L", "B1-H", "B2", "C"]

    # 1. 取標的
    sp500    = get_sp500_tickers()[:UNIVERSE_SIZE]
    universe = list(dict.fromkeys(["SPY"] + sp500))

    # 2. 下載歷史資料
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=365 * DATA_YEARS)
    print(f"下載 {len(universe)} 檔資料（{start_dt} ~ {end_dt}）...")
    raw = yf.download(universe, start=str(start_dt), end=str(end_dt),
                      auto_adjust=True, progress=False)
    prices: pd.DataFrame = raw["Close"].dropna(axis=1, how="all")
    print(f"成功取得 {len(prices.columns)} 檔，{len(prices)} 個交易日\n")

    spy    = prices["SPY"]
    stocks = [c for c in prices.columns if c != "SPY"]

    # 3. 逐月快照
    today_idx = len(prices) - 1
    records   = []

    for m in range(SNAPSHOT_MONTHS, 0, -1):
        snap_idx  = today_idx - m * 21
        if snap_idx < 756:
            continue
        snap_date = prices.index[snap_idx].date()

        # 動能篩選
        scored = []
        for sym in stocks:
            mom = momentum_score(prices[sym], snap_idx)
            if pd.isna(mom) or mom <= 0:
                continue
            scored.append((sym, mom))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:TOP_N_MOMENTUM]

        for sym, mom in top:
            a1y = alpha_vs_spy(prices[sym], spy, snap_idx, 252)
            a3y = alpha_vs_spy(prices[sym], spy, snap_idx, 756)
            rsi = calc_rsi(prices[sym], snap_idx)
            grp = classify(a1y, a3y, rsi)

            for fwd in FORWARD_WINDOWS:
                fwd_idx = snap_idx + fwd
                if fwd_idx >= len(prices):
                    continue
                s0, s1     = prices[sym].iloc[snap_idx], prices[sym].iloc[fwd_idx]
                spy0, spy1 = spy.iloc[snap_idx], spy.iloc[fwd_idx]
                if pd.isna(s0) or pd.isna(s1):
                    continue
                fwd_ret    = (s1 / s0 - 1) * 100
                fwd_excess = fwd_ret - (spy1 / spy0 - 1) * 100

                records.append({
                    "date"      : snap_date,
                    "symbol"    : sym,
                    "momentum"  : mom * 100,
                    "alpha_1y"  : a1y,
                    "alpha_3y"  : a3y,
                    "rsi"       : rsi,
                    "group"     : grp,
                    "fwd_days"  : fwd,
                    "fwd_ret"   : fwd_ret,
                    "fwd_excess": fwd_excess,
                })

    if not records:
        print("資料不足，無法分析")
        return

    df = pd.DataFrame(records)

    # ── 4a. 第一層：A / B1整體 / B2 / C ─────────────────────────────
    df["group_l1"] = df["group"].replace({"B1-L": "B1", "B1-H": "B1"})
    labels_l1 = {
        "A" : "雙優    1Y>0 AND 3Y>0",
        "B1": "B1混合  1Y>0, 3Y≤0  （全體）",
        "B2": "B2混合  1Y≤0, 3Y>0",
        "C" : "雙差    1Y≤0 AND 3Y≤0",
    }

    for fwd in FORWARD_WINDOWS:
        months = fwd // 21
        print(f"{'='*70}")
        print(f"  【第一層】前向 {months} 個月（{fwd} 交易日）")
        print(f"{'='*70}")
        sub = df[df["fwd_days"] == fwd].copy()
        sub["group"] = sub["group_l1"]
        print_table(sub, ["A", "B1", "B2", "C"], labels_l1)
        print()

    # ── 4b. 第二層：B1 按 RSI 細分 ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  【第二層】B1 細分：RSI < {RSI_HOT}（冷） vs RSI ≥ {RSI_HOT}（熱）")
    print(f"{'='*70}")

    b1_labels = {
        "B1-L": labels["B1-L"],
        "B1-H": labels["B1-H"],
    }
    for fwd in FORWARD_WINDOWS:
        months = fwd // 21
        print(f"\n  -- 前向 {months} 個月 --")
        sub = df[df["fwd_days"] == fwd]
        print_table(sub, ["B1-L", "B1-H"], b1_labels)

    # ── 4c. 對照：A vs B1-L（潛在反轉）─────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  【對照】A（雙優）vs B1-冷（潛在反轉）")
    print(f"{'='*70}")
    compare_labels = {
        "A"   : labels["A"],
        "B1-L": labels["B1-L"],
    }
    for fwd in FORWARD_WINDOWS:
        months = fwd // 21
        print(f"\n  -- 前向 {months} 個月 --")
        sub = df[df["fwd_days"] == fwd]
        print_table(sub, ["A", "B1-L"], compare_labels)

    # ── 4d. 分佈分析：如果你總是踩到爛 B1 ──────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  【壞手氣分析】移除 APP 後，B1 剩餘 vs A 的完整分佈")
    print(f"  （模擬：你完全錯過寶藏，只拿到普通 B1）")
    print(f"{'='*70}")

    for fwd in FORWARD_WINDOWS:
        months = fwd // 21
        sub    = df[df["fwd_days"] == fwd]
        a_exc  = sub[sub["group_l1"] == "A"]["fwd_excess"]
        b1_all = sub[sub["group_l1"] == "B1"]["fwd_excess"]
        b1_noapp = sub[(sub["group_l1"] == "B1") & (sub["symbol"] != "APP")]["fwd_excess"]

        print(f"\n  -- 前向 {months} 個月超額報酬分佈 --")
        print(f"  {'':30s}  {'A':>7}  {'B1全體':>7}  {'B1去除APP':>10}")
        print(f"  {'-'*58}")

        def fmt(s, pct):
            return f"{np.percentile(s, pct):>+6.1f}%"

        rows_dist = [
            ("最慘（最小值）",   0),
            ("倒楣 10%（p10）", 10),
            ("倒楣 25%（p25）", 25),
            ("中位數（p50）",   50),
            ("幸運 75%（p75）", 75),
            ("幸運 90%（p90）", 90),
            ("最好（最大值）",  100),
        ]
        for label_d, pct in rows_dist:
            v_a   = fmt(a_exc,     pct) if pct < 100 else f"{a_exc.max():>+6.1f}%"
            v_b1  = fmt(b1_all,    pct) if pct < 100 else f"{b1_all.max():>+6.1f}%"
            v_b1n = fmt(b1_noapp,  pct) if pct < 100 else f"{b1_noapp.max():>+6.1f}%"
            if pct == 0:
                v_a   = f"{a_exc.min():>+6.1f}%"
                v_b1  = f"{b1_all.min():>+6.1f}%"
                v_b1n = f"{b1_noapp.min():>+6.1f}%"
            print(f"  {label_d:<30}  {v_a:>7}  {v_b1:>7}  {v_b1n:>10}")

        print()
        for label_d, threshold in [("虧損機率（超額 < 0）", 0),
                                    ("大虧機率（超額 < -10%）", -10),
                                    ("血崩機率（超額 < -20%）", -20)]:
            pa   = (a_exc   < threshold).mean() * 100
            pb1  = (b1_all  < threshold).mean() * 100
            pb1n = (b1_noapp < threshold).mean() * 100
            print(f"  {label_d:<30}  {pa:>6.0f}%  {pb1:>6.0f}%  {pb1n:>9.0f}%")

    # ── 5. B1 各標的出現次數與平均表現（找規律）────────────────────
    print(f"\n\n{'='*70}")
    print(f"  【B1 標的明細】各股在 B1 組出現次數 & 平均 6M 超額報酬")
    print(f"{'='*70}")
    b1_detail = (
        df[(df["group"].isin(["B1-L", "B1-H"])) & (df["fwd_days"] == 126)]
        .groupby("symbol")
        .agg(
            count    = ("fwd_ret", "count"),
            avg_exc  = ("fwd_excess", "mean"),
            med_exc  = ("fwd_excess", "median"),
            avg_rsi  = ("rsi", "mean"),
            avg_a3y  = ("alpha_3y", "mean"),
        )
        .sort_values("avg_exc", ascending=False)
    )
    print(f"  {'標的':<8} {'出現':>4}  {'平均超額':>8}  {'中位超額':>8}  {'平均RSI':>8}  {'平均3Y alpha':>12}")
    print(f"  {'-'*55}")
    for sym, row in b1_detail.iterrows():
        a3y_str = f"{row['avg_a3y']:+.0f}%" if pd.notna(row['avg_a3y']) else "  N/A"
        print(f"  {sym:<8} {row['count']:>4}  {row['avg_exc']:>+7.1f}%  "
              f"{row['med_exc']:>+7.1f}%  {row['avg_rsi']:>7.1f}  {a3y_str:>12}")


if __name__ == "__main__":
    main()
