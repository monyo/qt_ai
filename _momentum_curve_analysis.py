"""
動能曲線形態分析

探討動能股的價格曲線形態：
  拋物線型（凸曲線）：漲幅逐漸趨緩，頂部有圓弧過渡
  飆股型（凹曲線）：加速度持續為正，失去動能後驟降

核心指標：
  return_21d   = price[t]/price[t-21] - 1       （一階導數：斜率）
  acceleration = return_21d[t] - return_21d[t-21]（二階導數：曲率）
  正值 = 動能加速（飆股型）；負值 = 動能減速（見頂型）

用法：
  python _momentum_curve_analysis.py --stock GLW
  python _momentum_curve_analysis.py --compare GLW APP NVDA PLTR
  python _momentum_curve_analysis.py --portfolio
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_loader import fetch_stock_data
from src.portfolio import load_portfolio


# ── 指標計算 ──────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """輸入含 Close 欄位的 DataFrame，回傳加上指標欄位的副本"""
    close = df["Close"].copy()
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["log_close"] = np.log(close)
    out["ma60"] = close.rolling(60).mean()
    out["ma200"] = close.rolling(200).mean()

    # 一階導數：21日報酬率（%）
    out["return_21d"] = (close / close.shift(21) - 1) * 100

    # 一階導數：252日報酬率（%）
    out["return_252d"] = (close / close.shift(252) - 1) * 100

    # 混合動能（系統公式）
    out["mom_mixed"] = 0.5 * out["return_21d"] + 0.5 * out["return_252d"]

    # 二階導數：加速度（%pts / 21日）
    out["acceleration"] = out["return_21d"] - out["return_21d"].shift(21)

    return out


def load_data(symbol: str, period: str = "3y") -> pd.DataFrame | None:
    """下載/讀取快取，回傳加了指標的 DataFrame；失敗回傳 None"""
    df = fetch_stock_data(symbol, period=period)
    if df is None or df.empty or "Close" not in df.columns:
        print(f"  ✗ 無法取得 {symbol} 資料")
        return None
    # tz-naive
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx.tz_convert(None)
    return compute_indicators(df)


# ── 功能一：單股四格圖 ─────────────────────────────────────────────
def plot_single(symbol: str):
    print(f"分析 {symbol}...")
    ind = load_data(symbol)
    if ind is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"{symbol}  Momentum Curve Analysis", fontsize=14, fontweight="bold")

    # Panel 1: Log price + MA60 + MA200
    ax = axes[0]
    ax.plot(ind.index, ind["log_close"], color="#1f77b4", linewidth=1.5, label="Log Price")
    ax.plot(ind.index, np.log(ind["ma60"]), color="orange", linewidth=1, linestyle="--", label="MA60")
    ax.plot(ind.index, np.log(ind["ma200"]), color="red", linewidth=1, linestyle="--", label="MA200")
    ax.set_ylabel("Log Price")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: 21d return (1st derivative)
    ax = axes[1]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(ind.index, ind["return_21d"], 0,
                    where=ind["return_21d"] >= 0, color="green", alpha=0.3)
    ax.fill_between(ind.index, ind["return_21d"], 0,
                    where=ind["return_21d"] < 0, color="red", alpha=0.3)
    ax.plot(ind.index, ind["return_21d"], color="#2ca02c", linewidth=1)
    ax.set_ylabel("21d Return (%)")
    ax.grid(True, alpha=0.3)

    # Panel 3: Acceleration (2nd derivative) — key panel
    ax = axes[2]
    ax.axhline(0, color="black", linewidth=0.8)
    accel = ind["acceleration"]
    ax.fill_between(ind.index, accel, 0,
                    where=accel >= 0, color="green", alpha=0.4, label="Accelerating (rocket)")
    ax.fill_between(ind.index, accel, 0,
                    where=accel < 0, color="red", alpha=0.4, label="Decelerating (topping)")
    ax.plot(ind.index, accel, color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Momentum Accel (%pts)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 4: Mixed momentum score
    ax = axes[3]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(ind.index, ind["mom_mixed"], 0,
                    where=ind["mom_mixed"] >= 0, color="#1f77b4", alpha=0.3)
    ax.fill_between(ind.index, ind["mom_mixed"], 0,
                    where=ind["mom_mixed"] < 0, color="red", alpha=0.3)
    ax.plot(ind.index, ind["mom_mixed"], color="#1f77b4", linewidth=1)
    ax.set_ylabel("Mixed Momentum (%)")
    ax.grid(True, alpha=0.3)

    # X 軸格式
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out_path = f"data/curve_analysis_{symbol}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 儲存至 {out_path}")

    # 印出最近加速度數值摘要
    recent = ind["acceleration"].dropna().tail(20)
    last = float(ind["acceleration"].dropna().iloc[-1])
    sign = "加速 ▲" if last >= 0 else "減速 ▼"
    last_return = float(ind["return_21d"].dropna().iloc[-1])
    print(f"  最近加速度: {last:+.1f}%pts  ({sign})")
    print(f"  最近 21d 報酬: {last_return:+.1f}%")
    # 近 20 日加速度正負比例
    pos_pct = (recent >= 0).sum() / len(recent) * 100
    print(f"  近 20 日加速度正值比例: {pos_pct:.0f}%")


# ── 功能二：多股曲率對比 ────────────────────────────────────────────
COMPARE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

def plot_compare(symbols: list[str]):
    print(f"曲率對比：{', '.join(symbols)}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Momentum Acceleration Comparison (2nd Derivative)", fontsize=14, fontweight="bold")

    all_ind = {}
    for sym in symbols:
        ind = load_data(sym)
        if ind is not None:
            all_ind[sym] = ind

    if not all_ind:
        print("No data available")
        return

    # Panel 1: Normalized price (common start date)
    ax = axes[0]
    dates_list = [ind.index for ind in all_ind.values()]
    common_start = max(d[0] for d in dates_list)
    for i, (sym, ind) in enumerate(all_ind.items()):
        sliced = ind.loc[ind.index >= common_start, "close"]
        if sliced.empty:
            continue
        norm = sliced / sliced.iloc[0] * 100  # base=100
        ax.plot(sliced.index, norm, color=COMPARE_COLORS[i % len(COMPARE_COLORS)],
                linewidth=1.5, label=sym)
    ax.axhline(100, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Relative Return (base=100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Acceleration (5-day smoothed for clarity)
    ax = axes[1]
    ax.axhline(0, color="black", linewidth=0.8)
    for i, (sym, ind) in enumerate(all_ind.items()):
        sliced = ind.loc[ind.index >= common_start, "acceleration"]
        smoothed = sliced.rolling(5, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed,
                color=COMPARE_COLORS[i % len(COMPARE_COLORS)],
                linewidth=1.5, label=sym, alpha=0.85)
    ax.set_ylabel("Momentum Accel (%pts, 5d smooth)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out_path = "data/curve_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 儲存至 {out_path}")

    # 印出各股最近狀態
    print()
    print(f"  {'股票':<8} {'最近加速度':>12} {'21d報酬':>10} {'狀態':>8}")
    print("  " + "-" * 42)
    for sym, ind in all_ind.items():
        last_acc = float(ind["acceleration"].dropna().iloc[-1])
        last_ret = float(ind["return_21d"].dropna().iloc[-1])
        status = "加速 ▲" if last_acc >= 0 else "減速 ▼"
        print(f"  {sym:<8} {last_acc:>+11.1f}%  {last_ret:>+9.1f}%  {status}")


# ── 功能三：持倉掃描 ─────────────────────────────────────────────────
def scan_portfolio():
    port = load_portfolio()
    positions = port.get("positions", {})
    if not positions:
        print("無持倉")
        return

    symbols = [sym for sym, pos in positions.items()
               if not pos.get("core", False)]

    print(f"掃描 {len(symbols)} 個持倉的動能加速度狀態...\n")

    rows = []
    for sym in sorted(symbols):
        ind = load_data(sym)
        if ind is None:
            continue
        accel_s = ind["acceleration"].dropna()
        ret21_s = ind["return_21d"].dropna()
        mom_s   = ind["mom_mixed"].dropna()
        if accel_s.empty or ret21_s.empty:
            continue
        last_acc = float(accel_s.iloc[-1])
        last_ret21 = float(ret21_s.iloc[-1])
        last_mom = float(mom_s.iloc[-1]) if not mom_s.empty else float("nan")
        # 近 10 日加速度均值（趨勢方向）
        acc10 = float(ind["acceleration"].dropna().tail(10).mean())
        rows.append({
            "symbol": sym,
            "last_acc": last_acc,
            "acc10_mean": acc10,
            "return_21d": last_ret21,
            "mom_mixed": last_mom,
        })

    if not rows:
        print("無法取得任何持倉資料")
        return

    df = pd.DataFrame(rows).sort_values("last_acc", ascending=False)

    # ── 加速中（飆股型）──
    accel_mask = df["last_acc"] >= 0
    print("=" * 60)
    print("  加速中（動能持續擴張）")
    print(f"  {'股票':<8} {'加速度':>9} {'10日均':>9} {'21d報酬':>9} {'混合動能':>10}")
    print("  " + "-" * 50)
    for _, r in df[accel_mask].iterrows():
        print(f"  {r['symbol']:<8} {r['last_acc']:>+8.1f}%  "
              f"{r['acc10_mean']:>+8.1f}%  {r['return_21d']:>+8.1f}%  "
              f"{r['mom_mixed']:>+9.1f}%")

    # ── 減速中（見頂型）──
    print()
    print("  減速中（動能開始收縮）")
    print(f"  {'股票':<8} {'加速度':>9} {'10日均':>9} {'21d報酬':>9} {'混合動能':>10}")
    print("  " + "-" * 50)
    for _, r in df[~accel_mask].iterrows():
        print(f"  {r['symbol']:<8} {r['last_acc']:>+8.1f}%  "
              f"{r['acc10_mean']:>+8.1f}%  {r['return_21d']:>+8.1f}%  "
              f"{r['mom_mixed']:>+9.1f}%")
    print("=" * 60)

    # 加速度連續負值警告（可能正進入拋物線減速階段）
    print()
    print("  ⚠️  近 10 日平均加速度 < -5%pts（拋物線減速警示）：")
    warn = df[df["acc10_mean"] < -5]
    if warn.empty:
        print("  （無）")
    else:
        for _, r in warn.iterrows():
            print(f"  {r['symbol']}  10日均加速度 {r['acc10_mean']:+.1f}%pts  "
                  f"21d報酬 {r['return_21d']:+.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="動能曲線形態分析（一/二階導數）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stock", metavar="SYM",
                       help="單股四格圖（例：--stock GLW）")
    group.add_argument("--compare", nargs="+", metavar="SYM",
                       help="多股曲率對比（例：--compare GLW APP NVDA PLTR）")
    group.add_argument("--portfolio", action="store_true",
                       help="掃描全部持倉的動能加速度狀態")
    args = parser.parse_args()

    if args.stock:
        plot_single(args.stock.upper())
    elif args.compare:
        plot_compare([s.upper() for s in args.compare])
    elif args.portfolio:
        scan_portfolio()


if __name__ == "__main__":
    main()
