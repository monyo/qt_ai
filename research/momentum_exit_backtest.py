#!/usr/bin/env python3
"""動能退出策略回測工具（嚴謹版）

比較以下策略在持倉標的 10 年歷史中的表現：
- Buy & Hold
- Fixed-15%（固定停損：跌破進場價 -15%，冷卻後重新進場）
- Mom-0%（混合動能 < 0 出場，動能 > +2% 重新進場）
- Mom-5%（混合動能 < -5% 出場，動能 > +2% 重新進場）
- Mom-10%（混合動能 < -10% 出場，動能 > +2% 重新進場）
- Combined-0%（Fixed-15% + Mom-0%，先觸發者出場）
- Combined-10%（Fixed-15% + Mom-10%，先觸發者出場）

設計改進（相對上一版）：
1. 期間從 3Y 延長至 10Y，涵蓋 2022 熊市、2020 COVID 崩盤等多個市場週期
2. 出場後允許重新進場（使策略可持續運作，與 B&H 公平比較）
   - Fixed-15%：5 日冷卻後以當日價格重新進場
   - Mom 策略：動能回升至 > MOM_REENTRY 時重新進場
3. 加入 CAGR% 年化報酬，跨期間比較更精確
4. 計算完整 10Y 期間的資產曲線，MDD 反映整個週期最大回撤

混合動能 = 50% × 21日報酬 + 50% × 252日報酬（與現系統一致）
每 5 個交易日重算動能（避免過度頻繁換手）
動能策略前 252 天不觸發（等待足夠的歷史資料）

Usage:
    python momentum_exit_backtest.py
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import json
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

PORTFOLIO_PATH = "data/portfolio.json"
PERIOD = "10y"
MOM_RECALC_DAYS = 5    # 每幾個交易日重算動能
MOM_WARMUP_DAYS = 252  # 前 N 天不觸發動能出場（等待足夠歷史資料）
MOM_REENTRY = 0.02     # 動能出場後，動能需回升至 +2% 才重新進場（避免震盪反覆）
FIXED_COOLDOWN = 5     # 固定停損觸發後，等待 N 天再重新進場

STRATEGIES = [
    {"name": "Buy & Hold",   "fixed": None,  "mom": None},
    {"name": "Fixed-15%",    "fixed": -0.15, "mom": None},
    {"name": "Mom-0%",       "fixed": None,  "mom":  0.0},
    {"name": "Mom-5%",       "fixed": None,  "mom": -0.05},
    {"name": "Mom-10%",      "fixed": None,  "mom": -0.10},
    {"name": "Combined-0%",  "fixed": -0.15, "mom":  0.0},
    {"name": "Combined-10%", "fixed": -0.15, "mom": -0.10},
]


def load_symbols():
    """讀取持倉標的（排除 core=True）"""
    with open(PORTFOLIO_PATH) as f:
        data = json.load(f)
    positions = data.get("positions", {})
    return [sym for sym, pos in positions.items() if not pos.get("core", False)]


def fetch_data(symbol):
    """下載 10Y 日線資料，需至少 500 天"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=PERIOD)
        if df.empty or len(df) < 500:
            return None
        return df[["Close"]].copy()
    except Exception:
        return None


def calc_mixed_momentum(prices, idx):
    """計算第 idx 天的混合動能（50% 21日 + 50% 252日）"""
    price = prices[idx]
    p21 = prices[idx - 21] if idx >= 21 else prices[0]
    p252 = prices[idx - 252] if idx >= 252 else prices[0]
    return 0.5 * (price / p21 - 1) + 0.5 * (price / p252 - 1)


def simulate_strategy(df, fixed_threshold, mom_threshold):
    """模擬完整 10Y 的資產曲線（含重新進場）

    出場規則：
    - fixed_threshold: 跌破進場成本 N% 出場（如 -0.15）
    - mom_threshold:   混合動能低於門檻出場（如 -0.10）
    - 兩者同時存在時，任一觸發即出場

    重新進場規則：
    - 固定停損策略：5 日冷卻後以當日收盤重新進場
    - 動能策略：冷卻後且動能回升至 > MOM_REENTRY 時重新進場

    Returns:
        dict: Return%, CAGR%, MDD%, Trade_Count
    """
    prices = df["Close"].values
    n = len(prices)

    # B&H：始終持有，直接計算
    if fixed_threshold is None and mom_threshold is None:
        cumulative = prices / prices[0]
        peak = np.maximum.accumulate(cumulative)
        mdd = ((cumulative - peak) / peak).min() * 100
        total_return = (cumulative[-1] - 1) * 100
        years = n / 252
        cagr = (cumulative[-1] ** (1 / years) - 1) * 100 if years > 0 else 0.0
        return {
            "Return%": round(total_return, 2),
            "CAGR%": round(cagr, 2),
            "MDD%": round(mdd, 2),
            "Trade_Count": 1,
        }

    # 其他策略：逐日模擬
    in_position = True
    entry_price = prices[0]
    cooldown_counter = 0
    mom_counter = 0
    last_mom = None
    trade_count = 1

    # 用 daily_returns 追蹤每日策略報酬（0 = 持現金，daily_ret = 持股）
    daily_returns = np.zeros(n - 1)

    for i in range(1, n):
        price = prices[i]
        prev_price = prices[i - 1]
        daily_ret = (price / prev_price - 1)

        # 更新動能（warmup 過後每 MOM_RECALC_DAYS 天）
        if mom_threshold is not None and i >= MOM_WARMUP_DAYS:
            if mom_counter % MOM_RECALC_DAYS == 0:
                last_mom = calc_mixed_momentum(prices, i)
            mom_counter += 1

        # 冷卻計數
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # --- 出場判斷（若目前持倉中）---
        if in_position:
            triggered = False

            if fixed_threshold is not None:
                pnl = (price - entry_price) / entry_price
                if pnl <= fixed_threshold:
                    triggered = True

            if not triggered and mom_threshold is not None and last_mom is not None:
                if last_mom <= mom_threshold:
                    triggered = True

            if triggered:
                in_position = False
                cooldown_counter = FIXED_COOLDOWN

        # --- 重新進場判斷（若目前持現金且冷卻結束）---
        elif cooldown_counter == 0:
            can_enter = True
            # 動能策略：需動能回升才重新進場
            if mom_threshold is not None:
                if last_mom is None or last_mom < MOM_REENTRY:
                    can_enter = False

            if can_enter:
                in_position = True
                entry_price = price
                trade_count += 1

        # 記錄今日報酬
        daily_returns[i - 1] = daily_ret if in_position else 0.0

    # --- 計算績效 ---
    cumulative = np.cumprod(1 + daily_returns)
    total_return = (cumulative[-1] - 1) * 100

    years = n / 252
    cagr = (cumulative[-1] ** (1 / years) - 1) * 100 if years > 0 else 0.0

    peak = np.maximum.accumulate(cumulative)
    mdd = ((cumulative - peak) / peak).min() * 100

    return {
        "Return%": round(total_return, 2),
        "CAGR%": round(cagr, 2),
        "MDD%": round(mdd, 2),
        "Trade_Count": trade_count,
    }


def run_backtest():
    symbols = load_symbols()
    print(f"\n=== 動能退出策略回測 | 期間: {PERIOD} | 標的: {len(symbols)} 支 | {date.today()} ===")
    print(f"    設定：warmup={MOM_WARMUP_DAYS}天, 動能重算每{MOM_RECALC_DAYS}天, 重入門檻={MOM_REENTRY*100:.0f}%, 冷卻={FIXED_COOLDOWN}天\n")

    all_results = []
    succeeded = []

    for symbol in symbols:
        print(f"  分析 {symbol}...", end=" ", flush=True)
        df = fetch_data(symbol)
        if df is None:
            print("數據不足，跳過")
            continue

        for s in STRATEGIES:
            try:
                metrics = simulate_strategy(df, s["fixed"], s["mom"])
                all_results.append({
                    "Symbol": symbol,
                    "Strategy": s["name"],
                    **metrics,
                })
            except Exception as e:
                all_results.append({
                    "Symbol": symbol,
                    "Strategy": s["name"],
                    "Return%": float("nan"),
                    "CAGR%": float("nan"),
                    "MDD%": float("nan"),
                    "Trade_Count": 0,
                })

        print(f"完成（{len(df)} 天）")
        succeeded.append(symbol)

    if not all_results:
        print("\n沒有成功分析任何標的。")
        return

    df_results = pd.DataFrame(all_results)

    # === 個股詳細結果 ===
    print(f"\n{'='*90}")
    print("  個股詳細比較")
    print(f"{'='*90}")

    for symbol in succeeded:
        sym_df = df_results[df_results["Symbol"] == symbol].copy()
        print(f"\n--- {symbol} ---")
        print(sym_df[["Strategy", "Return%", "CAGR%", "MDD%", "Trade_Count"]].to_string(index=False))

    # === 策略總結（所有標的平均）===
    print(f"\n{'='*90}")
    print("  策略總結（所有標的平均）")
    print(f"{'='*90}\n")

    summary = df_results.groupby("Strategy").agg(
        avg_return=("Return%", "mean"),
        avg_cagr=("CAGR%", "mean"),
        avg_mdd=("MDD%", "mean"),
        avg_trades=("Trade_Count", "mean"),
    ).round(2)

    strategy_order = [s["name"] for s in STRATEGIES]
    summary = summary.reindex(strategy_order)

    bh_cagr = summary.loc["Buy & Hold", "avg_cagr"]
    summary["vs B&H CAGR"] = (summary["avg_cagr"] - bh_cagr).round(2)

    # 風險調整報酬（Return / |MDD|）
    summary["Calmar"] = (summary["avg_cagr"] / summary["avg_mdd"].abs().replace(0, 0.01)).round(3)

    summary.columns = ["平均Return%", "平均CAGR%", "平均MDD%", "平均交易次數", "vs B&H CAGR", "Calmar比"]
    print(summary.to_string())

    # === 結論 ===
    print(f"\n{'='*90}")
    print("  回測結論")
    print(f"{'='*90}\n")

    best_cagr = summary["平均CAGR%"].idxmax()
    best_mdd = summary["平均MDD%"].idxmax()   # MDD 負數，idxmax = 最小虧損
    best_calmar = summary["Calmar比"].idxmax()

    print(f"  最高年化報酬:   {best_cagr} (CAGR {summary.loc[best_cagr, '平均CAGR%']:+.1f}%)")
    print(f"  最小最大回撤:   {best_mdd} (MDD {summary.loc[best_mdd, '平均MDD%']:.1f}%)")
    print(f"  Calmar 最佳:   {best_calmar} (CAGR/|MDD| = {summary.loc[best_calmar, 'Calmar比']:.3f})")

    fixed_cagr = summary.loc["Fixed-15%", "平均CAGR%"]
    bh_cagr_val = summary.loc["Buy & Hold", "平均CAGR%"]

    print(f"\n  Fixed-15% vs B&H：{fixed_cagr - bh_cagr_val:+.1f}% CAGR，MDD 改善 {summary.loc['Fixed-15%', '平均MDD%'] - summary.loc['Buy & Hold', '平均MDD%']:+.1f}%")

    combined_better = [
        s for s in ["Combined-0%", "Combined-10%"]
        if summary.loc[s, "Calmar比"] > summary.loc["Fixed-15%", "Calmar比"]
    ]
    if combined_better:
        print(f"\n  ✓ Combined 策略 Calmar 優於純 Fixed-15%: {', '.join(combined_better)}")
        print("    建議：可將動能門檻納入現有系統的出場條件")
    else:
        print(f"\n  ✗ Combined 策略 Calmar 未優於純 Fixed-15%")
        print("    建議：現有固定停損已夠用，不需加入動能出場")

    # === 儲存 CSV ===
    csv_path = f"data/backtest_momentum_exit_{date.today().strftime('%Y%m%d')}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n詳細結果已儲存至: {csv_path}")


if __name__ == "__main__":
    run_backtest()
