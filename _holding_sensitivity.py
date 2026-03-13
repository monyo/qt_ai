"""ROTATE 保護期長短敏感度測試（臨時腳本）"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import portfolio_backtest as pb
from portfolio_backtest import (
    load_portfolio_symbols, fetch_universe_prices, fetch_sector_map,
    build_aligned_prices, fetch_spy_regime, PortfolioSimulator, StrategyConfig,
    get_sp500_tickers, TOP_N_SP500, calc_bnh_metrics,
)

print("載入快取資料...")
sp500 = get_sp500_tickers()[:TOP_N_SP500]
portfolio_syms = load_portfolio_symbols()
universe = list(dict.fromkeys(["SPY"] + sp500 + portfolio_syms))
raw = fetch_universe_prices(universe)
sector_map = fetch_sector_map(list(raw.keys()))
aligned, common_dates = build_aligned_prices(raw)
spy_bull = fetch_spy_regime(aligned, len(common_dates))
n = len(common_dates)
print(f"資料就緒（{len(aligned)} 支，{n} 天）\n")

# SPY B&H 基準
spy_bnh = calc_bnh_metrics(aligned["SPY"], n)
print(f"  {'SPY B&H':12s}  CAGR {spy_bnh['CAGR%']:+.1f}%  MDD {spy_bnh['MDD%']:.1f}%  Calmar {spy_bnh['Calmar']:.3f}")
print()

# 保護期掃描
test_days = [0, 7, 14, 21, 30, 45, 60]
results = []

for days in test_days:
    label = f"{days}天" if days > 0 else "無限制"
    cfg = StrategyConfig(f"Hold-{label}", min_hold_days=days)
    m = PortfolioSimulator(cfg).run(aligned, sector_map, spy_bull, common_dates)
    m["保護期"] = label
    m["vs SPY"] = round(m["CAGR%"] - spy_bnh["CAGR%"], 1)
    results.append(m)
    print(f"  {label:6s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}  換手 {m['Turnover%/yr']:.0f}%/yr  vs SPY {m['vs SPY']:+.1f}%")

# 同時跑追蹤停損版本（現行最佳策略）的保護期敏感度
print("\n  --- 加入追蹤停損（+TrailingStop）---\n")
results_ts = []
for days in test_days:
    label = f"{days}天" if days > 0 else "無限制"
    cfg = StrategyConfig(f"TS-{label}", trailing=True, min_hold_days=days)
    m = PortfolioSimulator(cfg).run(aligned, sector_map, spy_bull, common_dates)
    m["保護期"] = label
    m["vs SPY"] = round(m["CAGR%"] - spy_bnh["CAGR%"], 1)
    results_ts.append(m)
    print(f"  {label:6s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}  換手 {m['Turnover%/yr']:.0f}%/yr  vs SPY {m['vs SPY']:+.1f}%")

# 結果表
print(f"\n{'='*75}")
print("  Baseline 保護期敏感度")
print(f"{'='*75}\n")
df = pd.DataFrame(results).set_index("保護期")
base_cal = df.loc["無限制", "Calmar"]
df["vs 無限制"] = (df["Calmar"] - base_cal).round(3)
print(df[["CAGR%", "MDD%", "Calmar", "vs 無限制", "vs SPY", "Turnover%/yr"]].to_string())

print(f"\n{'='*75}")
print("  +TrailingStop 保護期敏感度")
print(f"{'='*75}\n")
df_ts = pd.DataFrame(results_ts).set_index("保護期")
base_cal_ts = df_ts.loc["無限制", "Calmar"]
df_ts["vs 無限制"] = (df_ts["Calmar"] - base_cal_ts).round(3)
print(df_ts[["CAGR%", "MDD%", "Calmar", "vs 無限制", "vs SPY", "Turnover%/yr"]].to_string())

best_b  = df["Calmar"].idxmax()
best_ts = df_ts["Calmar"].idxmax()
print(f"\n  Baseline 最佳保護期：{best_b}（Calmar {df.loc[best_b, 'Calmar']:.3f}）")
print(f"  +TrailingStop 最佳保護期：{best_ts}（Calmar {df_ts.loc[best_ts, 'Calmar']:.3f}）")
print(f"\n  目前系統設定：30天")
