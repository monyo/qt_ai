"""追蹤停損門檻敏感度測試（臨時腳本）"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import portfolio_backtest as pb
from portfolio_backtest import (
    load_portfolio_symbols, fetch_universe_prices, fetch_sector_map,
    build_aligned_prices, fetch_spy_regime, PortfolioSimulator, StrategyConfig,
    get_sp500_tickers, TOP_N_SP500,
)

print("載入快取資料...")
sp500 = get_sp500_tickers()[:TOP_N_SP500]
portfolio_syms = load_portfolio_symbols()
universe = list(dict.fromkeys(["SPY"] + sp500 + portfolio_syms))
raw = fetch_universe_prices(universe)
sector_map = fetch_sector_map(list(raw.keys()))
aligned, common_dates = build_aligned_prices(raw)
spy_bull = fetch_spy_regime(aligned, len(common_dates))
print(f"資料就緒（{len(aligned)} 支，{len(common_dates)} 天）\n")

# Baseline
cfg_base = StrategyConfig("Baseline", trailing=False)
m = PortfolioSimulator(cfg_base).run(aligned, sector_map, spy_bull, common_dates)
m["Threshold"] = "— 無追蹤 —"
results = [m]
print(f"  {'— 無追蹤 —':12s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}")

# 敏感度掃描
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    pb.TRAILING_PCT = t
    cfg = StrategyConfig(f"Trailing-{int(t*100)}%", trailing=True)
    m = PortfolioSimulator(cfg).run(aligned, sector_map, spy_bull, common_dates)
    m["Threshold"] = f"-{int(t*100)}%"
    results.append(m)
    print(f"  {m['Threshold']:12s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}  換手 {m['Turnover%/yr']:.0f}%/yr")

df = pd.DataFrame(results).set_index("Threshold")
base_cal = df.loc["— 無追蹤 —", "Calmar"]
df["vs Baseline"] = (df["Calmar"] - base_cal).round(3)

print(f"\n{'='*65}")
print("  追蹤停損門檻敏感度")
print(f"{'='*65}\n")
print(df[["CAGR%", "MDD%", "Calmar", "vs Baseline", "Trades"]].to_string())
print()
best = df["Calmar"].idxmax()
print(f"  Calmar 最佳門檻：{best}（Calmar {df.loc[best, 'Calmar']:.3f}）")
