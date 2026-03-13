"""波動率調整停損敏感度測試（臨時腳本）

停損公式：stop_pct = -k × 14日平均絕對日報酬 × sqrt(21)
≈ k 個月波動率
例：STX 日波動 ~3% → 月波動 ~13.7% → k=1.0 時停損 -13.7%
    HWM 日波動 ~1.5% → 月波動 ~6.9% → k=1.0 時停損 -6.9%
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import portfolio_backtest as pb
from portfolio_backtest import (
    load_portfolio_symbols, fetch_universe_prices, fetch_sector_map,
    build_aligned_prices, fetch_spy_regime, PortfolioSimulator, StrategyConfig,
    get_sp500_tickers, TOP_N_SP500, calc_bnh_metrics, calc_vol_map,
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
print(f"資料就緒（{len(aligned)} 支，{n} 天）")

# 預計算波動率
print("計算波動率圖...", end=" ", flush=True)
vol_map = calc_vol_map(aligned)
print("完成\n")

# SPY B&H 基準
spy_bnh = calc_bnh_metrics(aligned["SPY"], n)
spy_cagr = spy_bnh["CAGR%"]
print(f"  {'SPY B&H':15s}  CAGR {spy_cagr:+.1f}%  MDD {spy_bnh['MDD%']:.1f}%  Calmar {spy_bnh['Calmar']:.3f}")
print()

# 固定停損基準（現行）
cfg_fixed = StrategyConfig("Fixed -15%（現行）", vol_stop_k=0.0)
m_fixed = PortfolioSimulator(cfg_fixed).run(aligned, sector_map, spy_bull, common_dates, vol_map)
m_fixed["label"] = "Fixed -15%（現行）"
print(f"  {'Fixed -15%（現行）':20s}  CAGR {m_fixed['CAGR%']:+.1f}%  MDD {m_fixed['MDD%']:.1f}%  Calmar {m_fixed['Calmar']:.3f}  換手 {m_fixed['Turnover%/yr']:.0f}%/yr")

results = [m_fixed]

# 波動率停損掃描
for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    label = f"Vol×{k:.2f}"
    cfg = StrategyConfig(label, vol_stop_k=k)
    m = PortfolioSimulator(cfg).run(aligned, sector_map, spy_bull, common_dates, vol_map)
    m["label"] = label
    results.append(m)
    print(f"  {label:20s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}  換手 {m['Turnover%/yr']:.0f}%/yr")

# 加追蹤停損的版本
print("\n  --- 加入追蹤停損（+TrailingStop）---\n")
results_ts = []

cfg_ts_fixed = StrategyConfig("Fixed+Trail", trailing=True, vol_stop_k=0.0)
m = PortfolioSimulator(cfg_ts_fixed).run(aligned, sector_map, spy_bull, common_dates, vol_map)
m["label"] = "Fixed+Trail"
results_ts.append(m)
print(f"  {'Fixed+Trail':20s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}  換手 {m['Turnover%/yr']:.0f}%/yr")

for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    label = f"Vol×{k:.2f}+Trail"
    cfg = StrategyConfig(label, trailing=True, vol_stop_k=k)
    m = PortfolioSimulator(cfg).run(aligned, sector_map, spy_bull, common_dates, vol_map)
    m["label"] = label
    results_ts.append(m)
    print(f"  {label:20s}  CAGR {m['CAGR%']:+.1f}%  MDD {m['MDD%']:.1f}%  Calmar {m['Calmar']:.3f}  換手 {m['Turnover%/yr']:.0f}%/yr")

# 結果表
print(f"\n{'='*85}")
print("  波動率調整停損 vs 固定停損（Baseline，無追蹤停損）")
print(f"{'='*85}\n")
df = pd.DataFrame(results).set_index("label")
base_cal = df.loc["Fixed -15%（現行）", "Calmar"]
df["vs Fixed"] = (df["Calmar"] - base_cal).round(3)
df["vs SPY"]   = (df["CAGR%"] - spy_cagr).round(1)
print(df[["CAGR%", "MDD%", "Calmar", "vs Fixed", "vs SPY", "Turnover%/yr"]].to_string())

print(f"\n{'='*85}")
print("  波動率調整停損 vs 固定停損（+TrailingStop）")
print(f"{'='*85}\n")
df_ts = pd.DataFrame(results_ts).set_index("label")
base_cal_ts = df_ts.loc["Fixed+Trail", "Calmar"]
df_ts["vs Fixed+Trail"] = (df_ts["Calmar"] - base_cal_ts).round(3)
df_ts["vs SPY"] = (df_ts["CAGR%"] - spy_cagr).round(1)
print(df_ts[["CAGR%", "MDD%", "Calmar", "vs Fixed+Trail", "vs SPY", "Turnover%/yr"]].to_string())

best_b  = df["Calmar"].idxmax()
best_ts = df_ts["Calmar"].idxmax()
print(f"\n  Baseline 最佳：{best_b}（Calmar {df.loc[best_b, 'Calmar']:.3f}）")
print(f"  +TrailingStop 最佳：{best_ts}（Calmar {df_ts.loc[best_ts, 'Calmar']:.3f}）")
