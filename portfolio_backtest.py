#!/usr/bin/env python3
"""投組層級回測模擬器

模擬完整投組管理流程（ADD / ROTATE / EXIT），驗證以下規則對長期績效的影響：
- 停利規則（漲 +30% 出半倉，剩餘停損移至成本 breakeven）
- 追蹤停損（從持倉最高點回落 -25% 出場）
- 板塊集中度上限（同板塊最多 3 檔）

策略比較（6 種，所有策略均啟用 SPY MA200 體制偵測）：
  Baseline            : Fixed-15% + Regime（現有系統近似）
  +ProfitTake         : 加入停利（+30% 出半倉）
  +TrailingStop       : 加入追蹤停損（從高點 -25%）
  +SectorLimit        : 加入板塊上限（同板塊最多 3 檔）
  +ProfitTake+Sector  : 停利 + 板塊限制
  +All                : 停利 + 追蹤停損 + 板塊限制

候選池：S&P 500 前 100 + 目前持倉
時間範圍：10 年
起始資金：$130,000

Usage:
    python portfolio_backtest.py
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import get_sp500_tickers, fetch_stock_data

# ── 常數 ──────────────────────────────────────────────────────────────────────
PORTFOLIO_PATH  = "data/portfolio.json"
SECTOR_MAP_PATH = "data/sector_map.json"
PERIOD          = "10y"
INITIAL_CASH    = 130_000.0
MAX_POSITIONS   = 30
MIN_POSITION_VAL = 500.0     # 最小建倉金額
FIXED_STOP      = -0.15      # 固定停損（相對成本）
ROTATE_GAP      = 0.10       # ROTATE 動能差門檻
REBALANCE_DAYS  = 5          # 每幾個交易日重平衡
MOM_WARMUP      = 252        # 動能計算暖身期（天）
TOP_N_SP500     = 503        # S&P 500 全部（None 或大數字 = 全取）

# 停利 / 追蹤停損參數
PROFIT_TAKE_PCT = 0.30       # 漲幅 ≥ +30% 觸發停利
TRAILING_PCT    = 0.25       # 從高點回落 ≥ -25% 出場
MAX_PER_SECTOR  = 3          # 同板塊最多 N 檔


@dataclass
class StrategyConfig:
    name: str
    profit_take: bool = False   # 啟用停利
    trailing: bool = False      # 啟用追蹤停損
    sector_limit: bool = False  # 啟用板塊上限
    min_hold_days: int = 0      # ROTATE 最短持有天數（0 = 無限制）
    vol_stop_k: float = 0.0    # 波動率停損倍數（0 = 固定 -15%；>0 = k × 月波動率）


STRATEGIES = [
    StrategyConfig("Baseline"),
    StrategyConfig("+ProfitTake",        profit_take=True),
    StrategyConfig("+TrailingStop",      trailing=True),
    StrategyConfig("+SectorLimit",       sector_limit=True),
    StrategyConfig("+ProfitTake+Sector", profit_take=True, sector_limit=True),
    StrategyConfig("+All",               profit_take=True, trailing=True, sector_limit=True),
]


# ── 資料取得 ───────────────────────────────────────────────────────────────────

def load_portfolio_symbols() -> list:
    with open(PORTFOLIO_PATH) as f:
        data = json.load(f)
    return list(data.get("positions", {}).keys())


def fetch_universe_prices(symbols: list) -> dict:
    """下載全部標的 10Y 日線，回傳 {symbol: pd.Series(Close, DatetimeIndex)}"""
    prices = {}
    for sym in symbols:
        try:
            df = fetch_stock_data(sym, period=PERIOD)
            if df.empty or len(df) < 500:
                continue
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
            idx = pd.to_datetime(close.index, utc=True).tz_convert(None)
            close.index = idx.normalize()
            close = close[~close.index.duplicated(keep="last")]
            prices[sym] = close.astype(float)
        except Exception:
            pass
    return prices


def fetch_sector_map(symbols: list) -> dict:
    """取得板塊分類，優先從 cache 讀取，缺少的才向 yfinance 請求"""
    if os.path.exists(SECTOR_MAP_PATH):
        with open(SECTOR_MAP_PATH) as f:
            cached = json.load(f)
    else:
        cached = {}

    missing = [s for s in symbols if s not in cached]
    if missing:
        print(f"  取得 {len(missing)} 支板塊分類...", end=" ", flush=True)
        for sym in missing:
            try:
                info = yf.Ticker(sym).info
                cached[sym] = info.get("sector", "Unknown") or "Unknown"
            except Exception:
                cached[sym] = "Unknown"
        with open(SECTOR_MAP_PATH, "w") as f:
            json.dump(cached, f, indent=2)
        print("完成")

    return {s: cached.get(s, "Unknown") for s in symbols}


def build_aligned_prices(raw: dict) -> tuple:
    """以 SPY 交易日為基準對齊所有股票（ffill），回傳 (aligned_np, common_dates)"""
    ref = raw.get("SPY", max(raw.values(), key=len))
    common_dates = ref.index

    aligned = {}
    for sym, series in raw.items():
        s = series.reindex(common_dates, method="ffill")
        aligned[sym] = s.values.astype(float)

    return aligned, common_dates


def fetch_spy_regime(aligned: dict, n: int) -> np.ndarray:
    """計算每日 SPY 體制（True=BULL, False=BEAR），缺資料一律視為 BULL"""
    spy = aligned.get("SPY")
    if spy is None:
        return np.ones(n, dtype=bool)
    ma200 = pd.Series(spy).rolling(200).mean().values
    bull = np.where(np.isnan(ma200), True, spy > ma200)
    return bull.astype(bool)


# ── 動能計算 ───────────────────────────────────────────────────────────────────

def calc_mixed_momentum(arr: np.ndarray, idx: int) -> float:
    """混合動能：50% 21日 + 50% 252日"""
    p = arr[idx]
    if p <= 0 or np.isnan(p):
        return float("nan")
    p21  = arr[idx - 21]  if idx >= 21  else arr[0]
    p252 = arr[idx - 252] if idx >= 252 else arr[0]
    if p21 <= 0 or p252 <= 0:
        return float("nan")
    return 0.5 * (p / p21 - 1) + 0.5 * (p / p252 - 1)


# ── Buy & Hold 基準計算 ────────────────────────────────────────────────────────

def calc_bnh_metrics(price_arr: np.ndarray, n_days: int) -> dict:
    """計算買入持有（Buy & Hold）績效指標，以初始資金換算規模"""
    vals  = price_arr * (INITIAL_CASH / price_arr[0])
    years = n_days / 252
    total_ret = (vals[-1] / vals[0] - 1) * 100
    cagr      = ((vals[-1] / vals[0]) ** (1 / years) - 1) * 100
    peak      = np.maximum.accumulate(vals)
    mdd       = ((vals - peak) / peak).min() * 100
    calmar    = cagr / abs(mdd) if mdd != 0 else 0.0
    return {
        "Return%":      round(total_ret, 1),
        "CAGR%":        round(cagr,      1),
        "MDD%":         round(mdd,        1),
        "Calmar":       round(calmar,     3),
        "Trades":       1,
        "Turnover%/yr": 0.0,
    }


# ── 波動率計算 ─────────────────────────────────────────────────────────────────

def calc_vol_map(aligned: dict, window: int = 14) -> dict:
    """計算各股票每日 N 日滾動波動率（日報酬標準差），回傳 {sym: np.ndarray}

    停損公式：stop_pct = -k × daily_vol × sqrt(21)
    約等於「k 個月波動率」為停損幅度，自動對高波動股放寬、低波動股收緊。
    """
    vol_map = {}
    for sym, arr in aligned.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(
                arr[:-1] > 0,
                np.log(np.where(arr[1:] > 0, arr[1:], arr[:-1]) / arr[:-1]),
                0.0,
            )
        vol = pd.Series(np.abs(rets)).rolling(window, min_periods=5).mean().fillna(0.02).values
        # 對齊：第 0 天用第 1 天的值
        vol_map[sym] = np.concatenate([[vol[0]], vol])
    return vol_map


# ── 投組模擬器 ─────────────────────────────────────────────────────────────────

VOL_STOP_FLOOR = 0.05   # 最緊停損（-5%，不讓低波動股被噪音洗掉）
VOL_STOP_CAP   = 0.30   # 最寬停損（-30%）


def _calc_stop(cfg: "StrategyConfig", vol_map: dict, sym: str, d_idx: int) -> float:
    """計算進場時的停損幅度（負數）"""
    if cfg.vol_stop_k <= 0 or vol_map is None:
        return FIXED_STOP  # 固定 -15%
    daily_vol = vol_map.get(sym, np.zeros(max(d_idx + 1, 1)))[d_idx]
    monthly_vol = daily_vol * np.sqrt(21)   # 換算成月波動率
    raw = -cfg.vol_stop_k * monthly_vol
    return max(-VOL_STOP_CAP, min(-VOL_STOP_FLOOR, raw))


class PortfolioSimulator:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def run(self, aligned: dict, sector_map: dict,
            spy_bull: np.ndarray, common_dates: pd.DatetimeIndex,
            vol_map: dict = None) -> dict:

        cfg   = self.cfg
        cash  = INITIAL_CASH
        # positions: {sym: {shares, avg_price, high_price, profit_taken}}
        positions: dict = {}
        daily_vals: list = []
        trade_count   = 0
        total_buy_val = 0.0

        syms = list(aligned.keys())
        n    = len(common_dates)

        for d_idx in range(n):

            # ── 當日價格快取 ─────────────────────────────────
            px = {s: aligned[s][d_idx] for s in syms}

            # ── 更新持倉最高價 ────────────────────────────────
            for sym, pos in positions.items():
                p = px.get(sym, np.nan)
                if np.isnan(p) or p <= 0:
                    continue
                if p > pos["high_price"]:
                    pos["high_price"] = p

            # ── 每日 EXIT 檢查 ────────────────────────────────
            to_exit = []
            for sym, pos in positions.items():
                p = px.get(sym, np.nan)
                if np.isnan(p) or p <= 0:
                    continue

                # 1. Regime 出場（BEAR 市場）
                if not spy_bull[d_idx]:
                    to_exit.append(sym)
                    continue

                # 2. Fixed / Vol stop（breakeven stop 若已停利，否則看策略）
                pnl = (p - pos["avg_price"]) / pos["avg_price"]
                threshold = 0.0 if pos.get("profit_taken") else pos.get("stop_pct", FIXED_STOP)
                if pnl <= threshold:
                    to_exit.append(sym)
                    continue

                # 3. Trailing stop（從最高點回落 -25%）
                if cfg.trailing:
                    from_high = (p - pos["high_price"]) / pos["high_price"]
                    if from_high <= -TRAILING_PCT:
                        to_exit.append(sym)
                        continue

                # 4. 停利（+30% 出半倉，僅觸發一次）
                if cfg.profit_take and not pos.get("profit_taken"):
                    if pnl >= PROFIT_TAKE_PCT:
                        half = max(1, pos["shares"] // 2)
                        cash += half * p
                        pos["shares"] -= half
                        pos["profit_taken"] = True
                        trade_count += 1
                        total_buy_val += 0  # sell-side 不計入 turnover

            for sym in to_exit:
                p = px.get(sym, positions[sym]["avg_price"])
                if np.isnan(p) or p <= 0:
                    p = positions[sym]["avg_price"]
                cash += positions[sym]["shares"] * p
                trade_count += 1
                del positions[sym]

            # ── 每週重平衡 ────────────────────────────────────
            if d_idx % REBALANCE_DAYS == 0 and d_idx >= MOM_WARMUP:

                # 計算全部標的動能排名
                mom: dict = {}
                for s in syms:
                    m = calc_mixed_momentum(aligned[s], d_idx)
                    if not np.isnan(m):
                        mom[s] = m
                ranked = sorted(mom.items(), key=lambda x: -x[1])

                # ROTATE：每週最多 1 次，汰弱換強（排除保護期內持倉）
                if positions:
                    rotatable = {
                        s: mom.get(s, -999.0) for s in positions
                        if d_idx - positions[s].get("entry_idx", 0) >= cfg.min_hold_days
                    }
                    if rotatable:
                        worst_sym = min(rotatable, key=rotatable.get)
                        worst_mom = rotatable[worst_sym]

                        for cand_sym, cand_mom in ranked:
                            if cand_sym in positions:
                                continue
                            if cand_mom - worst_mom < ROTATE_GAP:
                                break

                            # 板塊限制
                            if cfg.sector_limit:
                                sec = sector_map.get(cand_sym, "Unknown")
                                cnt = sum(1 for s in positions
                                          if sector_map.get(s, "Unknown") == sec)
                                if cnt >= MAX_PER_SECTOR:
                                    continue

                            p_sell = px.get(worst_sym, np.nan)
                            p_buy  = px.get(cand_sym, np.nan)
                            if np.isnan(p_sell) or np.isnan(p_buy) or p_buy <= 0:
                                continue

                            proceeds = positions[worst_sym]["shares"] * p_sell
                            cash += proceeds
                            del positions[worst_sym]
                            trade_count += 1

                            shares = int(proceeds // p_buy)
                            if shares > 0:
                                cost = shares * p_buy
                                cash -= cost
                                total_buy_val += cost
                                positions[cand_sym] = {
                                    "shares": shares,
                                    "avg_price": p_buy,
                                    "high_price": p_buy,
                                    "profit_taken": False,
                                    "entry_idx": d_idx,
                                    "stop_pct": _calc_stop(cfg, vol_map, cand_sym, d_idx),
                                }
                                trade_count += 1
                            break  # 每週最多 1 次 ROTATE

                # ADD：填補空缺槽位
                avail = MAX_POSITIONS - len(positions)
                if avail > 0 and cash >= MIN_POSITION_VAL:
                    per_slot = cash / avail

                    for cand_sym, cand_mom in ranked:
                        if len(positions) >= MAX_POSITIONS:
                            break
                        if cand_sym in positions:
                            continue

                        # 板塊限制
                        if cfg.sector_limit:
                            sec = sector_map.get(cand_sym, "Unknown")
                            cnt = sum(1 for s in positions
                                      if sector_map.get(s, "Unknown") == sec)
                            if cnt >= MAX_PER_SECTOR:
                                continue

                        p_buy = px.get(cand_sym, np.nan)
                        if np.isnan(p_buy) or p_buy <= 0:
                            continue

                        shares = int(per_slot // p_buy)
                        if shares <= 0:
                            continue

                        cost = shares * p_buy
                        cash -= cost
                        total_buy_val += cost
                        positions[cand_sym] = {
                            "shares": shares,
                            "avg_price": p_buy,
                            "high_price": p_buy,
                            "profit_taken": False,
                            "entry_idx": d_idx,
                            "stop_pct": _calc_stop(cfg, vol_map, cand_sym, d_idx),
                        }
                        trade_count += 1

            # ── 記錄當日淨值 ──────────────────────────────────
            port_val = cash + sum(
                pos["shares"] * max(px.get(s, pos["avg_price"]), 0.01)
                for s, pos in positions.items()
            )
            daily_vals.append(port_val)

        return self._metrics(daily_vals, trade_count, total_buy_val, n)

    @staticmethod
    def _metrics(daily_vals: list, trades: int, buy_val: float, n_days: int) -> dict:
        vals  = np.array(daily_vals)
        years = n_days / 252
        total_ret = (vals[-1] / vals[0] - 1) * 100
        cagr      = ((vals[-1] / vals[0]) ** (1 / years) - 1) * 100
        peak      = np.maximum.accumulate(vals)
        mdd       = ((vals - peak) / peak).min() * 100
        calmar    = cagr / abs(mdd) if mdd != 0 else 0.0
        avg_val   = vals.mean()
        turnover  = buy_val / avg_val / years * 100  # %/year

        return {
            "Return%":      round(total_ret, 1),
            "CAGR%":        round(cagr,      1),
            "MDD%":         round(mdd,        1),
            "Calmar":       round(calmar,     3),
            "Trades":       trades,
            "Turnover%/yr": round(turnover,   0),
        }


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print(f"\n=== 投組層級回測 | {PERIOD} | 起始 ${INITIAL_CASH:,.0f} | {date.today()} ===\n")

    # 取得候選標的
    print(f"  取得 S&P 500 全部標的...", end=" ", flush=True)
    try:
        sp500 = get_sp500_tickers()[:TOP_N_SP500]
        print(f"共 {len(sp500)} 支")
    except Exception:
        sp500 = []
        print("失敗，僅使用持倉標的")

    portfolio_syms = load_portfolio_symbols()
    universe = list(dict.fromkeys(["SPY"] + sp500 + portfolio_syms))
    print(f"  總候選池：{len(universe)} 支（含 SPY 基準）")

    # 下載價格（有快取則略過）
    print(f"\n  下載 {len(universe)} 支股票 {PERIOD} 歷史價格（有快取則略過）...")
    raw = fetch_universe_prices(universe)
    print(f"  成功取得：{len(raw)} 支（失敗/不足者已略過）\n")

    if "SPY" not in raw:
        print("❌ SPY 資料缺失，無法繼續")
        return

    # 板塊分類
    sector_map = fetch_sector_map(list(raw.keys()))

    # 對齊日期
    aligned, common_dates = build_aligned_prices(raw)
    spy_bull = fetch_spy_regime(aligned, len(common_dates))

    n = len(common_dates)
    bear_pct = (~spy_bull).mean() * 100
    print(f"  模擬期間：{common_dates[0].date()} → {common_dates[-1].date()}（{n} 交易日）")
    print(f"  SPY 熊市佔比：{bear_pct:.1f}%")
    print(f"  有效標的數：{len(aligned)}（含 SPY）\n")

    # 逐策略模擬
    results = []
    for cfg in STRATEGIES:
        print(f"  模擬 {cfg.name:25s}...", end=" ", flush=True)
        sim = PortfolioSimulator(cfg)
        m = sim.run(aligned, sector_map, spy_bull, common_dates)
        m["Strategy"] = cfg.name
        results.append(m)
        print(
            f"CAGR {m['CAGR%']:+.1f}%  "
            f"MDD {m['MDD%']:.1f}%  "
            f"Calmar {m['Calmar']:.3f}  "
            f"換手 {m['Turnover%/yr']:.0f}%/yr"
        )

    # ── SPY Buy & Hold 基準 ───────────────────────────────
    spy_bnh = calc_bnh_metrics(aligned["SPY"], n)
    spy_bnh["Strategy"] = "SPY B&H"
    spy_cagr = spy_bnh["CAGR%"]
    spy_cal  = spy_bnh["Calmar"]
    print(f"\n  SPY B&H 基準：CAGR {spy_cagr:+.1f}%  MDD {spy_bnh['MDD%']:.1f}%  Calmar {spy_cal:.3f}")

    # ── 結果表 ────────────────────────────────────────────
    df = pd.DataFrame(results).set_index("Strategy")
    df = df.reindex([c.name for c in STRATEGIES])
    baseline_cagr = df.loc["Baseline", "CAGR%"]
    df.insert(4, "vs Baseline", (df["CAGR%"] - baseline_cagr).round(1))
    df.insert(5, "vs SPY", (df["CAGR%"] - spy_cagr).round(1))

    # 加入 SPY B&H 為最後一行（參考用）
    spy_row = pd.DataFrame([{
        "Return%":      spy_bnh["Return%"],
        "CAGR%":        spy_bnh["CAGR%"],
        "MDD%":         spy_bnh["MDD%"],
        "Calmar":       spy_bnh["Calmar"],
        "vs Baseline":  round(spy_cagr - baseline_cagr, 1),
        "vs SPY":       0.0,
        "Trades":       spy_bnh["Trades"],
        "Turnover%/yr": spy_bnh["Turnover%/yr"],
    }], index=["SPY B&H"])
    df = pd.concat([df, spy_row])

    print(f"\n{'='*105}")
    print("  策略比較（投組層級，全期平均）")
    print(f"{'='*105}\n")
    print(df.to_string())

    # ── 結論 ──────────────────────────────────────────────
    print(f"\n{'='*105}")
    print("  回測結論")
    print(f"{'='*105}\n")

    best = df.drop("SPY B&H")["Calmar"].idxmax()
    print(f"  Calmar 最佳策略：{best}（{df.loc[best, 'Calmar']:.3f}）\n")

    print(f"  動能策略 vs SPY B&H（CAGR {spy_cagr:+.1f}%，Calmar {spy_cal:.3f}）：")
    baseline_cal = df.loc["Baseline", "Calmar"]
    for name, label in [("Baseline", "Baseline   "), ("+ProfitTake", "+停利      "),
                        ("+TrailingStop", "+追蹤停損  "), ("+SectorLimit", "+板塊上限  "),
                        ("+All", "+三規則組合")]:
        vs_spy  = df.loc[name, "vs SPY"]
        cal_diff = df.loc[name, "Calmar"] - spy_cal
        icon = "✓" if vs_spy > 0 else "✗"
        print(f"  {icon} {label}：CAGR {vs_spy:+.1f}% vs SPY  "
              f"Calmar {df.loc[name, 'Calmar']:.3f}（{cal_diff:+.3f} vs SPY）")

    # 儲存 CSV
    csv_path = f"data/backtest_portfolio_{date.today().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_path)
    print(f"\n詳細結果已儲存至：{csv_path}")


if __name__ == "__main__":
    run_backtest()
