"""
ML% 交易稽核：回算 2026/02 以來每筆 ADD 的「當時 ML%」，與實際結果比對

使用方式：
    conda run -n qt_env python _ml_trade_audit.py
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ── 常數（與 ml_scorer.py 一致）────────────────────────────────────────────
FEATURE_COLS = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct",
    "breadth", "vix_level", "vix_ma_ratio",
    "oil_ret_21d", "oil_ret_63d", "oil_ma_ratio", "oil_vs_spy_21d",
    "sector_rel_21d", "sector_rel_63d", "sector_rel_126d", "sector_streak",
    "month_sin", "month_cos",
    "sec_XLB", "sec_XLC", "sec_XLE", "sec_XLF", "sec_XLI",
    "sec_XLK", "sec_XLP", "sec_XLRE", "sec_XLU", "sec_XLV", "sec_XLY",
]
SECTOR_ETF_MAP = {
    "Technology": "XLK", "Information Technology": "XLK",
    "Financial Services": "XLF", "Financials": "XLF",
    "Healthcare": "XLV", "Health Care": "XLV",
    "Energy": "XLE", "Utilities": "XLU",
    "Industrials": "XLI", "Basic Materials": "XLB",
    "Materials": "XLB", "Real Estate": "XLRE",
    "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))


# ── 讀取交易紀錄 ────────────────────────────────────────────────────────────

def load_trades():
    with open("data/portfolio.json") as f:
        port = json.load(f)
    txns = port.get("transactions", [])
    adds  = [t for t in txns if t.get("action") == "ADD"]
    exits = [t for t in txns if t.get("action") == "EXIT"]
    return adds, exits, port


def match_outcomes(adds, exits, current_prices):
    """
    對每筆 ADD，找到對應的 EXIT（同標的，日期在 ADD 之後最早的一筆）。
    若未出場則用現價計算未實現報酬。
    """
    # 按標的整理 exits
    exit_by_sym = {}
    for e in sorted(exits, key=lambda x: x["date"]):
        exit_by_sym.setdefault(e["symbol"], []).append(e)

    results = []
    used_exits = set()
    for a in sorted(adds, key=lambda x: x["date"]):
        sym   = a["symbol"]
        entry_date  = a["date"]
        entry_price = a.get("price", 0)

        # 找第一筆在 ADD 之後的 EXIT（同標的）
        exit_info = None
        for i, e in enumerate(exit_by_sym.get(sym, [])):
            key = (sym, e["date"], i)
            if e["date"] >= entry_date and key not in used_exits:
                exit_info = e
                used_exits.add(key)
                break

        if exit_info:
            exit_price = exit_info.get("price", 0)
            ret = (exit_price / entry_price - 1) * 100 if entry_price > 0 else None
            status = f"已出場 {exit_info['date']}"
            outcome_price = exit_price
        else:
            curr = current_prices.get(sym)
            ret = (curr / entry_price - 1) * 100 if (curr and entry_price > 0) else None
            status = "持有中"
            outcome_price = curr

        results.append({
            "date":          entry_date,
            "symbol":        sym,
            "entry_price":   entry_price,
            "outcome_price": outcome_price,
            "ret_pct":       ret,
            "status":        status,
        })
    return results


# ── 特徵計算（在指定歷史日期切片）─────────────────────────────────────────

def compute_features_at(sym, trade_date_str, close_all, vix_s, oil_s, spy_s,
                        etf_close, sector_map, medians):
    """
    以 trade_date 為截止點，計算單支股票在該日期的 36 個特徵。
    回傳 pd.Series（FEATURE_COLS）或 None。
    """
    td = pd.Timestamp(trade_date_str)

    # 取股票到 trade_date 為止的收盤價序列
    if sym not in close_all.columns:
        return None
    s = close_all[sym].dropna()
    s = s[s.index <= td]
    if len(s) < 63:
        return None

    p0 = float(s.iloc[-1])

    def mom(n):
        if len(s) <= n:
            return np.nan
        return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1)

    def vol(n):
        if len(s) < n + 1:
            return np.nan
        lr = np.log(s.iloc[-n:] / s.iloc[-n:].shift(1)).dropna()
        return float(lr.std() * np.sqrt(252))

    feats = {
        "mom_5d":        mom(5),
        "mom_21d":       mom(21),
        "mom_63d":       mom(63),
        "mom_126d":      mom(126),
        "mom_252d":      mom(252),
        "vol_21d":       vol(21),
        "vol_63d":       vol(63),
        "near52":        float(p0 / s.rolling(252, min_periods=50).max().iloc[-1]) if len(s) >= 50 else np.nan,
        "dist_ma50":     float(p0 / s.rolling(50, min_periods=20).mean().iloc[-1] - 1) if len(s) >= 20 else np.nan,
        "dist_ma200":    float(p0 / s.rolling(200, min_periods=60).mean().iloc[-1] - 1) if len(s) >= 60 else np.nan,
        "bounce_pct":    float(p0 / s.rolling(40, min_periods=10).min().iloc[-1] - 1) if len(s) >= 10 else np.nan,
        "from_high_pct": float(p0 / s.rolling(40, min_periods=10).max().iloc[-1] - 1) if len(s) >= 10 else np.nan,
        "breadth":       0.55,   # 歷史廣度難以精確重建，用中性值
    }

    # VIX
    v_s = vix_s[vix_s.index <= td].dropna()
    if len(v_s) >= 20:
        vl = float(v_s.iloc[-1])
        vm = float(v_s.rolling(63, min_periods=20).mean().iloc[-1])
        feats["vix_level"]    = vl
        feats["vix_ma_ratio"] = vl / vm if vm > 0 else np.nan
    else:
        feats["vix_level"] = feats["vix_ma_ratio"] = np.nan

    # 油價（USO）
    o_s = oil_s[oil_s.index <= td].dropna()
    s_s = spy_s[spy_s.index <= td].dropna()

    def pct_n(series, n):
        if len(series) <= n:
            return np.nan
        return float(series.iloc[-1] / series.iloc[-(n + 1)] - 1) * 100

    if len(o_s) >= 22:
        feats["oil_ret_21d"]    = pct_n(o_s, 21)
        feats["oil_ret_63d"]    = pct_n(o_s, 63)
        om200 = float(o_s.rolling(200, min_periods=60).mean().iloc[-1]) if len(o_s) >= 60 else np.nan
        feats["oil_ma_ratio"]   = float(o_s.iloc[-1]) / om200 if om200 and om200 > 0 else np.nan
        spy_r21 = pct_n(s_s, 21) if len(s_s) > 21 else 0
        feats["oil_vs_spy_21d"] = (feats["oil_ret_21d"] or 0) - (spy_r21 or 0)
    else:
        feats["oil_ret_21d"] = feats["oil_ret_63d"] = feats["oil_ma_ratio"] = feats["oil_vs_spy_21d"] = np.nan

    # 板塊特徵
    etf = sector_map.get(sym, "")
    for win, key in [(21, "sector_rel_21d"), (63, "sector_rel_63d"), (126, "sector_rel_126d")]:
        feats[key] = np.nan
        if etf and etf in etf_close.columns:
            e_s = etf_close[etf][etf_close.index <= td].dropna()
            s_s2 = s_s
            if len(e_s) > win and len(s_s2) > win:
                er = float(e_s.iloc[-1] / e_s.iloc[-(win + 1)] - 1)
                sr = float(s_s2.iloc[-1] / s_s2.iloc[-(win + 1)] - 1)
                feats[key] = er - sr

    # sector_streak（近月板塊連勝）
    streak = 0
    if etf and etf in etf_close.columns:
        e_m = etf_close[etf][etf_close.index <= td].dropna().resample("ME").last()
        s_m = spy_s[spy_s.index <= td].dropna().resample("ME").last()
        for i in range(1, min(13, len(e_m), len(s_m))):
            er = float(e_m.iloc[-i] / e_m.iloc[-i - 1] - 1) if len(e_m) > i else 0
            sr = float(s_m.iloc[-i] / s_m.iloc[-i - 1] - 1) if len(s_m) > i else 0
            if er > sr:
                streak += 1
            else:
                break
    feats["sector_streak"] = float(streak)

    # 季節性
    month = td.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    # 板塊 one-hot
    for e in SECTOR_ETFS:
        feats[f"sec_{e}"] = 1.0 if etf == e else 0.0

    row = pd.Series(feats)[FEATURE_COLS]
    # 用訓練集中位數填補缺值
    row = row.fillna(medians)
    return row


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    # 1. 載入模型
    print("載入 ML 模型...")
    with open("data/_ml_model.pkl", "rb") as f:
        payload = pickle.load(f)
    model   = payload["model"]
    scaler  = payload["scaler"]
    medians = payload["medians"]

    # 2. 載入交易記錄
    adds, exits, port = load_trades()
    print(f"ADD {len(adds)} 筆，EXIT {len(exits)} 筆")

    # 3. 載入板塊映射
    sector_map = {}
    if os.path.exists("data/_ml_sector_map.pkl"):
        sm = pd.read_pickle("data/_ml_sector_map.pkl")
        sector_map = sm.to_dict() if isinstance(sm, pd.Series) else sm

    # 4. 確定所有要下載的股票
    all_syms = list({a["symbol"] for a in adds})
    fetch_syms = list(set(all_syms) | set(SECTOR_ETFS) | {"SPY", "^VIX", "USO"})
    print(f"下載 {len(fetch_syms)} 支股票歷史資料（period=500d）...")
    raw = yf.download(fetch_syms, period="500d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_all = raw["Close"]
    else:
        close_all = raw
    close_all.index = pd.to_datetime(close_all.index).tz_localize(None)
    close_all = close_all.ffill()

    spy_s = close_all.get("SPY", pd.Series(dtype=float)).dropna()
    vix_s = close_all.get("^VIX", pd.Series(dtype=float)).dropna()
    oil_s = close_all.get("USO",  pd.Series(dtype=float)).dropna()

    etf_close = pd.DataFrame()
    for e in SECTOR_ETFS:
        if e in close_all.columns:
            etf_close[e] = close_all[e]

    # 5. 取得現價（用於計算「持有中」的未實現報酬）
    current_prices = {}
    for sym in all_syms:
        if sym in close_all.columns:
            s = close_all[sym].dropna()
            if not s.empty:
                current_prices[sym] = float(s.iloc[-1])

    # 6. 對每筆 ADD 計算當時 ML%
    print("計算每筆 ADD 的歷史 ML%...")
    outcomes = match_outcomes(adds, exits, current_prices)

    # 合併 ML%
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(model)

    rows = []
    for item in outcomes:
        sym        = item["symbol"]
        trade_date = item["date"]

        feat_row = compute_features_at(
            sym, trade_date, close_all, vix_s, oil_s, spy_s,
            etf_close, sector_map, medians
        )

        if feat_row is None:
            ml_pct = None
            shap_str = "資料不足"
        else:
            x_scaled = scaler.transform(feat_row.values.reshape(1, -1))
            prob = float(model.predict_proba(x_scaled)[0, 1])
            ml_pct = round(prob * 100, 1)

            # SHAP top 2
            sv = explainer.shap_values(x_scaled)[0]
            top2_idx = np.argsort(np.abs(sv))[::-1][:2]
            parts = []
            for idx in top2_idx:
                label = FEATURE_COLS[idx]
                arrow = "↑" if sv[idx] > 0 else "↓"
                parts.append(f"{arrow}{label}")
            shap_str = " | ".join(parts)

        rows.append({
            "date":   trade_date,
            "symbol": sym,
            "ml_pct": ml_pct,
            "shap":   shap_str,
            "entry":  item["entry_price"],
            "exit":   item["outcome_price"],
            "ret":    item["ret_pct"],
            "status": item["status"],
        })

    # 7. 排序：依日期，再依 ML% 降序
    rows.sort(key=lambda x: (x["date"], -(x["ml_pct"] or 0)))

    # ── 顯示 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  ML% 交易稽核  2026/02–03  （ML% 為當時特徵計算，非事後）")
    print("=" * 90)
    print(f"  {'日期':<12} {'標的':<7} {'ML%':>5}  {'進場':>8}  {'結果':>8}  {'報酬':>7}  {'狀態':<16}  {'主要ML因素'}")
    print("-" * 90)

    for r in rows:
        ml_str  = f"{r['ml_pct']:.0f}%" if r["ml_pct"] is not None else "N/A"
        ret_str = f"{r['ret']:+.1f}%" if r["ret"] is not None else "N/A"
        exit_str = f"${r['exit']:.2f}" if r["exit"] else "—"

        # 顏色提示（僅文字）
        if r["ml_pct"] is not None:
            confidence = "🟢" if r["ml_pct"] >= 60 else ("🟡" if r["ml_pct"] >= 50 else "🔴")
        else:
            confidence = "⚪"

        if r["ret"] is not None:
            win = "✅" if r["ret"] > 0 else "❌"
        else:
            win = "…"

        print(f"  {r['date']:<12} {r['symbol']:<7} {confidence}{ml_str:>4}  ${r['entry']:>7.2f}  {exit_str:>9}  {ret_str:>7}  {win} {r['status']:<14}  {r['shap']}")

    # ── 統計摘要 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  分組統計（ML%）")
    print("=" * 90)
    print(f"  {'組別':<20} {'筆數':>5}  {'平均報酬':>9}  {'勝率':>7}  {'停損/出場率':>10}  {'未實現平均':>10}")
    print("-" * 90)

    buckets = [
        ("🟢 高信心 ML≥60%",  lambda r: r["ml_pct"] is not None and r["ml_pct"] >= 60),
        ("🟡 中等  50≤ML<60", lambda r: r["ml_pct"] is not None and 50 <= r["ml_pct"] < 60),
        ("🔴 低信心 ML<50%",   lambda r: r["ml_pct"] is not None and r["ml_pct"] < 50),
    ]

    for label, cond in buckets:
        group = [r for r in rows if cond(r)]
        if not group:
            continue
        rets = [r["ret"] for r in group if r["ret"] is not None]
        if not rets:
            avg_r = win_r = "N/A"
        else:
            avg_r = f"{np.mean(rets):+.1f}%"
            win_r = f"{sum(1 for r in rets if r > 0)/len(rets)*100:.0f}%"

        closed   = [r for r in group if "已出場" in r["status"]]
        open_pos = [r for r in group if "持有中" in r["status"]]
        open_rets = [r["ret"] for r in open_pos if r["ret"] is not None]
        unrealized = f"{np.mean(open_rets):+.1f}%" if open_rets else "—"

        print(f"  {label:<20} {len(group):>5}  {avg_r:>9}  {win_r:>7}  {len(closed):>4}/{len(group):<4}        {unrealized:>10}")

    # ── ML% 最低的 ADD 是否都是問題標的 ────────────────────────────────────
    print()
    print("=" * 90)
    print("  當時 ML% 最低的 5 筆 ADD（信心最低的操作）")
    print("=" * 90)
    sorted_by_ml = sorted([r for r in rows if r["ml_pct"] is not None], key=lambda x: x["ml_pct"])
    for r in sorted_by_ml[:5]:
        ret_str = f"{r['ret']:+.1f}%" if r["ret"] is not None else "N/A"
        print(f"  {r['date']} {r['symbol']:<7} ML:{r['ml_pct']:.0f}%  報酬:{ret_str}  {r['status']}")

    print()
    print("  當時 ML% 最高的 5 筆 ADD（信心最強的操作）")
    print("=" * 90)
    for r in sorted_by_ml[-5:][::-1]:
        ret_str = f"{r['ret']:+.1f}%" if r["ret"] is not None else "N/A"
        print(f"  {r['date']} {r['symbol']:<7} ML:{r['ml_pct']:.0f}%  報酬:{ret_str}  {r['status']}")

    print()


if __name__ == "__main__":
    main()
