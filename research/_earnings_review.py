"""
_earnings_review.py

用財報 beat 歷史重新檢視目前所有持倉：
  - 過去 4 季 EPS beat 率（實際 vs 分析師預期）
  - 連續 beat 數（最近開始算）
  - 平均 surprise %
  - 結合 1Y alpha、3Y alpha、動能，給出綜合評分

排序邏輯：需注意的放前面（財報弱 + alpha 差），強的放後面。
"""

import os, sys, json, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from src.data_loader import fetch_current_prices
from src.momentum import rank_by_momentum, calculate_alpha_batch, calculate_alpha_3y_batch

DATA_PATH = "data/portfolio.json"
QUARTERS   = 4   # 取幾季來評估


# ── 工具函數 ───────────────────────────────────────────────────────
def fetch_earnings_beats(symbol: str, n: int = QUARTERS) -> dict:
    """
    回傳 dict:
      beat_count       : 最近 n 季中 beat 幾次
      beat_rate        : beat 率 %
      consecutive      : 從最近一季往回數的連續 beat 數
      avg_surprise_pct : 平均 surprise（%）
      available        : 是否有資料
    """
    try:
        h = yf.Ticker(symbol).earnings_history
        if h is None or h.empty:
            return {"available": False}

        # 最新 n 季（升序排列，取最後 n 筆）
        recent = h.tail(n).copy()
        if len(recent) == 0:
            return {"available": False}

        beat_count = int((recent["epsDifference"] > 0).sum())
        beat_rate  = beat_count / len(recent) * 100

        # 從最新往舊計算連續 beat
        consecutive = 0
        for diff in reversed(recent["epsDifference"].tolist()):
            if diff > 0:
                consecutive += 1
            else:
                break

        avg_surprise = float(recent["surprisePercent"].mean()) * 100  # 轉成 %

        return {
            "available"  : True,
            "beat_count" : beat_count,
            "total"      : len(recent),
            "beat_rate"  : beat_rate,
            "consecutive": consecutive,
            "avg_surprise": avg_surprise,
        }
    except Exception:
        return {"available": False}


def earnings_signal(e: dict) -> str:
    """🟢 / 🟡 / 🔴 / ⬜"""
    if not e.get("available"):
        return "⬜"
    br = e["beat_rate"]
    cs = e["consecutive"]
    if br >= 75 and cs >= 2:
        return "🟢"
    elif br >= 50:
        return "🟡"
    else:
        return "🔴"


def alpha_signal(a) -> str:
    if a is None:
        return " N/A "
    if a > 0:
        return f"🟢{a:+.0f}%"
    elif a > -20:
        return f"🟡{a:+.0f}%"
    else:
        return f"🔴{a:+.0f}%"


def classify_holding(e, alpha_1y, alpha_3y) -> str:
    """給持倉一個 B1 / A 類型分類（同 ADD 候選的邏輯）"""
    ok_1y = alpha_1y is None or alpha_1y > 0
    ok_3y = alpha_3y is None or alpha_3y > 0
    if ok_1y and ok_3y:
        return "A-雙優"
    elif ok_1y:
        return "B1-近優"
    elif ok_3y:
        return "B2-遠優"
    else:
        return "C-雙差"


# ── 主程式 ────────────────────────────────────────────────────────
def main():
    print("=== 持倉財報 Beat 歷史 Review ===")
    print(f"日期：{datetime.date.today()}  |  取近 {QUARTERS} 季\n")

    # 1. 讀取持倉
    with open(DATA_PATH) as f:
        pf = json.load(f)
    positions = pf.get("positions", {})
    symbols   = list(positions.keys())
    print(f"持倉 {len(symbols)} 檔，開始抓資料...\n")

    # 2. 取得報價 / 動能 / alpha
    prices_map = fetch_current_prices(symbols + ["SPY"])
    spy_price  = prices_map.get("SPY", 1)

    print("計算動能與 alpha...")
    momentum_list = rank_by_momentum(symbols)
    momentum_map  = {m["symbol"]: m for m in momentum_list}

    alpha_1y_map  = calculate_alpha_batch(symbols)
    alpha_3y_map  = calculate_alpha_3y_batch(symbols)

    # 3. 逐一抓財報
    rows = []
    for sym in symbols:
        pos       = positions[sym]
        avg_price = pos.get("avg_price", 0)
        shares    = pos.get("shares", 0)
        core      = pos.get("core", False)
        favorite  = pos.get("favorite", False)

        cur_price = prices_map.get(sym, avg_price)
        pnl_pct   = (cur_price / avg_price - 1) * 100 if avg_price else 0

        m         = momentum_map.get(sym, {})
        momentum  = m.get("momentum")
        mom_rank  = m.get("rank")

        a1y = alpha_1y_map.get(sym)
        a3y = alpha_3y_map.get(sym)

        print(f"  {sym:6s} 財報資料...", end=" ", flush=True)
        earn = fetch_earnings_beats(sym)
        print("✓" if earn.get("available") else "無資料")

        rows.append({
            "symbol"    : sym,
            "core"      : core,
            "favorite"  : favorite,
            "shares"    : shares,
            "avg_price" : avg_price,
            "cur_price" : cur_price,
            "pnl_pct"   : pnl_pct,
            "momentum"  : momentum,
            "mom_rank"  : mom_rank,
            "alpha_1y"  : a1y,
            "alpha_3y"  : a3y,
            "earn"      : earn,
            "e_sig"     : earnings_signal(earn),
            "grp"       : classify_holding(earn, a1y, a3y),
        })

    # 4. 排序：財報弱 + alpha 差的優先顯示
    def sort_key(r):
        e = r["earn"]
        beat_rate = e.get("beat_rate", 50) if e.get("available") else 50
        a1y = r["alpha_1y"] or 0
        a3y = r["alpha_3y"] or 0
        concern = -(beat_rate / 100 + (1 if a1y > 0 else 0) + (1 if a3y > 0 else 0))
        return concern

    rows.sort(key=sort_key)

    # 5. 輸出主表
    print()
    print("=" * 100)
    print(f"  {'標的':<6}  {'P&L':>7}  {'動能':>7}  {'1Y Alpha':>9}  {'3Y Alpha':>9}  "
          f"{'Beat(4Q)':>8}  {'連續':>4}  {'Avg驚':>6}  {'財報':>4}  {'分組':<8}  備註")
    print("  " + "-" * 97)

    concerns = []
    for r in rows:
        sym   = r["symbol"]
        pnl   = r["pnl_pct"]
        mom   = r["momentum"]
        rank  = r["mom_rank"]
        a1y   = r["alpha_1y"]
        a3y   = r["alpha_3y"]
        e     = r["earn"]
        e_sig = r["e_sig"]
        grp   = r["grp"]

        pnl_str  = f"{pnl:+.1f}%"
        mom_str  = f"#{rank} {mom:+.1f}%" if rank and mom is not None else "  N/A"
        a1y_str  = alpha_signal(a1y)
        a3y_str  = alpha_signal(a3y)

        if e.get("available"):
            beat_str = f"{e['beat_count']}/{e['total']}"
            cons_str = str(e["consecutive"])
            surp_str = f"{e['avg_surprise']:+.1f}%"
        else:
            beat_str = " N/A"
            cons_str = "-"
            surp_str = "  N/A"

        tags = []
        if r["core"]:      tags.append("🔒core")
        if r["favorite"]:  tags.append("⭐fav")
        if pnl < -10:      tags.append("⚠️P&L低")
        if mom is not None and mom < -5:   tags.append("📉動能弱")
        if e.get("available") and e["beat_rate"] < 50:  tags.append("❗財報差")
        tag_str = " ".join(tags)

        print(f"  {sym:<6}  {pnl_str:>7}  {mom_str:>10}  {a1y_str:>9}  {a3y_str:>9}  "
              f"{beat_str:>8}  {cons_str:>4}  {surp_str:>6}  {e_sig:>4}  {grp:<8}  {tag_str}")

        if "❗財報差" in tag_str or ("⚠️P&L低" in tag_str and "📉動能弱" in tag_str):
            concerns.append(sym)

    # 6. 摘要統計
    available = [r for r in rows if r["earn"].get("available")]
    print()
    print("=" * 100)
    print(f"\n  財報資料取得：{len(available)}/{len(rows)} 檔")

    if available:
        avg_beat = sum(r["earn"]["beat_rate"] for r in available) / len(available)
        all_green = sum(1 for r in available if r["e_sig"] == "🟢")
        all_red   = sum(1 for r in available if r["e_sig"] == "🔴")
        print(f"  投組整體 beat 率：{avg_beat:.0f}%  |  🟢 {all_green} 檔  🔴 {all_red} 檔")

    grp_counts = {}
    for r in rows:
        grp_counts[r["grp"]] = grp_counts.get(r["grp"], 0) + 1
    print(f"\n  Alpha 分組分佈：")
    for g, c in sorted(grp_counts.items()):
        print(f"    {g}: {c} 檔")

    if concerns:
        print(f"\n  ⚠️  需關注（財報差或 P&L 低 + 動能弱）：{', '.join(concerns)}")
    else:
        print(f"\n  ✅ 無明顯財報或動能雙重警訊的持倉")


if __name__ == "__main__":
    main()
