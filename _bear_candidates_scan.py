"""
BEAR 模式下的 ADD 候選掃描（忽略 BEAR 限制，純看動能+Alpha+趨勢狀態）
用途：了解「如果不限制，會買什麼」
"""
import sys, json
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import get_sp500_tickers
from src.momentum import rank_by_momentum, calculate_trend_state_batch
import yfinance as yf
import numpy as np

# 載入持倉
with open("data/portfolio.json") as f:
    pf = json.load(f)
held = set(pf["positions"].keys())

tickers = get_sp500_tickers()
candidates = [t for t in tickers if t not in held]
print(f"候選池: {len(candidates)} 支（排除 {len(held)} 個持倉）")

# 取動能前 60
print("計算動能分數（前60）...")
ranked = rank_by_momentum(candidates, top_n=60)
top30 = ranked[:30]
print(f"取得 {len(ranked)} 支排名結果")

# 批次計算趨勢狀態（前30）
print("計算趨勢狀態（前30）...")
top30_syms = [item["symbol"] for item in top30]
trend_map = calculate_trend_state_batch(top30_syms)

# 讀取 alpha
def get_alpha_yf(sym, years):
    try:
        spy = yf.Ticker("SPY").history(period=f"{years+1}y", auto_adjust=True)["Close"]
        stk = yf.Ticker(sym).history(period=f"{years+1}y", auto_adjust=True)["Close"]
        n = years * 252
        if len(spy) < n or len(stk) < n:
            return None
        spy_ret = (spy.iloc[-1] / spy.iloc[-n] - 1) * 100
        stk_ret = (stk.iloc[-1] / stk.iloc[-n] - 1) * 100
        return float(stk_ret - spy_ret)
    except Exception:
        return None

print("\n正在取得 Alpha 資料（批次）...")

results = []
for i, item in enumerate(top30):
    sym = item["symbol"]
    mom = item.get("momentum", item.get("mom_mixed", 0) if "mom_mixed" in item else 0)
    rsi = item.get("rsi", 0) or 0

    # 趨勢狀態
    td = trend_map.get(sym)
    if td:
        state = td["state"]
        from_high = td["from_high_pct"]
        bounce = td["bounce_pct"]
        if state == "轉強":
            trend_str = f"↗️轉強(+{bounce:.0f}%)"
        elif state == "轉弱":
            trend_str = f"↘️轉弱({from_high:.0f}%)"
        else:
            trend_str = "→盤整"
    else:
        state = "未知"
        trend_str = "N/A"

    a1 = get_alpha_yf(sym, 1)
    a3 = get_alpha_yf(sym, 3)

    if a1 is None:
        qualifies = None
        alpha_note = "alpha N/A"
    elif a1 > 0 and (a3 is None or a3 > -30):
        qualifies = True
        alpha_note = f"1Y{a1:+.0f}%  3Y{a3:+.0f}%" if a3 is not None else f"1Y{a1:+.0f}%"
    else:
        qualifies = False
        a3_str = f"3Y{a3:+.0f}%" if a3 is not None else "3Y N/A"
        alpha_note = f"1Y{a1:+.0f}%  {a3_str}"

    # 倒V警告：動能正 + 轉弱
    inv_v = (mom > 0 and state == "轉弱")

    results.append({
        "rank": i + 1,
        "symbol": sym,
        "momentum": mom,
        "rsi": rsi,
        "alpha_1y": a1,
        "alpha_3y": a3,
        "qualifies": qualifies,
        "alpha_note": alpha_note,
        "trend_str": trend_str,
        "state": state,
        "inv_v": inv_v,
    })

    inv_v_warn = "  ⚠️倒V" if inv_v else ""
    print(f"  [{i+1:2d}] {sym:<7} mom={mom:+.2f} rsi={rsi:.0f}  {alpha_note:<22}  {trend_str}{inv_v_warn}", flush=True)

# 輸出最終清單
print("\n" + "="*75)
print("  ADD 候選清單（動能前30，Alpha + 趨勢狀態）")
print("="*75)
print(f"{'排名':<4} {'代號':<8} {'動能分':>8} {'RSI':>6}  {'Alpha':<22}  {'趨勢'}")
print("-"*75)

main_list = [r for r in results if r["qualifies"] is True]
backup_list = [r for r in results if r["qualifies"] is False]

print("▸ 主清單（Alpha 合格）：")
for r in main_list[:10]:
    rsi_warn = " 🔴" if r["rsi"] > 80 else (" 🟡" if r["rsi"] > 75 else "")
    inv_v_warn = "  ⚠️倒V警告" if r["inv_v"] else ""
    trend_display = r["trend_str"]
    print(f"  {r['rank']:<4} {r['symbol']:<8} {r['momentum']:>+8.2f} {r['rsi']:>6.1f}  {r['alpha_note']:<22}  {trend_display}{rsi_warn}{inv_v_warn}")

print(f"\n▸ 備選（Alpha 不足）：")
for r in backup_list[:5]:
    inv_v_warn = "  ⚠️倒V警告" if r["inv_v"] else ""
    print(f"  {r['rank']:<4} {r['symbol']:<8} {r['momentum']:>+8.2f} {r['rsi']:>6.1f}  {r['alpha_note']:<22}  {r['trend_str']}{inv_v_warn}")

# 倒V警告彙總
inv_v_list = [r for r in results if r["inv_v"] and r["qualifies"] is True]
if inv_v_list:
    print(f"\n⚠️  倒V警告標的（動能正但趨勢轉弱，建議跳過）：{', '.join(r['symbol'] for r in inv_v_list)}")

print(f"\n主清單 {len(main_list)} 支  /  備選 {len(backup_list)} 支  /  共掃描 {len(results)} 支")
