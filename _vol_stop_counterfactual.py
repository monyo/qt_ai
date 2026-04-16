"""
_vol_stop_counterfactual.py

反事實模擬：如果從一開始就使用波動率分層停損，現在投組價值是多少？

方法：
1. 掃描 portfolio.json 所有已停損出場（EXIT）交易
2. 對每筆「明確停損（虧損 ≥ -12%）」，計算當時的年化波動率
3. 判斷波動率分層停損是否更寬（能避免被洗出場）
4. 若能避免：追蹤到現在的價格（或再次停損的價格），計算 P&L 差異

執行：
    conda run -n qt_env python _vol_stop_counterfactual.py
"""

import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── 資料 ─────────────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
with open("data/portfolio.json") as f:
    portfolio = json.load(f)

print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
df_high  = ohlcv["High"]
df_close.index = pd.to_datetime(df_close.index).tz_localize(None)
df_high.index  = pd.to_datetime(df_high.index).tz_localize(None)
last_cache_date = df_close.index[-1]
print(f"  快取截至：{last_cache_date.date()}")

# ── 波動率分層停損參數（與 risk.py 一致）─────────────────────────────────────
VOL_TIER_LOW  = 0.35
VOL_TIER_HIGH = 0.60

def vol_adjusted_fixed(vol):
    if vol is None or vol < VOL_TIER_LOW:
        return 0.15
    elif vol < VOL_TIER_HIGH:
        return 0.22
    else:
        return 0.25

def get_vol_at(sym, date_str, win=63):
    """計算 date 前 win 個交易日的年化波動率"""
    dt = pd.Timestamp(date_str)
    mask = df_close.index <= dt
    if sym not in df_close.columns:
        return None
    seg = df_close.loc[mask, sym].dropna().iloc[-win:]
    if len(seg) < 20:
        return None
    lr = np.log(seg / seg.shift(1)).dropna()
    return float(lr.std() * np.sqrt(252)) if len(lr) >= 15 else None

def get_price_at(sym, date_str):
    """從快取或 yfinance 取得某日收盤價"""
    dt = pd.Timestamp(date_str)
    if sym in df_close.columns and dt <= last_cache_date:
        mask = df_close.index <= dt
        px = df_close.loc[mask, sym].dropna()
        return float(px.iloc[-1]) if len(px) > 0 else None
    return None

# ── 從交易記錄找出「明確停損出場」───────────────────────────────────────────
# 策略：若某 EXIT 對應同期有 ADD/BUY，且虧損 ≥ -12%，視為停損出場
txs_global = portfolio["transactions"]

# 建立每支股票的 buy/sell 對應
from collections import defaultdict
buys_by_sym  = defaultdict(list)   # {sym: [(date, shares, price), ...]}
for tx in txs_global:
    act = tx["action"].upper()
    if act in ("ADD", "BUY"):
        buys_by_sym[tx["symbol"]].append((tx["date"], tx["shares"], tx["price"]))
    elif act == "EXIT":
        # 也把 positions 內的 transactions 包含
        pass

# 也納入 positions 裡的 transactions
for sym, pos in portfolio["positions"].items():
    for tx in pos.get("transactions", []):
        act = tx.get("action", "BUY").upper()
        if act in ("ADD", "BUY"):
            buys_by_sym[sym].append((tx["date"], tx["shares"], tx["price"]))

# 找所有 EXIT
exits = [(tx["symbol"], tx["date"], tx["shares"], tx["price"])
         for tx in txs_global if tx["action"].upper() == "EXIT"]

# ── 分析每筆 EXIT ─────────────────────────────────────────────────────────────
STOP_THRESHOLD = -0.12   # 虧損 ≥ -12% 視為停損出場（非一般 ROTATE）

candidates = []   # 可能被波動率分層停損「保住」的案例

for sym, exit_date, exit_sh, exit_px in exits:
    # 找最近一筆買入
    buys = sorted(buys_by_sym[sym], key=lambda x: x[0])
    relevant_buy = None
    for b_date, b_sh, b_px in reversed(buys):
        if b_date < exit_date:
            relevant_buy = (b_date, b_sh, b_px)
            break
    if not relevant_buy:
        continue

    buy_date, buy_sh, buy_px = relevant_buy
    pnl_pct = (exit_px - buy_px) / buy_px

    if pnl_pct > STOP_THRESHOLD:
        continue   # 獲利出場或輕微虧損（ROTATE），跳過

    # 計算進場時波動率
    vol = get_vol_at(sym, buy_date)
    std_fixed   = 0.15
    adj_fixed   = vol_adjusted_fixed(vol)
    std_stop_px = buy_px * (1 - std_fixed)
    adj_stop_px = buy_px * (1 - adj_fixed)

    # 判斷：波動率分層停損能否避免這次出場？
    saved = exit_px > adj_stop_px   # 出場價在調整後停損之上 → 若用更寬停損不會出場

    if vol is None or vol < VOL_TIER_LOW:
        vol_label = f"低 ({vol*100:.0f}%)" if vol else "N/A"
    elif vol < VOL_TIER_HIGH:
        vol_label = f"中 ({vol*100:.0f}%)"
    else:
        vol_label = f"高 ({vol*100:.0f}%)"

    candidates.append({
        "sym":         sym,
        "buy_date":    buy_date,
        "buy_px":      buy_px,
        "exit_date":   exit_date,
        "exit_px":     exit_px,
        "pnl_pct":     pnl_pct,
        "vol":         vol,
        "vol_label":   vol_label,
        "std_stop":    std_stop_px,
        "adj_stop":    adj_stop_px,
        "saved":       saved,
        "wider_stop":  adj_fixed > std_fixed,
    })

# ── 對「可能被保住」的案例，模擬持有到現在或再次停損 ───────────────────────
print(f"\n取得現在（或快取末尾）價格...")
saved_cases = [c for c in candidates if c["saved"] and c["wider_stop"]]
now_prices = {}
syms_need = list(set(c["sym"] for c in saved_cases))
if syms_need:
    try:
        hist = yf.download(syms_need, period="5d", auto_adjust=True, progress=False)["Close"]
        if hasattr(hist, "columns"):
            for s in syms_need:
                if s in hist.columns:
                    v = hist[s].dropna()
                    if len(v):
                        now_prices[s] = float(v.iloc[-1])
        elif len(syms_need) == 1:
            v = hist.dropna()
            if len(v):
                now_prices[syms_need[0]] = float(v.iloc[-1])
    except Exception as e:
        print(f"  yfinance 失敗: {e}")

# ── 輸出 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  反事實模擬：波動率分層停損 vs 標準停損（-15%）")
print(f"{'='*72}")
print(f"  分析的 EXIT 筆數（虧損 ≥-12%）：{len(candidates)}")
print(f"  其中波動率分層停損能避免的：{sum(1 for c in candidates if c['saved'] and c['wider_stop'])}\n")

print(f"  {'標的':<6}  {'進場':<11}  {'出場':<11}  {'損失':<7}  {'波動率':<10}  "
      f"{'標準停損':<8}  {'分層停損':<8}  {'能避免':^5}  {'避免後至今'}")
print(f"  {'─'*90}")

total_std_pnl = 0.0    # 實際結果：停損後收到現金
total_adj_pnl = 0.0    # 假設結果：繼續持有

for c in sorted(candidates, key=lambda x: x["buy_date"]):
    sym       = c["sym"]
    pnl_str   = f"{c['pnl_pct']*100:>+.1f}%"
    std_str   = f"${c['std_stop']:.2f}"
    adj_str   = f"${c['adj_stop']:.2f}" if c["wider_stop"] else "同標準"
    saved_str = "✅" if c["saved"] and c["wider_stop"] else "❌"

    # 若能避免：計算持有至今損益
    fwd_str   = ""
    if c["saved"] and c["wider_stop"]:
        now_px = now_prices.get(sym)
        exit_cost  = c["exit_px"] * c["buy_sh"] if "buy_sh" in c else c["exit_px"]
        if now_px:
            # 繼續持有到現在的 P&L（從買入價計算）
            hold_pct = (now_px - c["buy_px"]) / c["buy_px"]
            adj_pnl_diff = (now_px - c["exit_px"])   # per share 差值
            fwd_str = f"現價 ${now_px:.2f}  持有損益 {hold_pct*100:>+.1f}%"

            # 找實際買了幾股來計算總影響
            buy_sh = None
            for b_date, b_sh, b_px in buys_by_sym[sym]:
                if b_date == c["buy_date"] and abs(b_px - c["buy_px"]) < 0.02:
                    buy_sh = b_sh
                    break
            if buy_sh is None:
                buy_sh = 1  # 估算

            total_std_pnl += c["exit_px"] * buy_sh
            total_adj_pnl += now_px       * buy_sh
        else:
            fwd_str = "無法取得現價"

    print(f"  {sym:<6}  {c['buy_date']:<11}  {c['exit_date']:<11}  "
          f"{pnl_str:<7}  {c['vol_label']:<10}  {std_str:<8}  {adj_str:<8}  "
          f"{saved_str:^5}  {fwd_str}")

# ── 總影響 ───────────────────────────────────────────────────────────────────
diff = total_adj_pnl - total_std_pnl
print(f"\n{'─'*72}")
print(f"  「能避免」案例合計（現持有）：  ${total_adj_pnl:>10,.2f}")
print(f"  實際（停損後收到現金）合計：    ${total_std_pnl:>10,.2f}")
print(f"  差額（持有 vs 停損出場）：      ${diff:>+10,.2f}")
if diff > 0:
    print(f"\n  ✅ 若使用波動率分層停損，目前估計多賺 ${diff:,.2f}")
else:
    print(f"\n  ❌ 若使用波動率分層停損，目前估計少賺 ${abs(diff):,.2f}")

print(f"\n  ⚠️  注意：")
print(f"     1. 現金停在帳上可能被用於其他投資，此模擬忽略機會成本")
print(f"     2. 更寬的停損 = 每次停損時的損失更大，此模擬僅計算「避免停損」的案例")
print(f"     3. 若繼續持有最終也會觸發更寬的停損，需另行分析")
