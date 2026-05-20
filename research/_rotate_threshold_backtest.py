"""
_rotate_threshold_backtest.py

問題：ROTATE「轉強保護」的距高點門檻應該設在哪？

現行系統：bounce > 20% AND from_high > -5% → 轉強 → 不能被 ROTATE
SNDK 案例：from_high = -5.1%，差 0.1% 就被 ROTATE，隔天 +8.4%

比較策略（月度動能 top-5，持有最多 63 天）：
  A  -5%   現行系統（幾乎不保護，只有在高點附近才保護）
  B  -8%   稍寬（SNDK 那種情況不會被 ROTATE）
  C  -10%  中等寬鬆
  D  -15%  較寬（距高點 15% 以內都保護）
  E  無保護  不管趨勢狀態，只要動能差 >10% 就 ROTATE

ROTATE 觸發條件（除門檻外其餘固定）：
  - 持有 > 30 天
  - 候選動能 - 持倉動能 > 10%
  - 候選不在持倉中

執行：
    conda run -n qt_env python research/_rotate_threshold_backtest.py
"""

import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
PRICES_PATH = "data/_protection_bt_prices.pkl"

if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}"); sys.exit(1)

print("載入資料...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
prices   = pd.read_pickle(PRICES_PATH)

df_close = ohlcv["Close"]
df_close.index = pd.to_datetime(df_close.index)
if df_close.index.tz is not None:
    df_close.index = df_close.index.tz_localize(None)

prices = pd.DataFrame(prices)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)

# 共用 close（以 prices 為主，ohlcv 補充）
close = prices.copy()

# ── 預先計算特徵 ─────────────────────────────────────────────────────────────
print("計算動能與趨勢特徵...")
mom21  = close.pct_change(21)
mom252 = close.pct_change(252)
mom    = 0.5 * mom21 + 0.5 * mom252   # 混合動能

rolling_max40 = close.rolling(40).max()
rolling_min40 = close.rolling(40).min()
from_high_df  = (close / rolling_max40 - 1) * 100
bounce_pct_df = (close / rolling_min40 - 1) * 100

# 交易日索引
dates = close.index
syms  = close.columns.tolist()

# ── 月度再平衡點 ─────────────────────────────────────────────────────────────
rebal_dates = []
prev_month  = None
for d in dates[252:]:
    m = (d.year, d.month)
    if m != prev_month:
        rebal_dates.append(d)
        prev_month = m

print(f"  再平衡點：{len(rebal_dates)} 個  ({rebal_dates[0].date()} ~ {rebal_dates[-1].date()})")

# ── 回測函數 ─────────────────────────────────────────────────────────────────
PROTECT_DAYS   = 30    # 最少持有天數才可 ROTATE
MOM_GAP        = 0.10  # 動能差距門檻（小數，0.10 = 10%）
BOUNCE_THRESH  = 20.0  # 轉強：反彈幅度門檻
TOP_N          = 5     # 持倉槽位
MAX_HOLD       = 63    # 最長持有天數
STOP_FIXED     = -0.15 # 固定停損
STOP_TRAIL     = -0.25 # 追蹤停損

def is_转强(ti, sym, from_high_thresh):
    """判斷是否為轉強狀態（受 ROTATE 保護）"""
    bp = bounce_pct_df.iloc[ti].get(sym, np.nan)
    fh = from_high_df.iloc[ti].get(sym, np.nan)
    if np.isnan(bp) or np.isnan(fh):
        return False
    return bp > BOUNCE_THRESH and fh > from_high_thresh  # from_high > -X%

def run_backtest(from_high_thresh, label):
    """from_high_thresh: 轉強保護門檻，如 -5 表示距高點 > -5% 才算轉強"""
    portfolio = {}  # sym -> {entry_ti, entry_price, high}
    trades    = []

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri + 1]
        ti      = dates.get_loc(rd)

        # 當前動能排名
        mom_today = mom.iloc[ti]
        ranked    = mom_today.dropna().sort_values(ascending=False)
        held_syms = set(portfolio.keys())

        # ── ROTATE 檢查（持有 > PROTECT_DAYS，非轉強） ────────────────────
        # 非持倉候選（動能 > 0）
        candidates = [s for s in ranked.index
                      if s not in held_syms and ranked[s] > 0]

        rotated_out = set()
        rotated_in  = set()

        for sym, pos in list(portfolio.items()):
            if sym in rotated_out:
                continue
            hold_days = (rd - dates[pos["entry_ti"]]).days
            if hold_days < PROTECT_DAYS:
                continue
            if is_转强(ti, sym, from_high_thresh):
                continue  # 受保護，跳過

            pos_mom = mom_today.get(sym, np.nan)
            if np.isnan(pos_mom):
                continue

            # 找第一個夠強的候選
            for cand in candidates:
                if cand in rotated_in:
                    continue
                cand_mom = ranked[cand]
                if cand_mom - pos_mom > MOM_GAP:
                    # 執行 ROTATE
                    sell_price  = close.iloc[ti].get(sym, np.nan)
                    buy_price   = close.iloc[ti].get(cand, np.nan)
                    entry_price = close.iloc[pos["entry_ti"]].get(sym, np.nan)
                    if np.isnan(sell_price) or np.isnan(buy_price) or np.isnan(entry_price):
                        break
                    ret = (sell_price - entry_price) / entry_price
                    trades.append({"ret": ret, "type": "rotate_out", "sym": sym})
                    del portfolio[sym]
                    portfolio[cand] = {"entry_ti": ti, "entry_price": buy_price, "high": buy_price}
                    rotated_out.add(sym)
                    rotated_in.add(cand)
                    break

        held_syms = set(portfolio.keys())

        # ── 正常月度再平衡（補到 TOP_N 槽） ──────────────────────────────
        if len(portfolio) < TOP_N:
            for sym in ranked.index:
                if len(portfolio) >= TOP_N:
                    break
                if sym in held_syms:
                    continue
                if sym in rotated_in:
                    continue
                ep = close.iloc[ti].get(sym, np.nan)
                if np.isnan(ep) or ep <= 0:
                    continue
                portfolio[sym] = {"entry_ti": ti, "entry_price": ep, "high": ep}
                held_syms.add(sym)

        # ── 持倉更新至下次再平衡 ─────────────────────────────────────────
        next_ti = dates.get_loc(next_rd)
        to_remove = []
        for sym, pos in portfolio.items():
            for di in range(ti + 1, next_ti + 1):
                p = close.iloc[di].get(sym, np.nan)
                if np.isnan(p):
                    continue
                pos["high"] = max(pos["high"], p)
                # 停損檢查
                fixed_stop   = pos["entry_price"] * (1 + STOP_FIXED)
                trail_stop   = pos["high"] * (1 + STOP_TRAIL)
                stop_price   = max(fixed_stop, trail_stop)
                if p <= stop_price:
                    ret = (p - pos["entry_price"]) / pos["entry_price"]
                    trades.append({"ret": ret, "type": "stop", "sym": sym})
                    to_remove.append((sym, di))
                    break
            else:
                # 時間到期出場
                hold_days = (next_rd - dates[pos["entry_ti"]]).days
                if hold_days >= MAX_HOLD:
                    ep = close.iloc[next_ti].get(sym, np.nan)
                    if not np.isnan(ep):
                        ret = (ep - pos["entry_price"]) / pos["entry_price"]
                        trades.append({"ret": ret, "type": "time", "sym": sym})
                        to_remove.append((sym, next_ti))

        for sym, _ in to_remove:
            portfolio.pop(sym, None)

    if not trades:
        return None

    rets  = [t["ret"] for t in trades]
    n     = len(rets)
    avg   = np.mean(rets) * 100
    med   = np.median(rets) * 100
    win   = sum(1 for r in rets if r > 0) / n * 100
    stops = sum(1 for t in trades if t["type"] == "stop")
    rots  = sum(1 for t in trades if t["type"] == "rotate_out")
    p10   = np.percentile(rets, 10) * 100
    p90   = np.percentile(rets, 90) * 100
    return {
        "label": label,
        "n": n, "avg": avg, "med": med, "win": win,
        "stops": stops, "rots": rots, "p10": p10, "p90": p90
    }

# ── 執行各策略 ────────────────────────────────────────────────────────────────
print("\n執行回測...")
strategies = [
    (-5,   "A  -5%  現行"),
    (-8,   "B  -8%  稍寬"),
    (-10,  "C  -10% 中寬"),
    (-15,  "D  -15% 較寬"),
    (-999, "E  無保護"),
]

results = []
for thresh, label in strategies:
    r = run_backtest(thresh, label)
    if r:
        results.append(r)
        print(f"  {label} 完成（{r['n']} 筆交易，ROTATE {r['rots']} 次）")

# ── 輸出結果 ─────────────────────────────────────────────────────────────────
print(f"""
{'='*70}
  ROTATE 轉強保護門檻回測
{'='*70}
  策略               N     平均報酬  中位報酬   勝率   停損  ROTATE  P10     P90
  {'-'*68}""")

for r in results:
    mark = " ← 現行" if "現行" in r["label"] else ""
    print(f"  {r['label']:<18} {r['n']:>4}  {r['avg']:>+7.2f}%  {r['med']:>+7.2f}%  "
          f"{r['win']:>5.1f}%  {r['stops']:>4}  {r['rots']:>6}  "
          f"{r['p10']:>+6.1f}%  {r['p90']:>+6.1f}%{mark}")

print(f"""
  說明：
  - 轉強門檻越寬 → ROTATE 越少觸發 → 持倉翻動率降低
  - 中位報酬與勝率是最重要的指標（平均易被極端值拉偏）
  - ROTATE 次數越少不代表越好，關鍵是品質

  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股
  ⚠️  不含交易成本、滑點
""")
