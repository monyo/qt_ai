"""
_rotate_decel_equity_backtest.py

使用者質疑 _rotate_decel_backtest.py 的比較方式有問題：那份回測是把兩個策略
模擬出的「每一筆交易報酬率」全部丟進同一個池子比平均/中位數，但兩策略的
交易次數不同（A 49次 ROTATE、B 39次），這樣比較沒有反映複利效果，也不是
「一段時間下來的投組總報酬」。

這份腳本補上正確的比較方式：兩個策略（A=現行 ROTATE、B=新增賣出端自身動能
轉弱確認）各自從同一天、同一筆起始資金開始，逐日以「現金 + 持倉市值」
做權益曲線（equal-weight TOP 5、每月換倉、與現行 ROTATE/停損規則完全一致，
唯一差異是 B 多了自身動能轉弱檢查），跑完整段歷史後比較：
  總報酬% / CAGR% / MDD% / Calmar

執行：
    /home/mark/miniconda3/envs/qt_env/bin/python research/_rotate_decel_equity_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd

PRICES_PATH = "data/_protection_bt_prices.pkl"

print("載入資料...")
prices = pd.DataFrame(pd.read_pickle(PRICES_PATH))
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
close = prices.copy()

print("計算特徵...")
mom21 = close.pct_change(21)
mom252 = close.pct_change(252)
mom = 0.5 * mom21 + 0.5 * mom252
rolling_max40 = close.rolling(40).max()
rolling_min40 = close.rolling(40).min()
from_high_df = (close / rolling_max40 - 1) * 100
bounce_df = (close / rolling_min40 - 1) * 100

dates = close.index
rebal_dates = []
prev = None
for d in dates[252:]:
    m = (d.year, d.month)
    if m != prev:
        rebal_dates.append(d); prev = m

TOP_N = 5
MAX_HOLD = 63
STOP_FIX = -0.15
STOP_TRL = -0.25
BOUNCE_TH = 20.0
FH_TH = -15.0
GAP = 0.10
PROTECT_DAYS = 30
INITIAL_CASH = 100_000.0
# 注意：不套用 premarket.py 實際下單用的 0.85 現金安全係數——那是「留現金緩衝」的
# 執行面考量，兩策略都會一視同仁地拖累報酬，跟本次比較的重點（ROTATE 規則優劣）無關，
# 全押可以讓兩邊比較更乾淨。


def is_strong(ti, sym):
    bp = bounce_df.iloc[ti].get(sym, np.nan)
    fh = from_high_df.iloc[ti].get(sym, np.nan)
    return (not np.isnan(bp)) and (not np.isnan(fh)) and bp > BOUNCE_TH and fh > FH_TH


def run_equity_curve(require_decel=False, decel_lookback_months=2):
    """逐日模擬「現金 + 持倉市值」的完整權益曲線（equal-weight TOP 5，每月換倉）。

    require_decel=True 時，ROTATE 賣出前額外要求持股自身動能比
    decel_lookback_months 個月前更差；算不出來（新股/資料不足）則 fallback 不擋。
    """
    cash = INITIAL_CASH
    portfolio = {}  # sym -> {shares, ei, ep, hi}
    equity_dates = []
    equity_values = []

    def total_equity(ti):
        v = cash
        for sym, pos in portfolio.items():
            p = close.iloc[ti].get(sym, np.nan)
            if not np.isnan(p):
                v += pos["shares"] * p
        return v

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri + 1]
        ti = dates.get_loc(rd)
        mom_today = mom.iloc[ti]
        prev_ti = None
        if require_decel and ri >= decel_lookback_months:
            prev_rd = rebal_dates[ri - decel_lookback_months]
            prev_ti = dates.get_loc(prev_rd)

        ranked = mom_today.dropna().sort_values(ascending=False)
        held = set(portfolio)
        candidates = [s for s in ranked.index if s not in held and ranked[s] > 0]
        rot_out, rot_in = set(), set()

        # --- ROTATE ---
        for sym, pos in list(portfolio.items()):
            if sym in rot_out: continue
            if (rd - dates[pos["ei"]]).days < PROTECT_DAYS: continue
            if is_strong(ti, sym): continue
            pm = mom_today.get(sym, np.nan)
            if np.isnan(pm): continue
            if require_decel and prev_ti is not None:
                pm_prev = mom.iloc[prev_ti].get(sym, np.nan)
                if not np.isnan(pm_prev) and pm >= pm_prev:
                    continue
            for c in candidates:
                if c in rot_in: continue
                if ranked[c] - pm > GAP:
                    sp = close.iloc[ti].get(sym, np.nan)
                    cp = close.iloc[ti].get(c, np.nan)
                    if np.isnan(sp) or np.isnan(cp): break
                    cash += pos["shares"] * sp  # 全數賣出，回收現金
                    del portfolio[sym]
                    rot_out.add(sym); rot_in.add(c)
                    break  # 這個持倉已配對，換下一個；實際買入股數統一在下方按目標市值計算

        # --- 用目標權重統一買入：剛被 ROTATE 換入 + 補空位 ---
        # 直接用「目前 total_equity / TOP_N」當每檔目標市值，依序買滿到 TOP_N 檔
        target_val = total_equity(ti) / TOP_N
        # 先買 ROTATE 換入的候選
        for c in list(rot_in):
            cp = close.iloc[ti].get(c, np.nan)
            if np.isnan(cp) or cp <= 0: continue
            spend = min(target_val, cash)
            shares = math.floor(spend / cp)
            if shares <= 0: continue
            cash -= shares * cp
            portfolio[c] = {"shares": shares, "ei": ti, "ep": cp, "hi": cp}
        # 再補滿空位（初始建倉 / 停損或到期出場後留下的空位）
        held = set(portfolio)
        for sym in ranked.index:
            if len(portfolio) >= TOP_N: break
            if sym in held: continue
            ep = close.iloc[ti].get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue
            spend = min(target_val, cash)
            shares = math.floor(spend / ep)
            if shares <= 0: continue
            cash -= shares * ep
            portfolio[sym] = {"shares": shares, "ei": ti, "ep": ep, "hi": ep}
            held.add(sym)

        equity_dates.append(rd); equity_values.append(total_equity(ti))

        # --- 逐日檢查停損 / 到期，並記錄每日權益 ---
        next_ti = dates.get_loc(next_rd)
        to_rm = []
        for di in range(ti + 1, next_ti + 1):
            for sym, pos in list(portfolio.items()):
                if sym in to_rm: continue
                p = close.iloc[di].get(sym, np.nan)
                if np.isnan(p): continue
                pos["hi"] = max(pos["hi"], p)
                st = max(pos["ep"] * (1 + STOP_FIX), pos["hi"] * (1 + STOP_TRL))
                if p <= st:
                    cash += pos["shares"] * p
                    to_rm.append(sym)
            for sym in to_rm:
                portfolio.pop(sym, None)
            equity_dates.append(dates[di]); equity_values.append(total_equity(di))

        # --- 到期強制出場（MAX_HOLD 天），在下個換倉日執行 ---
        for sym, pos in list(portfolio.items()):
            if (next_rd - dates[pos["ei"]]).days >= MAX_HOLD:
                p = close.iloc[next_ti].get(sym, np.nan)
                if not np.isnan(p):
                    cash += pos["shares"] * p
                    del portfolio[sym]

    return pd.Series(equity_values, index=pd.to_datetime(equity_dates)).sort_index()


def metrics(equity: pd.Series):
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    running_max = equity.cummax()
    dd = equity / running_max - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return {
        "總報酬%": total_ret * 100, "CAGR%": cagr * 100,
        "MDD%": mdd * 100, "Calmar": calmar,
        "期末權益": equity.iloc[-1],
    }


print("執行 A（現行 ROTATE，無自身轉弱檢查）...")
eq_a = run_equity_curve(require_decel=False)
print("執行 B（新增賣出端自身動能轉弱確認，2個月回看）...")
eq_b = run_equity_curve(require_decel=True, decel_lookback_months=2)

m_a = metrics(eq_a)
m_b = metrics(eq_b)

print(f"""
{'='*72}
  ROTATE 策略權益曲線比較（A=現行 vs B=新增賣出端自身轉弱確認）
  起始資金 ${INITIAL_CASH:,.0f}，{eq_a.index[0].date()} ~ {eq_a.index[-1].date()}
{'='*72}
  策略                總報酬        CAGR      MDD      Calmar     期末權益
  {'-'*68}
  A 現行            {m_a['總報酬%']:>+8.1f}%  {m_a['CAGR%']:>+6.1f}%  {m_a['MDD%']:>6.1f}%  {m_a['Calmar']:>7.3f}  ${m_a['期末權益']:>12,.0f}
  B 賣出端轉弱確認   {m_b['總報酬%']:>+8.1f}%  {m_b['CAGR%']:>+6.1f}%  {m_b['MDD%']:>6.1f}%  {m_b['Calmar']:>7.3f}  ${m_b['期末權益']:>12,.0f}

  差異：總報酬 {m_b['總報酬%']-m_a['總報酬%']:+.1f}pp　CAGR {m_b['CAGR%']-m_a['CAGR%']:+.1f}pp　MDD {m_b['MDD%']-m_a['MDD%']:+.1f}pp　Calmar {m_b['Calmar']-m_a['Calmar']:+.3f}

  ⚠️  Survivorship bias：樣本為 S&P500「現有」成分股，不含交易成本/滑價，
      全額投入（無現金安全係數），僅供兩策略相對比較參考，非絕對報酬預期。
""")
