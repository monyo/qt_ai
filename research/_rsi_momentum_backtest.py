"""
RSI 超買 vs 正常 — 動能策略條件報酬回測

在動能前 N 名候選股中，比較：
  A. RSI ≤ 75（正常）
  B. RSI 75-80（輕微超買）
  C. RSI > 80（極度超買）
的未來 1 個月 alpha（vs SPY）

資料：S&P500 2021-2025（_protection_bt_prices.pkl）
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

CACHE_PATH  = 'data/_protection_bt_prices.pkl'
MOM_SHORT   = 21
MOM_LONG    = 252
RSI_PERIOD  = 14
TOP_N       = 30        # 動能前 N 名作為候選池
REBAL_FREQ  = 21        # 月度再平衡
FWD_DAYS    = 21        # 未來觀察期

# ── 載入資料 ──────────────────────────────────────────────────
print(f'載入 {CACHE_PATH}...')
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index).tz_localize(None)
prices = prices.sort_index()

# 下載 SPY
import yfinance as yf
spy_raw = yf.download('SPY', start='2020-01-01', end='2026-04-20',
                      auto_adjust=True, progress=False)['Close']
if isinstance(spy_raw, pd.DataFrame):
    spy_raw = spy_raw.iloc[:, 0]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

trading_days = prices.index
n_dates      = len(trading_days)
symbols      = list(prices.columns)
close_arr    = prices.values.astype(float)
sym_idx      = {s: i for i, s in enumerate(symbols)}

START_TI  = MOM_LONG + RSI_PERIOD + 5
rebal_tis = list(range(START_TI, n_dates - FWD_DAYS - 5, REBAL_FREQ))
print(f'資料：{len(symbols)} 支 / {n_dates} 日（{trading_days[0].date()} ~ {trading_days[-1].date()}）')
print(f'再平衡點：{len(rebal_tis)} 個\n')


# ── 工具函式 ──────────────────────────────────────────────────
def get_price(ti, si):
    if 0 <= ti < n_dates:
        v = close_arr[ti, si]
        return float(v) if np.isfinite(v) else None
    return None


def calc_rsi(ti, si, period=RSI_PERIOD):
    """計算 RSI（14日）"""
    if ti < period + 1:
        return None
    closes = [close_arr[ti - period + k, si] for k in range(period + 1)]
    closes = [c for c in closes if np.isfinite(c)]
    if len(closes) < period:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def calc_momentum(ti, si):
    p0  = get_price(ti, si)
    p21 = get_price(ti - MOM_SHORT, si)
    p252= get_price(ti - MOM_LONG, si)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1) + 0.5 * (p0/p252 - 1)


def spy_fwd(ti, days=FWD_DAYS):
    ref = trading_days[ti]
    idx = spy_raw.index.searchsorted(ref)
    if idx + days >= len(spy_raw):
        return None
    return float(spy_raw.iloc[idx + days]) / float(spy_raw.iloc[idx]) - 1


def stock_fwd(ti, si, days=FWD_DAYS):
    p0 = get_price(ti, si)
    pf = get_price(ti + days, si)
    if not (p0 and pf and p0 > 0):
        return None
    return pf / p0 - 1


# ── 主回測 ────────────────────────────────────────────────────
print('開始回測...')

records = []   # {ti, symbol, rsi, momentum, fwd_ret, spy_ret, alpha}

for ti in rebal_tis:
    # 計算全市場動能，取前 TOP_N
    scores = {}
    for sym in symbols:
        si = sym_idx[sym]
        m = calc_momentum(ti, si)
        if m is not None:
            scores[sym] = m
    top_syms = sorted(scores, key=scores.get, reverse=True)[:TOP_N]

    spy_r = spy_fwd(ti)
    if spy_r is None:
        continue

    for sym in top_syms:
        si  = sym_idx[sym]
        rsi = calc_rsi(ti, si)
        mom = scores[sym]
        fwd = stock_fwd(ti, si)
        if rsi is None or fwd is None:
            continue
        alpha = fwd - spy_r
        records.append({
            'ti': ti, 'date': trading_days[ti].date(),
            'symbol': sym, 'rsi': rsi,
            'momentum': mom * 100,
            'fwd_ret': fwd * 100,
            'spy_ret': spy_r * 100,
            'alpha': alpha * 100,
        })

df = pd.DataFrame(records)
print(f'有效樣本：{len(df)} 筆\n')


# ── 分組分析 ──────────────────────────────────────────────────
def group_stats(label, mask):
    sub = df[mask]
    n   = len(sub)
    if n == 0:
        return
    mean_alpha = sub['alpha'].mean()
    med_alpha  = sub['alpha'].median()
    win_rate   = (sub['alpha'] > 0).mean() * 100
    mean_fwd   = sub['fwd_ret'].mean()
    print(f'  {label:<20} n={n:>5}  alpha 均值 {mean_alpha:>+6.2f}%  中位 {med_alpha:>+6.2f}%  '
          f'勝率 {win_rate:>5.1f}%  絕對報酬 {mean_fwd:>+6.2f}%')

print('=' * 72)
print('  RSI 分組 vs SPY Alpha（動能前30名候選股，21日前瞻）')
print('=' * 72)
group_stats('全部候選股',    df['rsi'] >= 0)
print(f'  {"─"*68}')
group_stats('RSI ≤ 60',      df['rsi'] <= 60)
group_stats('RSI 60-70',     (df['rsi'] > 60) & (df['rsi'] <= 70))
group_stats('RSI 70-75',     (df['rsi'] > 70) & (df['rsi'] <= 75))
group_stats('RSI 75-80',     (df['rsi'] > 75) & (df['rsi'] <= 80))
group_stats('RSI 80-85',     (df['rsi'] > 80) & (df['rsi'] <= 85))
group_stats('RSI > 85',      df['rsi'] > 85)
print()
group_stats('RSI ≤ 75（正常）', df['rsi'] <= 75)
group_stats('RSI > 75（超買）', df['rsi'] > 75)
group_stats('RSI > 80（極超買）', df['rsi'] > 80)

# ── 月度趨勢：超買 vs 正常的 alpha 差 ────────────────────────
print()
print('=' * 72)
print('  按年度分析')
print('=' * 72)
df['year'] = pd.to_datetime(df['date']).dt.year
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    norm  = sub[sub['rsi'] <= 75]['alpha'].mean()
    overbought = sub[sub['rsi'] > 75]['alpha'].mean()
    diff  = overbought - norm if not (np.isnan(norm) or np.isnan(overbought)) else float('nan')
    n_norm = (sub['rsi'] <= 75).sum()
    n_ob   = (sub['rsi'] > 75).sum()
    print(f'  {yr}  正常(n={n_norm:>3}) {norm:>+6.2f}%  超買(n={n_ob:>3}) {overbought:>+6.2f}%  差 {diff:>+6.2f}%')

print()
print('  ※ alpha = 個股報酬 − SPY 同期報酬')
print('  ※ 候選池：每月動能前 30 名（S&P500）')
