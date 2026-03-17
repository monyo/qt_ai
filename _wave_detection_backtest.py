"""
浪偵測回測 v2（加入交易量 + 型態信號）

地面真相：偵測點往後 6 個月，alpha vs SPY > +20%
候選池：當下排名 21-150

偵測信號：
  ── 純技術（v1）──
  A. 排名速度：排名本月大幅上升（>30名），且現在在 21-80
  B. MA60 剛突破：10天內剛穿越 MA60 且 MA60 斜率轉正
  C. 低波動突破：40日震盪幅度 < 15%，近5天開始向上

  ── 交易量 + 型態（v2 新增）──
  E. OBV 蓄積：股價橫盤（<8%），但 OBV 持續上升（水位無聲上升）
  F. 量縮後量增突破：整理縮量，突破時放量 + 價格上漲
  G. 底部墊高：60 天內低點每次比前次高，顯示支撐越來越強

  D. 複合 2+（純技術）
  H. 複合 2+（含量能，任意 2 個以上）
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from src.data_loader import get_sp500_tickers

PRICE_CACHE  = "data/_protection_bt_prices.pkl"
VOL_CACHE    = "data/_wave_detect_volume.pkl"
DATA_START   = "2019-06-01"
DATA_END     = "2026-03-01"
BATCH_SIZE   = 50

MOM_SHORT    = 21
MOM_LONG     = 252
TOP_SYSTEM   = 20
RANK_MAX     = 150
FORWARD_DAYS = 126    # 6M
ALPHA_THRESH = 20.0

# ── 載入價格快取 ──────────────────────────────────────────────────────
if not os.path.exists(PRICE_CACHE):
    print(f"找不到 {PRICE_CACHE}，請先執行 _protection_period_backtest.py")
    raise SystemExit(1)

print(f"讀取價格快取：{PRICE_CACHE}")
prices = pd.read_pickle(PRICE_CACHE)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"價格：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日")

# ── 下載 / 載入交易量快取 ─────────────────────────────────────────────
if os.path.exists(VOL_CACHE):
    print(f"讀取交易量快取：{VOL_CACHE}")
    volumes = pd.read_pickle(VOL_CACHE)
else:
    tickers = get_sp500_tickers()
    print(f"分批下載交易量 {len(tickers)} 檔（每批 {BATCH_SIZE}）...")
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_vols = []
    for i, batch in enumerate(batches):
        print(f"  批次 {i+1}/{len(batches)}...", end=" ", flush=True)
        try:
            raw = yf.download(batch, start=DATA_START, end=DATA_END,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print("空"); continue
            vol = raw["Volume"]
            if isinstance(vol.columns, pd.MultiIndex):
                vol.columns = vol.columns.droplevel(0)
            all_vols.append(vol)
            print(f"OK ({vol.shape[1]} 檔)")
        except Exception as e:
            print(f"ERROR: {e}")
    volumes = pd.concat(all_vols, axis=1)
    volumes = volumes.loc[:, ~volumes.columns.duplicated()]
    volumes.index = pd.to_datetime(volumes.index, utc=True).tz_convert(None)
    os.makedirs("data", exist_ok=True)
    volumes.to_pickle(VOL_CACHE)
    print(f"已快取至 {VOL_CACHE}")

volumes.index = pd.to_datetime(volumes.index)
if volumes.index.tz is not None:
    volumes.index = volumes.index.tz_convert(None)
# 對齊到共同交易日
volumes = volumes.reindex(trading_days).ffill()
print(f"交易量：{volumes.shape[1]} 檔\n")

# SPY
spy_raw = yf.Ticker("SPY").history(start=DATA_START, end=DATA_END,
                                    auto_adjust=True)["Close"]
spy = spy_raw.copy()
spy.index = pd.to_datetime(spy.index).tz_localize(None)


# ── 基礎工具 ─────────────────────────────────────────────────────────
def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

def px(sym, ti):
    if ti < 0 or ti >= len(trading_days): return None
    v = prices.iloc[ti].get(sym)
    return float(v) if v is not None and pd.notna(v) else None

def vl(sym, ti):
    """交易量"""
    if ti < 0 or ti >= len(trading_days): return None
    if sym not in volumes.columns: return None
    v = volumes.iloc[ti].get(sym)
    return float(v) if v is not None and pd.notna(v) and v > 0 else None

def spy_px(ti):
    if ti < 0 or ti >= len(trading_days): return None
    d = trading_days[ti]
    si = spy.index.searchsorted(d)
    return float(spy.iloc[si]) if si < len(spy) else None

def mom_mixed(sym, ti):
    p0, p21, p252 = px(sym, ti), px(sym, ti-MOM_SHORT), px(sym, ti-MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0): return None
    return 0.5*(p0/p21-1)*100 + 0.5*(p0/p252-1)*100

def compute_ranks(ti):
    scores = {sym: m for sym in prices.columns
              if (m := mom_mixed(sym, ti)) is not None and m > 0}
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {sym: i+1 for i, sym in enumerate(ranked)}

def fwd_alpha(sym, ti, days=FORWARD_DAYS):
    p0s, p1s = px(sym, ti), px(sym, ti+days)
    p0b, p1b = spy_px(ti), spy_px(ti+days)
    if not (p0s and p1s and p0b and p1b and p0s > 0 and p0b > 0): return None
    return (p1s/p0s - p1b/p0b) * 100


# ── 偵測信號（純技術，v1）────────────────────────────────────────────
def sig_rank_velocity(rank_now, rank_prev):
    if rank_now is None or rank_prev is None: return False
    return rank_now <= 80 and (rank_prev - rank_now) >= 30

def sig_ma60_crossover(sym, ti):
    p_now = px(sym, ti)
    if not p_now: return False
    ma_now_vals = [px(sym, ti-k) for k in range(60)]
    ma_now_vals = [v for v in ma_now_vals if v]
    if len(ma_now_vals) < 50: return False
    ma_now = np.mean(ma_now_vals)
    if p_now < ma_now: return False
    ma_old_vals = [px(sym, ti-10-k) for k in range(60)]
    ma_old_vals = [v for v in ma_old_vals if v]
    if len(ma_old_vals) < 50: return False
    if np.mean(ma_now_vals) <= np.mean(ma_old_vals): return False
    for back in range(1, 11):
        p_b = px(sym, ti-back)
        mb_vals = [px(sym, ti-back-k) for k in range(60)]
        mb_vals = [v for v in mb_vals if v]
        if p_b and len(mb_vals) >= 50 and p_b < np.mean(mb_vals):
            return True
    return False

def sig_low_vol_price_breakout(sym, ti):
    """低波動整理 + 價格突破（不看量）"""
    consol = [px(sym, ti-k) for k in range(40)]
    consol = [v for v in consol if v]
    if len(consol) < 30: return False
    p_now = px(sym, ti)
    if not p_now or p_now <= 0: return False
    if (max(consol)-min(consol))/p_now >= 0.15: return False
    p_ago = px(sym, ti-5)
    return bool(p_ago and p_ago > 0 and (p_now/p_ago - 1) >= 0.03)


# ── 偵測信號（交易量 + 型態，v2）────────────────────────────────────
def sig_obv_accumulation(sym, ti, window=60):
    """
    OBV 蓄積：股價橫盤（變動 < 8%）但 OBV 持續上升
    意義：有人在安靜地累積（水位無聲上升）
    """
    obv = 0.0
    obv_start = None
    obv_end   = None
    for d in range(window, -1, -1):
        idx = ti - d
        p_c  = px(sym, idx)
        p_p  = px(sym, idx - 1)
        v    = vl(sym, idx)
        if not (p_c and p_p and v): continue
        obv += v if p_c > p_p else (-v if p_c < p_p else 0)
        if obv_start is None: obv_start = obv
        obv_end = obv
    if obv_start is None or obv_end is None: return False
    # OBV 上升幅度：用絕對量的百分比衡量
    avg_vol = np.mean([vl(sym, ti-k) or 0 for k in range(window)])
    if avg_vol <= 0: return False
    obv_rise = (obv_end - obv_start) / (avg_vol * window)  # 標準化
    # 股價幾乎沒動
    p_start = px(sym, ti - window)
    p_end   = px(sym, ti)
    if not (p_start and p_end and p_start > 0): return False
    price_change = abs(p_end / p_start - 1)
    return obv_rise > 0.05 and price_change < 0.08

def sig_vol_dry_breakout(sym, ti, base_win=60, dry_win=15, surge_win=5):
    """
    量縮後量增突破：整理期縮量 → 突破時放量 + 價格向上
    意義：能量蓄積後釋放（浪開始起）
    """
    base_vols = [vl(sym, ti-k) for k in range(base_win)]
    base_vols = [v for v in base_vols if v]
    if len(base_vols) < base_win // 2: return False
    avg_vol = np.mean(base_vols)
    if avg_vol <= 0: return False
    # 乾燥期（surge_win 之前的 dry_win 天）
    dry_vols = [vl(sym, ti-surge_win-k) for k in range(dry_win)]
    dry_vols = [v for v in dry_vols if v]
    if not dry_vols: return False
    # 突破期（最近 surge_win 天）
    surge_vols = [vl(sym, ti-k) for k in range(surge_win)]
    surge_vols = [v for v in surge_vols if v]
    if not surge_vols: return False
    # 價格上漲
    p_now = px(sym, ti)
    p_ago = px(sym, ti - surge_win)
    if not (p_now and p_ago and p_ago > 0): return False
    return (np.mean(dry_vols) / avg_vol < 0.65 and       # 乾燥期縮量
            np.mean(surge_vols) / avg_vol > 1.30 and     # 突破期放量
            (p_now / p_ago - 1) > 0.02)                  # 價格上漲 >2%

def sig_higher_lows(sym, ti, window=60, segments=4):
    """
    底部墊高：60天內每次回調低點比前次高
    意義：買盤越來越早介入，支撐墊高（浪的底部在抬升）
    """
    seg_len = window // segments
    lows = []
    for i in range(segments):
        start = ti - window + i * seg_len
        end   = start + seg_len
        seg_prices = [px(sym, k) for k in range(start, end)]
        seg_prices = [v for v in seg_prices if v]
        if seg_prices:
            lows.append(min(seg_prices))
    if len(lows) < 3: return False
    return all(lows[i] < lows[i+1] for i in range(len(lows)-1))


# ── 主回測循環 ────────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2020-06-01", "2024-09-01", freq="BMS")
records = []

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG + 30: continue
    if ti + FORWARD_DAYS >= len(trading_days): continue

    ranks_now  = compute_ranks(ti)
    ranks_prev = compute_ranks(ti - MOM_SHORT)

    for sym, rank_now in ranks_now.items():
        if rank_now <= TOP_SYSTEM or rank_now > RANK_MAX: continue
        alpha = fwd_alpha(sym, ti)
        if alpha is None: continue

        rank_prev = ranks_prev.get(sym)
        sA = sig_rank_velocity(rank_now, rank_prev)
        sB = sig_ma60_crossover(sym, ti)
        sC = sig_low_vol_price_breakout(sym, ti)
        sE = sig_obv_accumulation(sym, ti)
        sF = sig_vol_dry_breakout(sym, ti)
        sG = sig_higher_lows(sym, ti)
        sD = sum([sA, sB, sC]) >= 2
        sH = sum([sA, sB, sC, sE, sF, sG]) >= 2

        records.append({
            "date": str(ref.date()), "sym": sym,
            "rank_now": rank_now, "rank_prev": rank_prev,
            "alpha_6m": alpha,
            "is_good_wave": alpha >= ALPHA_THRESH,
            "sig_A": sA, "sig_B": sB, "sig_C": sC,
            "sig_D": sD,
            "sig_E": sE, "sig_F": sF, "sig_G": sG,
            "sig_H": sH,
        })

    print(f"  {ref.date()}", end="\r", flush=True)

print()
df = pd.DataFrame(records)
print(f"總候選事件：{len(df)}  好浪：{df['is_good_wave'].sum()}  "
      f"基準率：{df['is_good_wave'].mean()*100:.1f}%\n")


# ── 評估 ─────────────────────────────────────────────────────────────
def evaluate(df, sig_col, label):
    det      = df[df[sig_col]]
    all_good = df[df["is_good_wave"]]
    tp = det["is_good_wave"].sum()
    fp = len(det) - tp
    fn = len(all_good) - tp
    prec   = tp / len(det) * 100 if len(det) else 0
    recall = tp / len(all_good) * 100 if len(all_good) else 0
    f1     = 2*prec*recall/(prec+recall) if (prec+recall) else 0
    tp_alpha = det[det["is_good_wave"]]["alpha_6m"].mean() if tp else None
    fp_alpha = det[~det["is_good_wave"]]["alpha_6m"].mean() if fp else None
    # 期望值 = prec × tp_alpha + (1-prec) × fp_alpha（簡化版）
    ev = (prec/100)*(tp_alpha or 0) + (1-prec/100)*(fp_alpha or 0) if tp_alpha and fp_alpha else None
    return dict(label=label, n=len(det), tp=tp, fp=fp, fn=fn,
                prec=prec, recall=recall, f1=f1,
                tp_alpha=tp_alpha, fp_alpha=fp_alpha, ev=ev)

base_rate = df["is_good_wave"].mean() * 100
sigs = [
    ("sig_A", "A. 排名速度"),
    ("sig_B", "B. MA60 突破"),
    ("sig_C", "C. 低波動突破"),
    ("sig_D", "D. 複合技術 2+"),
    ("sig_E", "E. OBV 蓄積"),
    ("sig_F", "F. 量縮後量增突破"),
    ("sig_G", "G. 底部墊高"),
    ("sig_H", "H. 複合含量能 2+"),
]

print("=" * 72)
print(f"  浪偵測回測 v2  2020-2024  S&P500  基準率：{base_rate:.1f}%")
print("=" * 72)

results = [evaluate(df, sc, lb) for sc, lb in sigs]

print(f"\n  {'信號':<22} {'偵測數':>6} {'精準度':>8} {'召回率':>8} {'F1':>6}  "
      f"{'命中alpha':>10}  {'誤判alpha':>10}  {'期望值':>8}")
print("  " + "-" * 82)
for r in results:
    tag = " ★" if r["prec"] > base_rate + 2 else ""
    ta  = f"{r['tp_alpha']:+.1f}%" if r["tp_alpha"] else "  N/A"
    fa  = f"{r['fp_alpha']:+.1f}%" if r["fp_alpha"] else "  N/A"
    ev  = f"{r['ev']:+.1f}%" if r["ev"] else "  N/A"
    print(f"  {r['label']:<22} {r['n']:>6} {r['prec']:>7.1f}%  "
          f"{r['recall']:>7.1f}%  {r['f1']:>5.1f}%  {ta:>10}  {fa:>10}  {ev:>8}{tag}")

# ── 量能信號的「浪前形態」特寫 ────────────────────────────────────────
print()
print("=" * 72)
print("  量能信號與好浪率（OBV蓄積 × 底部墊高 交叉分析）")
print()
for sE_val, sG_val in [(True, True), (True, False), (False, True), (False, False)]:
    mask = (df["sig_E"] == sE_val) & (df["sig_G"] == sG_val)
    sub  = df[mask]
    if len(sub) == 0: continue
    rate = sub["is_good_wave"].mean() * 100
    avg_alpha = sub["alpha_6m"].mean()
    print(f"  OBV蓄積={str(sE_val):<5}  底部墊高={str(sG_val):<5}  "
          f"n={len(sub):4d}  好浪率={rate:5.1f}%  平均alpha={avg_alpha:+.1f}%")

# ── 量增突破 × 底部墊高 ─────────────────────────────────────────────
print()
print("  量增突破 × 底部墊高")
for sF_val, sG_val in [(True, True), (True, False), (False, True), (False, False)]:
    mask = (df["sig_F"] == sF_val) & (df["sig_G"] == sG_val)
    sub  = df[mask]
    if len(sub) == 0: continue
    rate = sub["is_good_wave"].mean() * 100
    avg_alpha = sub["alpha_6m"].mean()
    print(f"  量增突破={str(sF_val):<5}  底部墊高={str(sG_val):<5}  "
          f"n={len(sub):4d}  好浪率={rate:5.1f}%  平均alpha={avg_alpha:+.1f}%")

# ── 不同門檻下各信號精準度 ───────────────────────────────────────────
print()
print("=" * 72)
print("  好浪門檻敏感度（精準度 vs alpha 門檻）")
print(f"  {'門檻':>6}  {'基準率':>7}  " +
      "  ".join(f"{sc:>5}" for sc, _ in sigs))
print("  " + "-" * 72)
for thresh in [10, 15, 20, 25, 30]:
    gw   = df["alpha_6m"] >= thresh
    br   = gw.mean() * 100
    vals = []
    for sc, _ in sigs:
        det  = df[sc]
        tp   = (det & gw).sum()
        prec = tp / det.sum() * 100 if det.sum() else 0
        vals.append(f"{prec:>5.1f}%")
    print(f"  {thresh:>5}%  {br:>6.1f}%  " + "  ".join(vals))

# ── F 信號完整清單 ────────────────────────────────────────────────────
print()
print("=" * 72)
print("  F 信號（量縮後量增突破）完整偵測清單")
print(f"  {'日期':<12} {'股票':<7} {'排名':>5} {'6M alpha':>10}  {'結果':<8}  其他信號同時觸發")
print("  " + "-" * 65)
df_f = df[df["sig_F"]].sort_values("date").reset_index(drop=True)
for _, row in df_f.iterrows():
    status  = "✅ 好浪" if row["is_good_wave"] else "❌ 假浪"
    others  = []
    if row["sig_A"]: others.append("排名速度")
    if row["sig_B"]: others.append("MA60")
    if row["sig_E"]: others.append("OBV")
    if row["sig_G"]: others.append("底部墊高")
    co = " + ".join(others) if others else "—"
    print(f"  {row['date']:<12} {row['sym']:<7} {int(row['rank_now']):>5} "
          f"{row['alpha_6m']:>+9.1f}%  {status:<8}  {co}")

good_f = df_f[df_f["is_good_wave"]]
bad_f  = df_f[~df_f["is_good_wave"]]
print(f"\n  好浪 {len(good_f)} 筆，平均 6M alpha：{good_f['alpha_6m'].mean():+.1f}%")
print(f"  假浪 {len(bad_f)} 筆，平均 6M alpha：{bad_f['alpha_6m'].mean():+.1f}%")
