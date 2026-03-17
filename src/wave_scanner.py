"""
波浪警報掃描器

每日盤前偵測「量縮後量增突破（F）＋ 排名速度（A）」雙信號。
回測顯示：F 信號精準度 23%（基準率 12.9%），F+A 同時觸發時 4 年命中 META、ANET
共 2 次，精準度約 67%。

使用方式：
    from src.wave_scanner import scan_waves
    alerts = scan_waves(verbose=True)
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from src.data_loader import get_sp500_tickers

# ── 參數 ──────────────────────────────────────────────────────────────
_CACHE_PREFIX  = "data/_wave_daily_"
_BATCH_SIZE    = 50
_DATA_DAYS     = 420    # 確保涵蓋 252 個交易日（約 14 個月）
MOM_SHORT      = 21
MOM_LONG       = 252
TOP_SYSTEM     = 20     # 前 N 名已在主系統推薦池
RANK_MAX       = 150    # 候選池下限

# F 信號：量縮後量增突破
_VOL_BASE_WIN  = 60     # 基礎量（平均窗口）
_VOL_DRY_WIN   = 15     # 整理縮量天數
_VOL_SURGE_WIN = 5      # 突破放量天數
_VOL_DRY_RATIO = 0.65   # 縮量期：< 平均量 65%
_VOL_SURGE_RAT = 1.30   # 放量期：> 平均量 130%
_PRICE_UP_MIN  = 0.02   # 近 5 天漲幅 > 2%

# A 信號：排名速度
_RANK_VEL_MIN  = 30     # 排名至少進步幾名
_RANK_VEL_MAX  = 80     # 當下排名上限（過低的進步不算）


# ── 快取管理 ──────────────────────────────────────────────────────────
def _cache_path() -> str:
    return f"{_CACHE_PREFIX}{date.today()}.pkl"

def _clean_old_caches():
    today_file = os.path.basename(_cache_path())
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.startswith("_wave_daily_") and f != today_file:
                try:
                    os.remove(os.path.join("data", f))
                except OSError:
                    pass

def _load_or_download(verbose=False) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """回傳 (close_df, vol_df)，以交易日為索引。今天已有快取則直接讀取。"""
    cache = _cache_path()
    _clean_old_caches()

    if os.path.exists(cache):
        if verbose:
            print("  [波浪掃描] 讀取今日快取")
        data = pd.read_pickle(cache)
        return data["close"], data["volume"]

    tickers = get_sp500_tickers()
    end_dt  = date.today()
    start_dt = end_dt - timedelta(days=_DATA_DAYS)

    if verbose:
        print(f"  [波浪掃描] 首次下載 {len(tickers)} 檔（Close+Volume，快取後不重複下載）...")

    all_close, all_vol = [], []
    batches = [tickers[i:i+_BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    for batch in batches:
        try:
            raw = yf.download(batch, start=str(start_dt), end=str(end_dt),
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            c = raw["Close"]
            v = raw["Volume"]
            if isinstance(c.columns, pd.MultiIndex):
                c.columns = c.columns.droplevel(0)
                v.columns = v.columns.droplevel(0)
            all_close.append(c)
            all_vol.append(v)
        except Exception:
            pass

    if not all_close:
        return None, None

    def _merge(frames):
        df = pd.concat(frames, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        return df

    close_df = _merge(all_close)
    vol_df   = _merge(all_vol)

    os.makedirs("data", exist_ok=True)
    pd.to_pickle({"close": close_df, "volume": vol_df}, cache)
    if verbose:
        print(f"  [波浪掃描] 已快取至 {cache}")
    return close_df, vol_df


# ── 指標計算 ──────────────────────────────────────────────────────────
def _ranks(close_df: pd.DataFrame, ti: int) -> dict:
    """計算全市場 mom_mixed 排名 {sym: rank}，1 = 最強"""
    scores = {}
    for sym in close_df.columns:
        if ti < MOM_LONG or ti >= len(close_df):
            continue
        try:
            p0   = float(close_df.iloc[ti][sym])
            p21  = float(close_df.iloc[ti - MOM_SHORT][sym])
            p252 = float(close_df.iloc[ti - MOM_LONG][sym])
        except (KeyError, TypeError):
            continue
        if any(np.isnan(x) or x <= 0 for x in [p0, p21, p252]):
            continue
        m = 0.5*(p0/p21 - 1)*100 + 0.5*(p0/p252 - 1)*100
        if m > 0:
            scores[sym] = m
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {sym: i+1 for i, sym in enumerate(ranked)}


def _sig_vol_dry_breakout(sym: str, close_df: pd.DataFrame,
                          vol_df: pd.DataFrame, ti: int) -> bool:
    """F 信號：整理縮量後，近期放量 + 價格向上"""
    if sym not in vol_df.columns:
        return False
    try:
        # 基礎量（60 日平均）
        base = vol_df.iloc[ti - _VOL_BASE_WIN: ti][sym].dropna()
        if len(base) < 30 or base.mean() <= 0:
            return False
        avg_vol = base.mean()

        # 縮量期（surge 之前 15 天）
        dry_start = ti - _VOL_SURGE_WIN - _VOL_DRY_WIN
        dry_end   = ti - _VOL_SURGE_WIN
        dry = vol_df.iloc[dry_start:dry_end][sym].dropna()
        if dry.empty:
            return False

        # 放量期（近 5 天）
        surge = vol_df.iloc[ti - _VOL_SURGE_WIN: ti][sym].dropna()
        if surge.empty:
            return False

        # 價格近 5 天上漲
        p_now = float(close_df.iloc[ti][sym])
        p_ago = float(close_df.iloc[ti - _VOL_SURGE_WIN][sym])
        if np.isnan(p_now) or np.isnan(p_ago) or p_ago <= 0:
            return False

        return (dry.mean() / avg_vol  < _VOL_DRY_RATIO and
                surge.mean() / avg_vol > _VOL_SURGE_RAT and
                p_now / p_ago - 1      > _PRICE_UP_MIN)
    except Exception:
        return False


# ── 主掃描函式 ────────────────────────────────────────────────────────
def scan_waves(verbose: bool = False) -> list[dict]:
    """
    執行波浪偵測掃描。

    Returns:
        list[dict]，每筆包含：
            sym          股票代碼
            rank_now     當下排名
            rank_prev    上個月排名
            rank_vel     排名進步幅度（正值＝進步）
            mom_pct      當下混合動能（%）
            alert_level  "HIGH"（F+A 雙信號）or "WATCH"（僅 F 信號）
            signals      觸發信號描述
        依 alert_level 優先、rank_now 升序排列（排名越前越好）。
    """
    close_df, vol_df = _load_or_download(verbose=verbose)
    if close_df is None:
        return []

    td = close_df.index.sort_values()
    ti_now  = len(td) - 1
    ti_prev = ti_now - MOM_SHORT      # ~21 個交易日前（約一個月）

    if ti_prev < MOM_LONG:
        return []

    if verbose:
        print("  [波浪掃描] 計算全市場動能排名...")

    ranks_now  = _ranks(close_df, ti_now)
    ranks_prev = _ranks(close_df, ti_prev)

    alerts = []
    for sym, rank_now in ranks_now.items():
        if rank_now <= TOP_SYSTEM or rank_now > RANK_MAX:
            continue

        # F 信號（必要條件）
        if not _sig_vol_dry_breakout(sym, close_df, vol_df, ti_now):
            continue

        # A 信號
        rank_prev = ranks_prev.get(sym)
        rank_vel  = (rank_prev - rank_now) if rank_prev else 0
        sig_a = (rank_prev is not None
                 and rank_now <= _RANK_VEL_MAX
                 and rank_vel >= _RANK_VEL_MIN)

        # 估算當下動能
        try:
            p0, p21, p252 = (float(close_df.iloc[ti_now][sym]),
                             float(close_df.iloc[ti_now - MOM_SHORT][sym]),
                             float(close_df.iloc[ti_now - MOM_LONG][sym]))
            mom_pct = 0.5*(p0/p21 - 1)*100 + 0.5*(p0/p252 - 1)*100
        except Exception:
            mom_pct = None

        sigs = ["量縮量增突破"]
        if sig_a:
            sigs.append(f"排名↑{rank_vel}名（{rank_prev}→{rank_now}）")

        alerts.append({
            "sym":         sym,
            "rank_now":    rank_now,
            "rank_prev":   rank_prev,
            "rank_vel":    rank_vel,
            "mom_pct":     mom_pct,
            "alert_level": "HIGH" if sig_a else "WATCH",
            "signals":     " + ".join(sigs),
        })

    alerts.sort(key=lambda x: (0 if x["alert_level"] == "HIGH" else 1, x["rank_now"]))
    return alerts
