"""
市場廣度監控

使用波浪掃描器的每日快取計算 S&P500 股票廣度（% > 50MA）。
回測顯示：廣度加權 ADD 數量（max(1, round(breadth×5))）
比固定選 5 支好 +1%/+4.5% alpha（1M/3M）。

廣度分級：
  ≥ 65%   健康    → ADD ≤ 5 支（正常）
  50-65%  偏弱    → ADD ≤ 4 支
  40-50%  弱      → ADD ≤ 3 支  ⚠️
  30-40%  危險    → ADD ≤ 2 支  🔴
  < 30%   極危險  → ADD ≤ 1 支  🔴

使用：
    from src.breadth_monitor import get_breadth_status
    status = get_breadth_status()
"""
import os
import pandas as pd
import numpy as np
from datetime import date

_WAVE_CACHE_PREFIX = "data/_wave_daily_"
_MA_WIN            = 50
_MIN_VALID_STOCKS  = 100


def _load_close_df() -> pd.DataFrame | None:
    """載入今日波浪掃描器快取的收盤價 DataFrame。"""
    cache = f"{_WAVE_CACHE_PREFIX}{date.today()}.pkl"
    if not os.path.exists(cache):
        return None
    try:
        data = pd.read_pickle(cache)
        close_df = data.get("close")
        if close_df is None or close_df.empty:
            return None
        close_df.index = pd.to_datetime(close_df.index)
        if close_df.index.tz is not None:
            close_df.index = close_df.index.tz_convert(None)
        return close_df.sort_index()
    except Exception:
        return None


def get_stock_breadth() -> float | None:
    """
    計算今日 S&P500 股票廣度：高於 50MA 的比例。
    Returns float [0,1] or None if data unavailable.
    """
    close_df = _load_close_df()
    if close_df is None:
        return None

    ma50     = close_df.rolling(_MA_WIN, min_periods=30).mean()
    curr     = close_df.iloc[-1]
    ma50_now = ma50.iloc[-1]
    valid    = curr.notna() & ma50_now.notna()
    if valid.sum() < _MIN_VALID_STOCKS:
        return None
    return float((curr[valid] > ma50_now[valid]).mean())


def get_breadth_status() -> dict:
    """
    回傳市場廣度狀況字典：
        stock_breadth       float | None   % 股票 > MA50
        suggested_max_adds  int            建議最多 ADD 幾支
        level               str            "健康"/"偏弱"/"弱"/"危險"/"極危險"
        warning             bool           是否顯示警告
        emoji               str            狀態表情符號
        display_line        str            盤前報告顯示用的一行文字
    """
    breadth  = get_stock_breadth()

    if breadth is None:
        return {
            "stock_breadth":      None,
            "suggested_max_adds": 5,
            "level":              "未知",
            "warning":            False,
            "emoji":              "⬜",
            "display_line":       "廣度資料不可用（波浪掃描未執行）",
        }

    suggested = max(1, round(breadth * 5))

    if breadth >= 0.65:
        level, emoji, warning = "健康",  "🟢", False
    elif breadth >= 0.50:
        level, emoji, warning = "偏弱",  "🟡", False
    elif breadth >= 0.40:
        level, emoji, warning = "弱",    "🟠", True
    elif breadth >= 0.30:
        level, emoji, warning = "危險",  "🔴", True
    else:
        level, emoji, warning = "極危險","🔴", True

    display_line = (
        f"{emoji} 市場廣度: {breadth*100:.0f}% 股票 > MA50 "
        f"（{level}）  建議 ADD ≤ {suggested} 支"
    )
    if warning:
        display_line += "  ← 廣度偏低，集中押最強標的"

    return {
        "stock_breadth":      breadth,
        "suggested_max_adds": suggested,
        "level":              level,
        "warning":            warning,
        "emoji":              emoji,
        "display_line":       display_line,
    }
