"""
市場環境監控：VIX + 石油

盤前報告使用：
    from src.market_environment import get_market_environment
    env = get_market_environment()
    print(env["display_block"])   # 多行顯示文字
    env["vix_level"]              # float
    env["oil_ret_21d"]            # float（百分比）
    env["regime_label"]           # "平靜牛市" / "恐慌上升" / etc.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def get_market_environment() -> dict:
    """
    抓取 VIX + WTI 石油，回傳環境狀態與顯示文字。
    若抓取失敗，回傳 degraded dict（不影響盤前主流程）。
    """
    try:
        # 油價用 USO ETF（追蹤 WTI，無期貨轉倉跳價問題）
        raw = yf.download(
            ["^VIX", "USO", "SPY"],
            period="300d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw

        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.ffill()

        if len(close) < 30:
            return _degraded()

        # ── VIX ────────────────────────────────────────────────────────
        vix = close.get("^VIX")
        if vix is None or vix.dropna().empty:
            vix_level = None
            vix_ma63  = None
            vix_ratio = None
        else:
            vix_level = float(vix.iloc[-1])
            vix_ma63  = float(vix.rolling(63, min_periods=20).mean().iloc[-1])
            vix_ratio = vix_level / vix_ma63 if vix_ma63 and vix_ma63 > 0 else None

        # ── 石油 ─────────────────────────────────────────────────────────
        oil = close.get("USO")
        if oil is None or oil.dropna().empty:
            oil_ret_21 = None
            oil_ret_63 = None
            oil_ma_ratio = None
            oil_vs_spy = None
        else:
            oil = oil.dropna()
            spy_series = close.get("SPY", pd.Series(dtype=float)).dropna()

            def pct(series, n):
                if len(series) <= n:
                    return None
                v0 = float(series.iloc[-1])
                vn = float(series.iloc[-(n+1)])
                return (v0 / vn - 1) * 100 if vn > 0 else None

            oil_ret_21   = pct(oil, 21)
            oil_ret_63   = pct(oil, 63)
            oil_ma200    = oil.rolling(200, min_periods=60).mean().iloc[-1]
            oil_ma_ratio = float(oil.iloc[-1] / oil_ma200) if oil_ma200 and oil_ma200 > 0 else None

            if len(oil) > 21 and len(spy_series) > 21:
                spy_ret_21 = pct(spy_series, 21) or 0
                oil_vs_spy = (oil_ret_21 or 0) - spy_ret_21
            else:
                oil_vs_spy = None

        # ── 環境解讀 ───────────────────────────────────────────────────
        regime_label, regime_note, regime_emoji = _interpret(
            vix_level, vix_ratio, oil_ret_21
        )

        # ── 顯示文字 ───────────────────────────────────────────────────
        display_block = _format_display(
            vix_level, vix_ma63, vix_ratio,
            oil_ret_21, oil_ret_63, oil_ma_ratio,
            regime_label, regime_note, regime_emoji,
        )

        return {
            "ok":           True,
            "vix_level":    vix_level,
            "vix_ma63":     vix_ma63,
            "vix_ma_ratio": vix_ratio,
            "oil_ret_21d":  oil_ret_21,
            "oil_ret_63d":  oil_ret_63,
            "oil_ma_ratio": oil_ma_ratio,
            "oil_vs_spy_21d": oil_vs_spy,
            "regime_label": regime_label,
            "regime_note":  regime_note,
            "regime_emoji": regime_emoji,
            "display_block": display_block,
        }

    except Exception as e:
        return _degraded(str(e))


def _interpret(vix, vix_ratio, oil_21):
    """回傳 (短標籤, 補充說明, emoji)"""
    vix_high   = vix is not None and (vix >= 22 or (vix >= 18 and vix_ratio is not None and vix_ratio >= 1.3))
    vix_rising = vix_ratio is not None and vix_ratio >= 1.2
    oil_up     = oil_21 is not None and oil_21 >= 5

    if vix_high and oil_up:
        return (
            "滯脹恐慌",
            "高VIX + 油漲：板塊大分化，能源/原物料佔優，科技承壓。\n"
            "  動能策略自然傾向能源族群，相對 SPY alpha 歷史偏高。",
            "🔴",
        )
    if vix_high and not oil_up:
        return (
            "恐慌下跌",
            "高VIX + 油穩/跌：市場恐慌但通縮壓力低，通常為急跌後反彈前。\n"
            "  動能訊號可靠性下降，建議只執行最高信心度的 ADD。",
            "🟡",
        )
    if not vix_high and oil_up:
        return (
            "健康風險偏好",
            "低VIX + 油漲：經濟需求強勁，風險偏好正常，動能有效。",
            "🟢",
        )
    # 低 VIX + 油穩/跌
    return (
        "平靜牛市",
        "低VIX + 油穩：市場平靜，但大型科技主導指數。\n"
        "  個股相對 SPY 超額報酬歷史偏低，動能選股需更嚴格。",
        "🟢",
    )


def _format_display(vix_level, vix_ma63, vix_ratio,
                    oil_ret_21, oil_ret_63, oil_ma_ratio,
                    regime_label, regime_note, regime_emoji):
    lines = ["--- 市場環境 ---"]

    # VIX 行
    if vix_level is not None:
        vix_arrow = "↑" if (vix_ratio or 0) >= 1.1 else ("↓" if (vix_ratio or 1) < 0.9 else "→")
        vix_tag = "⚠️ 高恐慌" if vix_level >= 25 else ("低恐慌" if vix_level < 15 else "正常")
        vix_ma_str = f"（vs 63日均 {vix_ma63:.1f}，×{vix_ratio:.2f}）" if vix_ma63 else ""
        lines.append(f"  VIX   {vix_level:.1f} {vix_arrow}  {vix_tag}{vix_ma_str}")
    else:
        lines.append("  VIX   資料無法取得")

    # 石油行
    if oil_ret_21 is not None:
        oil_arrow = "↑ 急漲" if oil_ret_21 >= 10 else (
                    "↑ 上漲" if oil_ret_21 >= 3 else (
                    "↓ 下跌" if oil_ret_21 <= -5 else "平穩"))
        oil_ma_str = ""
        if oil_ma_ratio is not None:
            oil_ma_str = f"  （均線比 {oil_ma_ratio:.2f}×）"
        lines.append(f"  WTI   {oil_ret_21:+.1f}%（21日）{oil_arrow}{oil_ma_str}")
    else:
        lines.append("  WTI   資料無法取得")

    # 環境標籤
    lines.append(f"\n  {regime_emoji} 環境：{regime_label}")
    for note_line in regime_note.split("\n"):
        lines.append(f"  {note_line}")

    return "\n".join(lines)


def _degraded(err=""):
    return {
        "ok":            False,
        "vix_level":     None,
        "vix_ma63":      None,
        "vix_ma_ratio":  None,
        "oil_ret_21d":   None,
        "oil_ret_63d":   None,
        "oil_ma_ratio":  None,
        "oil_vs_spy_21d": None,
        "regime_label":  "無法取得",
        "regime_note":   err,
        "regime_emoji":  "⚪",
        "display_block": "--- 市場環境 ---\n  資料暫時無法取得",
    }
