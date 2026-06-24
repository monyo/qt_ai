import json
import os
from datetime import date as _date, timedelta
from src.portfolio import _ensure_tranches

TRANCHE_PARAMS = {
    "standard": {"fixed": -0.15, "trailing": -0.25, "protect": 30},
    "tight_2":  {"fixed": -0.10, "trailing": -0.15, "protect": 15},
    "tight_3":  {"fixed": -0.07, "trailing": -0.10, "protect":  7},
}

# 波動率分層停損（回測驗證：_vol_stop_backtest.py + _vol_stop_sensitivity.py）
# 中層 -16%/-24%：停損率 46% 但每次損失更淺（-12.5%），存活者品質更高（+24.7%）
# Calmar 12.845 vs -22% 的 11.391；MDD -31.2% vs -38.3%
VOL_TIER_LOW  = 0.35   # 年化波動率 < 35%：維持標準停損
VOL_TIER_HIGH = 0.60   # 年化波動率 > 60%：使用最寬停損

def vol_adjusted_stops(stop_type: str, vol: float | None) -> tuple[float, float]:
    """
    根據年化波動率調整 standard 批次的停損距離（回傳正值距離，如 0.16 = -16%）。
    tight_2 / tight_3 批次為金字塔設計，不做調整。
    """
    if stop_type != "standard":
        params = TRANCHE_PARAMS[stop_type]
        return abs(params["fixed"]), abs(params["trailing"])

    if vol is None or vol < VOL_TIER_LOW:
        return 0.15, 0.25   # 低波動（<35%）：標準
    elif vol < VOL_TIER_HIGH:
        return 0.16, 0.24   # 中波動（35–60%）：緊停損作品質過濾
    else:
        return 0.25, 0.35   # 高波動（>60%）：寬停損避免雜訊

# 動態收緊追蹤停損：獲利達 +25% 時，把追蹤停損從原始值收緊
# 回測顯示：Calmar 0.516 → 0.620（+0.104），優於分批停利
TIGHTEN_THRESHOLD = 0.25
TIGHTEN_MAP = {
    "standard": 0.15,   # 原 0.25 → 收緊至 0.15
    "tight_2":  0.10,   # 原 0.15 → 收緊至 0.10
    "tight_3":  0.07,   # 原 0.10 → 收緊至 0.07
}


def check_stop_loss(positions, current_prices, threshold=-0.35):
    """檢查持倉是否觸發極端停損（從成本價計算）

    Args:
        positions: 持倉 dict
        current_prices: {symbol: price}
        threshold: 停損閾值（預設 -35%）

    Returns:
        list of dict: [{"symbol": str, "pnl_pct": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        if price is None:
            continue
        pnl_pct = (price - pos["avg_price"]) / pos["avg_price"]
        if pnl_pct <= threshold:
            triggered.append({
                "symbol": symbol,
                "pnl_pct": round(pnl_pct * 100, 2),
                "reason": "extreme_stop",
            })
    return triggered


def check_fixed_stop(positions, current_prices, threshold=-0.15):
    """檢查持倉是否觸發固定停損（從成本價計算）

    根據回測結果，Fixed -15% 優於 Trailing -15%：
    - 不會被正常波動洗掉
    - 對贏家股票傷害較小
    - 設定簡單，券商掛一次停損限價單即可

    Args:
        positions: 持倉 dict
        current_prices: {symbol: price}
        threshold: 停損閾值（預設 -15%）

    Returns:
        list of dict: [{"symbol": str, "pnl_pct": float, "stop_price": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        avg_price = pos.get("avg_price", 0)
        if price is None or avg_price <= 0:
            continue

        pnl_pct = (price - avg_price) / avg_price
        stop_price = avg_price * (1 + threshold)

        if pnl_pct <= threshold:
            triggered.append({
                "symbol": symbol,
                "pnl_pct": round(pnl_pct * 100, 2),
                "avg_price": avg_price,
                "current_price": price,
                "stop_price": round(stop_price, 2),
                "reason": "fixed_stop",
            })
    return triggered


def check_trailing_stop(positions, current_prices, trailing_pct=0.25):
    """檢查持倉是否觸發追蹤停損（從持倉最高點回落指定比例）

    回測驗證（10Y S&P500 全市場）：
    - 無追蹤停損：CAGR +23.1%，MDD -31.4%，Calmar 0.734
    - 追蹤 -25%：CAGR +23.2%，MDD -26.4%，Calmar 0.882（最佳門檻）
    - -20% 至 -30% 效果相近，-25% 為最佳敏感度掃描結果

    Args:
        positions: 持倉 dict（每個 pos 需含 high_since_entry）
        current_prices: {symbol: price}
        trailing_pct: 從最高點回落觸發比例（預設 0.25 = -25%）

    Returns:
        list of dict: [{"symbol": str, "high_since_entry": float,
                        "current_price": float, "from_high_pct": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        high_price = pos.get("high_since_entry")
        if price is None or high_price is None or high_price <= 0:
            continue
        from_high = (price - high_price) / high_price
        if from_high <= -trailing_pct:
            triggered.append({
                "symbol": symbol,
                "high_since_entry": high_price,
                "current_price": price,
                "from_high_pct": round(from_high * 100, 2),
                "reason": "trailing_stop",
            })
    return triggered


def check_ma200_stop(positions, current_prices, ma200_prices):
    """檢查持倉是否跌破 MA200

    Args:
        positions: 持倉 dict
        current_prices: {symbol: price}
        ma200_prices: {symbol: ma200_value}

    Returns:
        list of dict: [{"symbol": str, "ma200": float, "current_price": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        ma200 = ma200_prices.get(symbol)
        if price is None or ma200 is None:
            continue

        if price < ma200:
            triggered.append({
                "symbol": symbol,
                "ma200": round(ma200, 2),
                "current_price": price,
                "below_pct": round((price - ma200) / ma200 * 100, 2),
                "reason": "ma200_stop",
            })
    return triggered


# 反停損獵殺對策（回測：組合策略 Calmar +3.46% vs 基準，2020-2026）
# B 兩日確認 + C 成交量確認 + D VIX 展寬
VIX_WIDEN = {25: 1.5, 35: 2.0}   # VIX 門檻 → 停損距離倍數
VIX_WIDEN_MAX = 0.50              # 追蹤停損展寬上限（不超過 -50%）
FIXED_WIDEN_MAX = 0.35            # 固定停損展寬上限（不超過 -35%）
VOL_CONFIRM_RATIO = 0.8           # 成交量確認門檻（低於 20MA×0.8 視為低量）


def _vix_multiplier(vix: float) -> float:
    if vix >= 35:
        return VIX_WIDEN[35]
    if vix >= 25:
        return VIX_WIDEN[25]
    return 1.0


def check_all_exit_conditions(positions, current_prices, ma200_prices,
                               fixed_threshold=-0.15, hard_threshold=-0.35,
                               trailing_pct=0.25, vix=20.0, volumes=None,
                               vol_map=None):
    """逐批次檢查出場條件，回傳需要出場的持倉

    每個 tranche 按各自的 stop_type 套用差異停損參數：
    - standard: fixed -15%, trailing -25%, 保護 30 天（第1批 / 撿便宜加碼）
    - tight_2:  fixed -10%, trailing -15%, 保護 15 天（上漲加碼第2批）
    - tight_3:  fixed -7%,  trailing -10%, 保護 7 天（上漲加碼第3批+）

    反停損獵殺對策（vix + volumes 啟用時）：
    - VIX 展寬：VIX>25 停損距離×1.5，VIX>35 ×2.0
    - 兩日確認：首次觸發設 stop_pending_since，隔日仍觸發才出場
    - 成交量確認：低量（<20MA×0.8）觸發視為低可信度，同樣進兩日確認

    Returns:
        (exits, pending_notices)
        exits:   {symbol: [tranche_exit, ...]}   — 確認出場（立即執行）
        pending: {symbol: [tranche_pending, ...]} — 首日觸發待確認（明日觀察）
        每個 symbol 的列表按 tranche_n 降序排列（先出最新批次）
    """
    today     = _date.today()
    today_str = str(today)
    vix_mult  = _vix_multiplier(vix or 20.0)
    exits     = {}
    pending   = {}

    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        if price is None:
            continue

        _ensure_tranches(pos)
        sym_exits   = []
        sym_pending = []

        # 成交量資料
        vol_data  = (volumes or {}).get(symbol, {})
        vol       = vol_data.get("volume", 0)
        vol_ma    = vol_data.get("vol_ma20", 0)
        low_vol   = vol_ma > 0 and vol < vol_ma * VOL_CONFIRM_RATIO
        vol_ratio = round(vol / vol_ma, 2) if vol_ma > 0 else None

        sym_vol = (vol_map or {}).get(symbol)  # 年化波動率（可能為 None）

        for t in pos["tranches"]:
            stop_type = t.get("stop_type", "standard")
            params    = TRANCHE_PARAMS.get(stop_type, TRANCHE_PARAMS["standard"])

            # 保護期檢查
            try:
                entry_date = _date.fromisoformat(t["entry_date"])
                days_held  = (today - entry_date).days
            except (ValueError, KeyError):
                days_held  = 999

            if days_held < params["protect"]:
                t.pop("stop_pending_since", None)
                continue

            n           = t["n"]
            entry_price = t["entry_price"]
            t_high      = t.get("high", entry_price)
            t_shares    = t["shares"]

            # ── 波動率分層調整停損基礎距離（standard 批次限定）────────────
            base_fixed_dist, base_trail_dist = vol_adjusted_stops(stop_type, sym_vol)

            # ── 計算有效停損位（含 VIX 展寬）────────────────────────────
            eff_fixed_dist  = min(base_fixed_dist * vix_mult, FIXED_WIDEN_MAX)
            fixed_stop_price = entry_price * (1 - eff_fixed_dist)

            eff_trailing      = t.get("trailing_pct", base_trail_dist)
            eff_trail_dist    = min(eff_trailing * vix_mult, VIX_WIDEN_MAX)
            trailing_stop_price = t_high * (1 - eff_trail_dist) if t_high > 0 else 0

            # 是否觸發任一停損
            fixed_hit   = price <= fixed_stop_price
            trail_hit   = t_high > 0 and price <= trailing_stop_price
            stop_hit    = fixed_hit or trail_hit

            # ── 價格回復：清除待確認 ──────────────────────────────────────
            if not stop_hit:
                if t.get("stop_pending_since"):
                    t.pop("stop_pending_since", None)
                # 3. MA200 停損（不做兩日確認）
                if stop_type == "standard":
                    ma200 = ma200_prices.get(symbol)
                    if ma200 is not None and price < ma200:
                        sym_exits.append({
                            "tranche_n":      n,
                            "tranche_shares": t_shares,
                            "reason":         "ma200_stop",
                            "message":        f"跌破 MA200（第{n}批，MA200 ${ma200:.2f}，{(price-ma200)/ma200*100:.1f}%）",
                            "details":        {"ma200": ma200, "current_price": price},
                        })
                        continue
                    # 4. 極端停損（-35%，不做兩日確認）
                    avg_price = pos.get("avg_price", entry_price)
                    if avg_price > 0:
                        pnl_pct = (price - avg_price) / avg_price
                        if pnl_pct <= hard_threshold:
                            sym_exits.append({
                                "tranche_n":      n,
                                "tranche_shares": t_shares,
                                "reason":         "extreme_stop",
                                "message":        f"極端停損觸發（第{n}批，從成本 {pnl_pct*100:.1f}%）",
                                "details":        {"avg_price": avg_price, "pnl_pct": pnl_pct, "current_price": price},
                            })
                continue

            # ── 固定/追蹤停損觸發（應用兩日確認）────────────────────────
            if fixed_hit:
                reason  = "fixed_stop"
                vix_tag = f"（VIX={vix:.0f}，展寬至 {eff_fixed_dist*100:.0f}%）" if vix_mult > 1 else ""
                msg     = (f"固定停損觸發{vix_tag}（第{n}批，成本 ${entry_price:.2f}，"
                           f"停損 ${fixed_stop_price:.2f}，目前 {(price-entry_price)/entry_price*100:.1f}%）")
                details = {"entry_price": entry_price, "stop_price": fixed_stop_price,
                           "current_price": price, "vix_mult": vix_mult}
            else:
                from_high = (price - t_high) / t_high
                reason  = "trailing_stop"
                vix_tag = f"（VIX={vix:.0f}，展寬至 {eff_trail_dist*100:.0f}%）" if vix_mult > 1 else ""
                msg     = (f"追蹤停損觸發{vix_tag}（第{n}批，最高 ${t_high:.2f}，"
                           f"回落 {from_high*100:.1f}%）")
                details = {"high": t_high, "trailing_stop": trailing_stop_price,
                           "current_price": price, "vix_mult": vix_mult}

            item = {
                "tranche_n":      n,
                "tranche_shares": t_shares,
                "reason":         reason,
                "message":        msg,
                "details":        details,
            }

            pending_since = t.get("stop_pending_since")
            if pending_since and pending_since < today_str:
                # 第二日確認（非同日重跑）→ 真正出場
                t.pop("stop_pending_since", None)
                sym_exits.append(item)
            else:
                # 第一日（或同日重跑）→ 設待確認
                t["stop_pending_since"] = today_str
                notice = dict(item)
                notice["pending_reason"] = "low_volume" if low_vol else "two_day_confirm"
                notice["vol_ratio"]      = vol_ratio
                notice["vix_mult"]       = vix_mult
                sym_pending.append(notice)

        if sym_exits:
            sym_exits.sort(key=lambda x: x["tranche_n"], reverse=True)
            exits[symbol] = sym_exits
        if sym_pending:
            sym_pending.sort(key=lambda x: x["tranche_n"], reverse=True)
            pending[symbol] = sym_pending

    return exits, pending


def update_dynamic_trailing(portfolio, current_prices):
    """檢查各批次獲利是否達 +25%，若是則收緊追蹤停損 trailing_pct

    Returns:
        list of str: 本次新收緊的 "(symbol 第N批)" 說明
    """
    newly_tightened = []
    for symbol, pos in portfolio.get("positions", {}).items():
        if pos.get("core"):
            continue
        price = current_prices.get(symbol)
        if price is None:
            continue
        _ensure_tranches(pos)
        for t in pos["tranches"]:
            stop_type = t.get("stop_type", "standard")
            # 已收緊過就跳過（trailing_pct 已存在）
            if "trailing_pct" in t:
                continue
            entry_price = t.get("entry_price", pos.get("avg_price", 0))
            if entry_price <= 0:
                continue
            pnl = (price - entry_price) / entry_price
            if pnl >= TIGHTEN_THRESHOLD:
                tightened = TIGHTEN_MAP.get(stop_type, 0.15)
                t["trailing_pct"] = tightened
                newly_tightened.append(f"{symbol} 第{t['n']}批（{stop_type}→{tightened*100:.0f}%）")
    return newly_tightened


# ── Winner Cycle：特殊池輪動策略 ───────────────────────────────────────────────
# 回測（_winner_cycle_backtest.py）：獲利>100% 的標的從高點回落 -10% 時賣出，
# 低點反彈 +10% 時買回，平均報酬 +102.5% vs 純持有 +33.1%，勝率 55%。
WINNER_CYCLE_PULLBACK        = 0.10    # 從週期高點回落 -10% 時賣出
WINNER_CYCLE_RECOVERY        = 0.10    # 從賣出後低點反彈 +10% 時買回
WINNER_CYCLE_COOLDOWN        = 5       # 賣出後至少等 5 天才能買回
WINNER_CYCLE_ALPHA_THRESHOLD = 100.0  # 1Y alpha > 100%（股票本身超額報酬，不依賴進場時機）
WINNER_CYCLE_WATCH_FILE      = "data/winner_cycle_watch.json"


def update_winner_cycle_highs(portfolio, current_prices, alpha_1y_map=None):
    """更新特殊池持倉的週期高點（winner_cycle_high）。

    進入特殊池條件（優先使用 alpha_1y_map，反映股票本身動能而非進場時機）：
    - 提供 alpha_1y_map：該標的 1Y alpha > WINNER_CYCLE_ALPHA_THRESHOLD（預設 100%）
    - 未提供：持倉獲利 > 100%（fallback，向後相容）

    Returns: 本次新進入特殊池的 symbol 列表
    """
    newly_entered = []
    for sym, pos in portfolio.get("positions", {}).items():
        if pos.get("core"):
            continue
        price = current_prices.get(sym)
        if price is None:
            continue

        if alpha_1y_map is not None:
            alpha = alpha_1y_map.get(sym)
            qualifies = alpha is not None and alpha > WINNER_CYCLE_ALPHA_THRESHOLD
        else:
            avg_price = pos.get("avg_price", 0)
            qualifies = avg_price > 0 and (price - avg_price) / avg_price >= 1.0

        if qualifies:
            if "winner_cycle_high" not in pos:
                # 用各批次追蹤到的最高價初始化（不是今天報價），避免忽略已發生的回落
                _ensure_tranches(pos)
                hist_high = max((t.get("high", t.get("entry_price", 0)) for t in pos["tranches"]), default=price)
                pos["winner_cycle_high"] = max(hist_high, price)
                pos["winner_cycle_pool_date"] = str(_date.today())
                newly_entered.append(sym)
            else:
                pos["winner_cycle_high"] = max(pos["winner_cycle_high"], price)
        else:
            pos.pop("winner_cycle_high", None)
            pos.pop("winner_cycle_pool_date", None)
    return newly_entered


def check_winner_cycle_exits(portfolio, current_prices):
    """找出特殊池中從週期高點回落 ≥ WINNER_CYCLE_PULLBACK 的持倉。

    Returns: {symbol: {"cycle_high", "current_price", "from_high_pct", "shares", "avg_price"}}
    """
    exits = {}
    for sym, pos in portfolio.get("positions", {}).items():
        if pos.get("core"):
            continue
        cycle_high = pos.get("winner_cycle_high")
        if cycle_high is None:
            continue
        price = current_prices.get(sym)
        if price is None:
            continue
        from_high = (price - cycle_high) / cycle_high
        if from_high <= -WINNER_CYCLE_PULLBACK:
            _ensure_tranches(pos)
            total_shares = sum(t["shares"] for t in pos["tranches"])
            pnl_pct = (price - pos.get("avg_price", price)) / pos.get("avg_price", price) * 100 if pos.get("avg_price") else 0
            exits[sym] = {
                "cycle_high": cycle_high,
                "current_price": price,
                "from_high_pct": round(from_high * 100, 1),
                "shares": total_shares,
                "avg_price": pos.get("avg_price", 0),
                "pnl_pct": round(pnl_pct, 2),
            }
    return exits


def load_winner_cycle_watch():
    """載入特殊池出場後的觀察名單"""
    if os.path.exists(WINNER_CYCLE_WATCH_FILE):
        try:
            with open(WINNER_CYCLE_WATCH_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_winner_cycle_watch(watch):
    """儲存特殊池觀察名單"""
    os.makedirs("data", exist_ok=True)
    with open(WINNER_CYCLE_WATCH_FILE, "w", encoding="utf-8") as f:
        json.dump(watch, f, ensure_ascii=False, indent=2)


def update_winner_cycle_watch_lows(watch, current_prices):
    """更新觀察名單中各標的的賣出後最低價"""
    for sym, entry in watch.items():
        price = current_prices.get(sym)
        if price is not None:
            entry["post_exit_low"] = min(entry.get("post_exit_low", price), price)


def check_winner_cycle_reentries(watch, current_prices):
    """找出觀察名單中已從低點反彈 ≥ WINNER_CYCLE_RECOVERY 且冷卻期已過的標的。

    Returns: {symbol: {"exit_price", "post_exit_low", "recovery_pct", "shares", "avg_price"}}
    """
    today = str(_date.today())
    reentries = {}
    for sym, entry in watch.items():
        if today <= entry.get("cooldown_end", ""):
            continue
        price = current_prices.get(sym)
        post_exit_low = entry.get("post_exit_low", entry.get("exit_price", 0))
        if price is None or post_exit_low <= 0:
            continue
        recovery = price / post_exit_low - 1
        if recovery >= WINNER_CYCLE_RECOVERY:
            reentries[sym] = {
                "exit_price": entry.get("exit_price", 0),
                "post_exit_low": post_exit_low,
                "recovery_pct": round(recovery * 100, 1),
                "shares": entry.get("shares", 0),
                "avg_price": entry.get("avg_price", 0),
                "current_price": price,
            }
    return reentries


def confirm_winner_cycle_exit(symbol, shares, exit_price, avg_price):
    """確認特殊池出場，寫入觀察名單等待回補"""
    watch = load_winner_cycle_watch()
    cooldown_end = str(_date.today() + timedelta(days=WINNER_CYCLE_COOLDOWN))
    watch[symbol] = {
        "exit_date": str(_date.today()),
        "exit_price": exit_price,
        "post_exit_low": exit_price,
        "cooldown_end": cooldown_end,
        "shares": shares,
        "avg_price": avg_price,
    }
    save_winner_cycle_watch(watch)


def confirm_winner_cycle_reentry(symbol):
    """確認特殊池回補，從觀察名單移除"""
    watch = load_winner_cycle_watch()
    watch.pop(symbol, None)
    save_winner_cycle_watch(watch)


def check_position_limit(portfolio, max_stocks=30):
    """回傳還能買幾檔個股"""
    individual_count = sum(
        1 for pos in portfolio["positions"].values()
        if not pos.get("core", False)
    )
    return max(max_stocks - individual_count, 0)
