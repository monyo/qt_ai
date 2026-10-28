"""價值實驗池（Value Experiment Pool）

獨立於主動能池的真金白銀前瞻實驗：篩選基本面便宜、可能尚未被市場充分反映的標的，驗證「低估反轉」因子是否有效。
動能只作為結果標籤與追蹤欄位，不是價值池的預先篩選門檻。
最多同時 5 檔，不設總金額上限。

設計依據見 /home/mark/.claude/plans/crystalline-shimmying-clarke.md：
- 不做歷史回測（yfinance 基本面資料無 point-in-time，回測等於用未來函數）
- 四層出場 + 畢業機制（動能轉強後併入主池，重用既有停損機制）

純函式模組，比照 src/risk.py 的 Winner Cycle 寫法：不碰檔案 I/O，
由 premarket_main.py 負責讀寫 portfolio.json / actions JSON。
"""

from datetime import date as _date, datetime

VALUE_POOL_MAX = 5
VALUE_HARD_STOP = -0.20          # 從成本下跌 -20% 出場（比主池標準 -15% 寬，容忍價值股正常波動）
VALUE_TAKE_PROFIT_PCT = 0.90     # 股價達分析師目標價 90% 即出場（論點已實現）
VALUE_TARGET_CUT_PCT = -0.15     # 進場後分析師目標價被下修超過此幅度 → 出場（論點被破壞）
VALUE_TIME_STOP_DAYS = 180       # 持有 6 個月
VALUE_TIME_STOP_BAND = 0.10      # 期間內漲跌未超過 ±10% 視為盤整，時間到就出場釋放資金

# 進場門檻（暫定，先求可用不做過度優化）
VALUE_MAX_FORWARD_PE = 25
VALUE_MAX_PEG = 1.5
VALUE_MIN_UPSIDE = 0.15          # 分析師目標價中位數上檔空間 > 15%
VALUE_MIN_REVENUE_GROWTH = 0.0
VALUE_ALPHA_3Y_FLOOR = -20.0     # 3Y alpha（百分點）門檻，排除結構性衰退股


def screen_value_candidates(symbols, momentum_map, alpha_3y_map, fundamentals, current_prices, held_symbols, slots,
                            excluded_symbols=None):
    """粗篩完整價值宇宙中的便宜候選，回傳最多 slots 檔。

    動能不參與入池資格判斷，只用於標記候選目前是否已開始被市場確認。

    Args:
        symbols: 完整候選標的清單（通常是 S&P 500 全部）
        momentum_map: {symbol: momentum_score}，僅用於結果標記，不作為篩選條件
        alpha_3y_map: {symbol: alpha_3y}，百分點（如 -20 代表 -20%）
        fundamentals: {symbol: {...}}，來自 data_loader.get_fundamentals_batch()
        current_prices: {symbol: price}，用於算目標價上檔空間
        held_symbols: 已持有標的（主池 + 價值池），排除重複布局
        slots: 還可以新增幾檔
        excluded_symbols: 已在主 ADD/ROTATE 清單的標的，避免與主池重複建議

    Returns:
        list[dict]: [{symbol, forward_pe, peg, target_mean, upside_pct,
                       revenue_growth, sector, momentum, alpha_3y}, ...]
                    按 upside_pct 降序排列
    """
    if slots <= 0:
        return []

    excluded_symbols = set(excluded_symbols or ())

    pool = []
    for sym in symbols:
        if sym in held_symbols or sym in excluded_symbols:
            continue

        alpha_3y = alpha_3y_map.get(sym)
        if alpha_3y is not None and alpha_3y < VALUE_ALPHA_3Y_FLOOR:
            continue  # 排除結構性衰退股

        momentum = momentum_map.get(sym)

        fnd = fundamentals.get(sym)
        if not fnd:
            continue

        forward_pe = fnd.get("forward_pe")
        peg = fnd.get("peg")
        target_mean = fnd.get("target_mean")
        revenue_growth = fnd.get("revenue_growth")

        if forward_pe is None or forward_pe <= 0 or forward_pe > VALUE_MAX_FORWARD_PE:
            continue
        if peg is not None and (peg <= 0 or peg > VALUE_MAX_PEG):
            continue
        if revenue_growth is not None and revenue_growth < VALUE_MIN_REVENUE_GROWTH:
            continue
        if not target_mean:
            continue

        current_price = current_prices.get(sym)
        upside_pct = None
        if current_price:
            upside_pct = target_mean / current_price - 1
        if upside_pct is None or upside_pct < VALUE_MIN_UPSIDE:
            continue

        pool.append({
            "symbol": sym,
            "forward_pe": forward_pe,
            "peg": peg,
            "target_mean": target_mean,
            "upside_pct": round(upside_pct * 100, 1),
            "revenue_growth": revenue_growth,
            "sector": fnd.get("sector"),
            "momentum": momentum,
            "alpha_3y": alpha_3y,
            "value_type": "value_with_confirmation" if (momentum or 0) > 0 else "early_value",
        })

    pool.sort(key=lambda c: c["upside_pct"], reverse=True)
    return pool[:slots]


def screen_independent_fundamental_candidates(symbols, fundamentals, current_prices):
    """從完整股票宇宙獨立初篩基本面/估值候選，不使用動能或價格報酬訊號。

    目標價上檔只用作研究線索，不能當作內在價值或買進結論。
    """
    candidates = []
    for symbol in symbols:
        fnd = fundamentals.get(symbol)
        price = current_prices.get(symbol)
        if not fnd or not price or price <= 0:
            continue

        forward_pe = fnd.get("forward_pe")
        peg = fnd.get("peg")
        revenue_growth = fnd.get("revenue_growth")
        target_mean = fnd.get("target_mean")
        if forward_pe is None or not 0 < forward_pe <= VALUE_MAX_FORWARD_PE:
            continue
        if peg is not None and not 0 < peg <= VALUE_MAX_PEG:
            continue
        if revenue_growth is None or revenue_growth < VALUE_MIN_REVENUE_GROWTH:
            continue
        if not target_mean or target_mean <= 0:
            continue

        upside_pct = (target_mean / price - 1) * 100
        if upside_pct < VALUE_MIN_UPSIDE * 100:
            continue

        candidates.append({
            "symbol": symbol,
            "forward_pe": forward_pe,
            "peg": peg,
            "target_mean": target_mean,
            "upside_pct": round(upside_pct, 1),
            "revenue_growth": revenue_growth,
            "earnings_growth": fnd.get("earnings_growth"),
            "gross_margin": fnd.get("gross_margin"),
            "operating_margin": fnd.get("operating_margin"),
            "free_cashflow": fnd.get("free_cashflow"),
            "operating_cashflow": fnd.get("operating_cashflow"),
            "total_debt": fnd.get("total_debt"),
            "total_cash": fnd.get("total_cash"),
            "roic": fnd.get("roic"),
            "shares_outstanding": fnd.get("shares_outstanding"),
            "market_cap": fnd.get("market_cap"),
            "enterprise_value": fnd.get("enterprise_value"),
            "sector": fnd.get("sector"),
            "current_price": price,
            "screen_source": "S&P 500 fundamentals/valuation; no momentum or return filters",
        })

    candidates.sort(key=lambda item: item["upside_pct"], reverse=True)
    return candidates


def prioritize_fundamental_screen_candidates(candidates, limit=20):
    """按可得的基本面品質訊號排序；目標價上檔僅作次要排序依據。"""
    quality_fields = (
        "free_cashflow", "operating_cashflow", "roic",
        "gross_margin", "operating_margin",
    )
    ranked = []
    for candidate in candidates:
        observed = [candidate[field] for field in quality_fields if candidate.get(field) is not None]
        score = sum(value > 0 for value in observed)
        ranked.append({
            **candidate,
            "screen_quality_score": score,
            "screen_quality_coverage": len(observed),
        })
    ranked.sort(
        key=lambda item: (
            item["screen_quality_score"],
            item["screen_quality_coverage"],
            item.get("upside_pct", 0),
        ),
        reverse=True,
    )
    return ranked[:max(0, limit)]


def check_value_exits(value_positions, current_prices, fundamentals_now):
    """四層出場判斷，依序檢查、互斥（一次只觸發一種原因）

    Args:
        value_positions: portfolio["value_positions"]，{symbol: {shares, avg_price,
                          entry_date, entry_target_price, ...}}
        current_prices: {symbol: price}
        fundamentals_now: {symbol: {...}}，來自 get_fundamentals_batch()，用於偵測目標價下修

    Returns:
        dict: {symbol: {"reason": "hard_stop"|"take_profit"|"fundamental_break"|"time_stop",
                         "current_price": float, "detail": str}}
    """
    exits = {}
    today = _date.today()

    for symbol, pos in value_positions.items():
        price = current_prices.get(symbol)
        if price is None:
            continue

        avg_price = pos.get("avg_price", 0)
        entry_target = pos.get("entry_target_price")
        pnl_pct = (price / avg_price - 1) if avg_price else 0

        # 1. 硬停損
        if pnl_pct <= VALUE_HARD_STOP:
            exits[symbol] = {
                "reason": "hard_stop",
                "current_price": price,
                "detail": f"從成本 ${avg_price:.2f} 下跌 {pnl_pct*100:.1f}%，觸及硬停損 {VALUE_HARD_STOP*100:.0f}%",
            }
            continue

        # 2. 停利：股價達目標價 90%
        if entry_target and price >= entry_target * VALUE_TAKE_PROFIT_PCT:
            exits[symbol] = {
                "reason": "take_profit",
                "current_price": price,
                "detail": f"股價 ${price:.2f} 已達進場時目標價 ${entry_target:.2f} 的 {price/entry_target*100:.0f}%，論點已實現",
            }
            continue

        # 3. 基本面轉壞：目標價被下修超過門檻
        fnd = fundamentals_now.get(symbol)
        if fnd and entry_target:
            new_target = fnd.get("target_mean")
            if new_target and entry_target > 0:
                target_change = new_target / entry_target - 1
                if target_change <= VALUE_TARGET_CUT_PCT:
                    exits[symbol] = {
                        "reason": "fundamental_break",
                        "current_price": price,
                        "detail": f"分析師目標價從 ${entry_target:.2f} 下修至 ${new_target:.2f}（{target_change*100:.1f}%），論點被破壞",
                    }
                    continue

        # 4. 時間停損：6 個月內盤整未突破
        entry_date_str = pos.get("entry_date")
        if entry_date_str:
            try:
                entry_date = datetime.fromisoformat(entry_date_str).date()
                held_days = (today - entry_date).days
                if held_days >= VALUE_TIME_STOP_DAYS and abs(pnl_pct) < VALUE_TIME_STOP_BAND:
                    exits[symbol] = {
                        "reason": "time_stop",
                        "current_price": price,
                        "detail": f"持有 {held_days} 天，股價仍在 ±{VALUE_TIME_STOP_BAND*100:.0f}% 盤整區間，釋放資金",
                    }
                    continue
            except Exception:
                pass

    return exits


def check_value_graduation(value_positions, momentum_rank_map, alpha_1y_map, alpha_3y_map, rank_threshold=50):
    """判斷是否有價值池持倉「畢業」轉入主池

    畢業條件：重用主池 ADD 的雙 alpha 門檻（1Y alpha>0 and 3Y alpha>-30%，
    見 src/premarket.py 的 _alpha_qualifies）且動能排名進入前 rank_threshold 名。
    這是內部轉籍（bookkeeping），不是真實交易，不經過 confirm 流程。

    Returns:
        list[str]: 可畢業的 symbol 清單
    """
    graduates = []
    for symbol in value_positions:
        a1y = alpha_1y_map.get(symbol)
        a3y = alpha_3y_map.get(symbol)
        if a1y is not None and a1y <= 0:
            continue
        if a3y is not None and a3y < -30:
            continue
        rank = momentum_rank_map.get(symbol)
        if rank is None or rank > rank_threshold:
            continue
        graduates.append(symbol)
    return graduates
