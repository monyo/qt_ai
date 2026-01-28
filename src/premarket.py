import math
from src.risk import check_stop_loss, check_position_limit

VERSION = "0.2.0"


def generate_actions(portfolio, current_prices, candidates=None, sentiment_scores=None):
    """盤前決策引擎

    Args:
        portfolio: 持倉狀態 dict
        current_prices: {symbol: price} 最新報價
        candidates: scan_candidates() 回傳的候選股列表 (可選)
        sentiment_scores: {symbol: {"score": float, "reason": str}} (可選)

    Returns:
        actions: list of action dicts
    """
    if candidates is None:
        candidates = []
    if sentiment_scores is None:
        sentiment_scores = {}

    actions = []
    action_id = 0
    positions = portfolio.get("positions", {})

    # === 1. 停損檢查（優先序最高） ===
    stop_loss_list = check_stop_loss(positions, current_prices)
    stop_loss_symbols = {item["symbol"] for item in stop_loss_list}

    # === 2. 從 candidates 取得賣出訊號 ===
    sell_signal_symbols = set()
    for c in candidates:
        if c.get("has_sell_signal"):
            sell_signal_symbols.add(c["Symbol"])

    # === 3. 遍歷所有持倉，產出 HOLD / EXIT ===
    for symbol, pos in positions.items():
        price = current_prices.get(symbol)
        pnl_pct = None
        if price is not None and pos["avg_price"] > 0:
            pnl_pct = round((price - pos["avg_price"]) / pos["avg_price"] * 100, 2)

        action_id += 1

        if pos.get("core", False):
            # 核心持倉：永遠 HOLD
            actions.append({
                "id": action_id,
                "action": "HOLD",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "pnl_pct": pnl_pct,
                "reason": "核心持倉",
                "source": "core_hold",
                "status": "auto",
            })
        elif symbol in stop_loss_symbols:
            # 硬停損觸發
            sl = next(item for item in stop_loss_list if item["symbol"] == symbol)
            actions.append({
                "id": action_id,
                "action": "EXIT",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "pnl_pct": pnl_pct,
                "reason": f"硬停損觸發（{sl['pnl_pct']}%）",
                "source": "stop_loss",
                "status": "pending",
            })
        elif symbol in sell_signal_symbols:
            # 技術面賣出訊號
            actions.append({
                "id": action_id,
                "action": "EXIT",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "pnl_pct": pnl_pct,
                "reason": "技術面賣出訊號（MA60/RSI）",
                "source": "strategy_signal",
                "status": "pending",
            })
        else:
            # 繼續持有
            reason = "持有中"
            # 嘗試附加技術面摘要
            c_match = next((c for c in candidates if c["Symbol"] == symbol), None)
            if c_match and c_match.get("has_today_signal"):
                reason = "持有中，技術面持續看多"

            actions.append({
                "id": action_id,
                "action": "HOLD",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "pnl_pct": pnl_pct,
                "reason": reason,
                "source": "strategy_signal",
                "status": "auto",
            })

    # === 4. 新增買入候選 ===
    available_slots = check_position_limit(portfolio)
    if available_slots > 0 and candidates:
        cash = portfolio.get("cash", 0)
        position_size = cash / max(available_slots, 1) if cash > 0 else 0

        # 篩選：有今日買入訊號 + 歷史報酬正 + 尚未持有
        buy_candidates = [
            c for c in candidates
            if c.get("has_today_signal")
            and c.get("Return%", -999) > 0
            and c["Symbol"] not in positions
        ]
        # 依 Return% 排序
        buy_candidates.sort(key=lambda x: x.get("Return%", 0), reverse=True)

        for c in buy_candidates[:available_slots]:
            action_id += 1
            symbol = c["Symbol"]
            price = current_prices.get(symbol, c.get("Price", 0))
            sentiment = sentiment_scores.get(symbol, {})
            sentiment_score = float(sentiment.get("score", 0.0))
            sentiment_reason = sentiment.get("reason", "")

            if price > 0 and position_size > 0:
                suggested_shares = math.floor(position_size / price)
            else:
                suggested_shares = 0

            # 組裝原因
            reason_parts = []
            if c.get("has_today_signal"):
                reason_parts.append("技術面買入訊號")
            if sentiment_score > 0.3:
                reason_parts.append(f"AI 情緒 {sentiment_score}（看多）")
            elif sentiment_score < -0.3:
                reason_parts.append(f"AI 情緒 {sentiment_score}（看空）")
            elif sentiment_reason:
                reason_parts.append(f"AI 情緒 {sentiment_score}（中立）")
            reason = " + ".join(reason_parts) if reason_parts else "技術面買入訊號"

            if suggested_shares == 0:
                reason += "（現金不足）"

            # 判斷 source
            source = c.get("source", "scanner")

            actions.append({
                "id": action_id,
                "action": "ADD",
                "symbol": symbol,
                "suggested_shares": suggested_shares,
                "current_price": price,
                "reason": reason,
                "source": source,
                "sentiment": sentiment_score,
                "backtest_return_pct": c.get("Return%"),
                "status": "pending",
            })

    return actions
