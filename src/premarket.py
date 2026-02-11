import math
from src.risk import check_all_exit_conditions, check_position_limit

VERSION = "0.4.0"  # 三層出場策略版本


def generate_actions(portfolio, current_prices, ma200_prices=None, momentum_ranks=None):
    """盤前決策引擎（動能策略 + 三層出場）

    Args:
        portfolio: 持倉狀態 dict
        current_prices: {symbol: price} 最新報價
        ma200_prices: {symbol: ma200_value} MA200 資料
        momentum_ranks: [{"symbol": str, "momentum": float, "rank": int}, ...]

    Returns:
        actions: list of action dicts
    """
    if ma200_prices is None:
        ma200_prices = {}
    if momentum_ranks is None:
        momentum_ranks = []

    actions = []
    action_id = 0
    positions = portfolio.get("positions", {})

    # 建立動能查詢表
    momentum_map = {m["symbol"]: m for m in momentum_ranks}

    # === 1. 三層出場條件檢查 ===
    # 1. 移動停利（-15% from high）
    # 2. MA200 停損
    # 3. 極端停損（-35% from cost）
    exit_signals = check_all_exit_conditions(
        positions, current_prices, ma200_prices,
        trailing_threshold=-0.15, hard_threshold=-0.35
    )

    # === 2. 遍歷所有持倉，產出 HOLD / EXIT ===
    for symbol, pos in positions.items():
        price = current_prices.get(symbol)
        pnl_pct = None
        if price is not None and pos["avg_price"] > 0:
            pnl_pct = round((price - pos["avg_price"]) / pos["avg_price"] * 100, 2)

        action_id += 1

        # 取得該持倉的動能資訊
        m_info = momentum_map.get(symbol, {})
        momentum = m_info.get("momentum")
        rank = m_info.get("rank")

        # 取得最高價資訊
        high_price = pos.get("high_since_entry")

        if pos.get("core", False):
            # 核心持倉：永遠 HOLD
            actions.append({
                "id": action_id,
                "action": "HOLD",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "high_since_entry": high_price,
                "pnl_pct": pnl_pct,
                "momentum": momentum,
                "reason": "核心持倉",
                "source": "core_hold",
                "status": "auto",
            })
        elif symbol in exit_signals:
            # 觸發出場條件
            exit_info = exit_signals[symbol]
            actions.append({
                "id": action_id,
                "action": "EXIT",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "high_since_entry": high_price,
                "pnl_pct": pnl_pct,
                "momentum": momentum,
                "reason": exit_info["message"],
                "source": exit_info["reason"],
                "status": "pending",
            })
        else:
            # 繼續持有（讓獲利奔跑）
            reason = "持有中"
            if momentum is not None:
                if momentum > 10:
                    reason = f"持有中，動能強勁 (+{momentum:.1f}%)"
                elif momentum > 0:
                    reason = f"持有中，動能正向 (+{momentum:.1f}%)"
                else:
                    reason = f"持有中，動能偏弱 ({momentum:.1f}%)"

            actions.append({
                "id": action_id,
                "action": "HOLD",
                "symbol": symbol,
                "shares": pos["shares"],
                "current_price": price,
                "avg_price": pos["avg_price"],
                "high_since_entry": high_price,
                "pnl_pct": pnl_pct,
                "momentum": momentum,
                "momentum_rank": rank,
                "reason": reason,
                "source": "momentum",
                "status": "auto",
            })

    # === 3. 新增買入候選（依動能排名） ===
    available_slots = check_position_limit(portfolio)
    if available_slots > 0 and momentum_ranks:
        cash = portfolio.get("cash", 0)
        position_size = cash / max(available_slots, 1) if cash > 0 else 0

        # 篩選：動能 > 0 + 尚未持有
        buy_candidates = [
            m for m in momentum_ranks
            if m.get("momentum", 0) > 0
            and m["symbol"] not in positions
        ]
        # 已經按動能排序，最多顯示 5 檔（避免資訊過載）
        max_add = min(5, available_slots)

        for m in buy_candidates[:max_add]:
            action_id += 1
            symbol = m["symbol"]
            momentum = m["momentum"]
            rank = m["rank"]
            price = current_prices.get(symbol, 0)

            if price > 0 and position_size > 0:
                suggested_shares = math.floor(position_size / price)
            else:
                suggested_shares = 0

            # 組裝原因
            reason = f"動能排名 #{rank}（+{momentum:.1f}%）"
            if suggested_shares == 0:
                reason += "（現金不足）"

            actions.append({
                "id": action_id,
                "action": "ADD",
                "symbol": symbol,
                "suggested_shares": suggested_shares,
                "current_price": price,
                "momentum": momentum,
                "momentum_rank": rank,
                "reason": reason,
                "source": "momentum",
                "status": "pending",
            })

    return actions
