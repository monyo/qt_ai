import math
from datetime import date
from src.risk import check_all_exit_conditions, check_position_limit

VERSION = "0.6.0"  # Fixed -15% 停損（回測驗證優於 Trailing）

# 汰弱留強參數
ROTATE_MOMENTUM_DIFF = 20      # 動能差距門檻 (%)
ROTATE_HOLDING_DAYS_MIN = 60   # 最少持有天數才允許換股


def generate_actions(portfolio, current_prices, ma200_prices=None, momentum_ranks=None, alpha_1y_map=None):
    """盤前決策引擎（動能策略 + 三層出場）

    Args:
        portfolio: 持倉狀態 dict
        current_prices: {symbol: price} 最新報價
        ma200_prices: {symbol: ma200_value} MA200 資料
        momentum_ranks: [{"symbol": str, "momentum": float, "rank": int}, ...]
        alpha_1y_map: {symbol: alpha_1y} 1 年超額報酬（vs SPY）

    Returns:
        actions: list of action dicts
    """
    if ma200_prices is None:
        ma200_prices = {}
    if momentum_ranks is None:
        momentum_ranks = []
    if alpha_1y_map is None:
        alpha_1y_map = {}

    actions = []
    action_id = 0
    positions = portfolio.get("positions", {})

    # 建立動能查詢表
    momentum_map = {m["symbol"]: m for m in momentum_ranks}

    # === 1. 出場條件檢查 ===
    # 1. 固定停損（-15% from cost）
    # 2. MA200 停損
    # 3. 極端停損（-35% from cost，備用）
    exit_signals = check_all_exit_conditions(
        positions, current_prices, ma200_prices,
        fixed_threshold=-0.15, hard_threshold=-0.35
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

        # 取得 1 年超額報酬
        alpha_1y = alpha_1y_map.get(symbol)

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
                "alpha_1y": alpha_1y,
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
                "alpha_1y": alpha_1y,
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
                "alpha_1y": alpha_1y,
                "reason": reason,
                "source": "momentum",
                "status": "auto",
            })

    # === 3. 新增買入候選（依動能排名） ===
    # 計算預估可用現金（假設 EXIT 全部執行，取 85% 避免價差）
    CASH_SAFETY_FACTOR = 0.85
    exit_proceeds = sum(
        current_prices.get(a["symbol"], 0) * a["shares"]
        for a in actions if a["action"] == "EXIT"
    )
    exit_count = sum(1 for a in actions if a["action"] == "EXIT")

    current_cash = portfolio.get("cash", 0)
    projected_cash = current_cash + exit_proceeds * CASH_SAFETY_FACTOR

    # 可用槽位 = 原本空位 + EXIT 釋放的位置
    base_slots = check_position_limit(portfolio)
    available_slots = base_slots + exit_count

    if available_slots > 0 and momentum_ranks:
        # 篩選：動能 > 0 + 尚未持有 + 不在 EXIT 名單
        exit_symbols = {a["symbol"] for a in actions if a["action"] == "EXIT"}
        buy_candidates = [
            m for m in momentum_ranks
            if m.get("momentum", 0) > 0
            and m["symbol"] not in positions
            and m["symbol"] not in exit_symbols
        ]

        # 策略 B: 集中火力
        # 計算實際要買入的檔數 (最多5檔)
        num_to_add = min(5, available_slots, len(buy_candidates))

        # 根據實際要買的檔數來決定單一部位大小
        if num_to_add > 0:
            position_size = projected_cash / num_to_add
        else:
            position_size = 0

        # 已經按動能排序，提出建議
        for m in buy_candidates[:num_to_add]:
            action_id += 1
            symbol = m["symbol"]
            momentum = m["momentum"]
            rank = m["rank"]
            price = current_prices.get(symbol, 0)
            alpha_1y = alpha_1y_map.get(symbol)

            if price > 0 and position_size > 0:
                suggested_shares = math.floor(position_size / price)
            else:
                suggested_shares = 0

            # 組裝原因（加入長期表現警示）
            reason = f"動能排名 #{rank}（+{momentum:.1f}%）"
            if suggested_shares == 0:
                reason += "（現金不足）"
            if alpha_1y is not None and alpha_1y < -20:
                reason += f"⚠️ 1年落後大盤 {alpha_1y:.0f}%"

            actions.append({
                "id": action_id,
                "action": "ADD",
                "symbol": symbol,
                "suggested_shares": suggested_shares,
                "current_price": price,
                "momentum": momentum,
                "momentum_rank": rank,
                "alpha_1y": alpha_1y,
                "reason": reason,
                "source": "momentum",
                "status": "pending",
            })

    # === 4. 汰弱留強：現金不足時建議換股 ===
    # 找出「現金不足」的強勢候選
    cash_short_candidates = [
        a for a in actions
        if a["action"] == "ADD" and a.get("suggested_shares", 0) == 0
    ]

    if cash_short_candidates:
        # 找出持倉中動能最弱的非核心股票（排除已在 EXIT 的）
        exit_symbols = {a["symbol"] for a in actions if a["action"] == "EXIT"}
        today = date.today()

        def get_holding_days(pos):
            """計算持有天數"""
            first_entry = pos.get("first_entry")
            if first_entry:
                try:
                    entry_date = date.fromisoformat(first_entry)
                    return (today - entry_date).days
                except ValueError:
                    return 999  # 無法解析就當作很久了
            return 999

        holdable_positions = [
            (sym, pos, momentum_map.get(sym, {}).get("momentum"), get_holding_days(pos))
            for sym, pos in positions.items()
            if not pos.get("core", False)
            and sym not in exit_symbols
            and momentum_map.get(sym, {}).get("momentum") is not None
            and get_holding_days(pos) >= ROTATE_HOLDING_DAYS_MIN  # 60天保護期
        ]

        # 按動能排序（最弱的在前）
        holdable_positions.sort(key=lambda x: x[2] if x[2] is not None else 999)

        # 對每個現金不足的候選，嘗試配對最弱持倉
        rotate_used = set()  # 已被配對的持倉
        for candidate in cash_short_candidates:
            candidate_momentum = candidate.get("momentum", 0)
            candidate_symbol = candidate["symbol"]
            candidate_price = candidate.get("current_price", 0)

            # 找一個還沒被配對的最弱持倉
            for sym, pos, pos_momentum, holding_days in holdable_positions:
                if sym in rotate_used:
                    continue

                # 條件：候選動能 - 持倉動能 > 20%（差距夠大才值得換）
                momentum_diff = candidate_momentum - (pos_momentum or 0)
                if momentum_diff > ROTATE_MOMENTUM_DIFF:
                    pos_price = current_prices.get(sym, 0)
                    pos_value = pos_price * pos["shares"]

                    # 計算換股後可買幾股
                    if candidate_price > 0:
                        new_shares = math.floor(pos_value * CASH_SAFETY_FACTOR / candidate_price)
                    else:
                        new_shares = 0

                    if new_shares > 0:
                        action_id += 1
                        actions.append({
                            "id": action_id,
                            "action": "ROTATE",
                            "sell_symbol": sym,
                            "sell_shares": pos["shares"],
                            "sell_price": pos_price,
                            "sell_momentum": pos_momentum,
                            "sell_pnl_pct": round((pos_price - pos["avg_price"]) / pos["avg_price"] * 100, 2) if pos["avg_price"] > 0 else None,
                            "sell_holding_days": holding_days,
                            "buy_symbol": candidate_symbol,
                            "buy_shares": new_shares,
                            "buy_price": candidate_price,
                            "buy_momentum": candidate_momentum,
                            "buy_alpha_1y": candidate.get("alpha_1y"),
                            "momentum_diff": round(momentum_diff, 1),
                            "reason": f"汰弱留強：動能差 +{momentum_diff:.0f}%（持有 {holding_days} 天）",
                            "source": "rotate",
                            "status": "pending",
                        })
                        rotate_used.add(sym)
                        break  # 這個候選已配對，換下一個

    return actions
