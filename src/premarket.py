import math
from datetime import date
from src.risk import check_all_exit_conditions, check_position_limit

VERSION = "0.7.2"  # RSI 只警告不過濾（回測顯示飆股超買仍可續漲）

# 汰弱留強參數（參考學術研究的定期重新排名邏輯）
ROTATE_MOMENTUM_DIFF = 10      # 動能差距門檻 (%)，從 20% 降至 10%
ROTATE_HOLDING_DAYS_MIN = 30   # 最少持有天數，從 60 天降至 30 天


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

    # RSI 警告參數（回測顯示飆股超買後仍可續漲，故只警告不過濾）
    RSI_OVERBOUGHT = 75  # RSI > 75 超買警告
    RSI_EXTREME = 80     # RSI > 80 極度超買警告

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
            rsi = m.get("rsi")
            rank = m["rank"]
            price = current_prices.get(symbol, 0)
            alpha_1y = alpha_1y_map.get(symbol)

            if price > 0 and position_size > 0:
                suggested_shares = math.floor(position_size / price)
            else:
                suggested_shares = 0

            # 組裝原因（加入警示）
            reason = f"動能排名 #{rank}（+{momentum:.1f}%）"
            if suggested_shares == 0:
                reason += "（現金不足）"
            if rsi is not None and rsi > RSI_EXTREME:
                reason += f" 🔴 RSI {rsi:.0f} 極度超買"
            elif rsi is not None and rsi > RSI_OVERBOUGHT:
                reason += f" 🟡 RSI {rsi:.0f} 超買"
            if alpha_1y is not None and alpha_1y < -20:
                reason += f" ⚠️ 1年落後大盤 {alpha_1y:.0f}%"

            actions.append({
                "id": action_id,
                "action": "ADD",
                "symbol": symbol,
                "suggested_shares": suggested_shares,
                "current_price": price,
                "momentum": momentum,
                "rsi": rsi,
                "momentum_rank": rank,
                "alpha_1y": alpha_1y,
                "reason": reason,
                "source": "momentum",
                "status": "pending",
            })

    # === 4. 汰弱留強：主動建議換股（不限於現金不足） ===
    # 找出持倉中動能最弱的非核心、非偏愛股票
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

    # 可換股的持倉：排除核心、偏愛、已出場、保護期內
    rotatable_positions = [
        (sym, pos, momentum_map.get(sym, {}).get("momentum"), get_holding_days(pos))
        for sym, pos in positions.items()
        if not pos.get("core", False)
        and not pos.get("favorite", False)  # 排除偏愛標的
        and sym not in exit_symbols
        and momentum_map.get(sym, {}).get("momentum") is not None
        and get_holding_days(pos) >= ROTATE_HOLDING_DAYS_MIN
    ]

    # 按動能排序（最弱的在前）
    rotatable_positions.sort(key=lambda x: x[2] if x[2] is not None else 999)

    # 找出所有強勢候選（不只是現金不足的）
    strong_candidates = [
        m for m in momentum_ranks
        if m.get("momentum", 0) > 0
        and m["symbol"] not in positions
        and m["symbol"] not in exit_symbols
    ]

    # 對每個弱勢持倉，檢查是否有夠強的候選可換
    rotate_used_candidates = set()  # 已被配對的候選
    for sym, pos, pos_momentum, holding_days in rotatable_positions:
        pos_price = current_prices.get(sym, 0)
        if pos_price <= 0:
            continue

        # 找一個還沒被配對的強勢候選
        for candidate in strong_candidates:
            if candidate["symbol"] in rotate_used_candidates:
                continue

            candidate_momentum = candidate.get("momentum", 0)
            candidate_symbol = candidate["symbol"]
            candidate_price = current_prices.get(candidate_symbol, 0)

            # 條件：候選動能 - 持倉動能 > 門檻
            momentum_diff = candidate_momentum - (pos_momentum or 0)
            if momentum_diff > ROTATE_MOMENTUM_DIFF:
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
                        "buy_alpha_1y": alpha_1y_map.get(candidate_symbol),
                        "momentum_diff": round(momentum_diff, 1),
                        "reason": f"汰弱留強：動能差 +{momentum_diff:.0f}%（持有 {holding_days} 天）",
                        "source": "rotate",
                        "status": "pending",
                    })
                    rotate_used_candidates.add(candidate_symbol)
                    break  # 這個持倉已配對，換下一個

    return actions
