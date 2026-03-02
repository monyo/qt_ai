import math
from datetime import date
from src.risk import check_all_exit_conditions, check_position_limit

VERSION = "0.9.0"  # 混合動能 50/50（21天+252天），回測 CAGR +62% vs 純21天 +48%

# 汰弱留強參數（參考學術研究的定期重新排名邏輯）
ROTATE_MOMENTUM_DIFF = 10      # 動能差距門檻 (%)，從 20% 降至 10%
ROTATE_HOLDING_DAYS_MIN = 30   # 最少持有天數，從 60 天降至 30 天


def generate_actions(portfolio, current_prices, ma200_prices=None, momentum_ranks=None, alpha_1y_map=None, trend_state_map=None, alpha_3y_map=None):
    """盤前決策引擎（動能策略 + 三層出場 + 趨勢狀態）

    Args:
        portfolio: 持倉狀態 dict
        current_prices: {symbol: price} 最新報價
        ma200_prices: {symbol: ma200_value} MA200 資料
        momentum_ranks: [{"symbol": str, "momentum": float, "rank": int}, ...]
        alpha_1y_map: {symbol: alpha_1y} 1 年超額報酬（vs SPY）
        trend_state_map: {symbol: {bounce_pct, from_high_pct, state}} 趨勢狀態
        alpha_3y_map: {symbol: alpha_3y} 3 年超額報酬（vs SPY）

    Returns:
        actions: list of action dicts
    """
    if ma200_prices is None:
        ma200_prices = {}
    if momentum_ranks is None:
        momentum_ranks = []
    if alpha_1y_map is None:
        alpha_1y_map = {}
    if trend_state_map is None:
        trend_state_map = {}
    if alpha_3y_map is None:
        alpha_3y_map = {}

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

        # 取得 1 年 / 3 年超額報酬
        alpha_1y = alpha_1y_map.get(symbol)
        alpha_3y = alpha_3y_map.get(symbol)

        # 取得趨勢狀態
        trend_state = trend_state_map.get(symbol)

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
                "trend_state": trend_state,
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
                "trend_state": trend_state,
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

            # 趨勢狀態警告
            if trend_state:
                if trend_state["state"] == "轉弱" and (momentum is not None and momentum > 0):
                    reason += " ⚠️ 倒V警告"
                elif trend_state["state"] == "轉強" and (momentum is not None and momentum < 0):
                    reason += " 💡 V轉回升中"

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
                "trend_state": trend_state,
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

    num_to_add = 0  # 供後續 post-rotate 計算使用

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
        ADD_BACKUP = 3  # 額外備選數，供用戶替換 1Y/3Y alpha 差的標的
        num_to_add = min(5, available_slots, len(buy_candidates))

        # 根據實際要買的檔數來決定單一部位大小
        if num_to_add > 0:
            position_size = projected_cash / num_to_add
        else:
            position_size = 0

        # 已經按動能排序，提出建議（主要候選 + 備選）
        # 備選過濾：3Y alpha 必須 >= 0，才是有效替代品
        primary_candidates = buy_candidates[:num_to_add]
        backup_pool = buy_candidates[num_to_add:]
        valid_backups = [
            m for m in backup_pool
            if alpha_3y_map.get(m["symbol"]) is None or alpha_3y_map.get(m["symbol"]) >= 0
        ][:ADD_BACKUP]
        candidates_to_show = [(m, False) for m in primary_candidates] + [(m, True) for m in valid_backups]

        for m, is_backup in candidates_to_show:
            action_id += 1
            symbol = m["symbol"]
            momentum = m["momentum"]
            rsi = m.get("rsi")
            rank = m["rank"]
            price = current_prices.get(symbol, 0)
            alpha_1y = alpha_1y_map.get(symbol)
            alpha_3y = alpha_3y_map.get(symbol)

            if price > 0 and position_size > 0:
                suggested_shares = math.floor(position_size / price)
            else:
                suggested_shares = 0

            # 組裝原因（加入警示）
            reason = f"動能排名 #{rank}（+{momentum:.1f}%）"
            if is_backup:
                reason = f"[備選] {reason}"
            if suggested_shares == 0 and not is_backup:
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
                "suggested_shares": 0 if is_backup else suggested_shares,
                "current_price": price,
                "momentum": momentum,
                "rsi": rsi,
                "momentum_rank": rank,
                "alpha_1y": alpha_1y,
                "alpha_3y": alpha_3y,
                "is_backup": is_backup,
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
                        "buy_alpha_3y": alpha_3y_map.get(candidate_symbol),
                        "momentum_diff": round(momentum_diff, 1),
                        "reason": f"汰弱留強：動能差 +{momentum_diff:.0f}%（持有 {holding_days} 天）",
                        "source": "rotate",
                        "status": "pending",
                    })
                    rotate_used_candidates.add(candidate_symbol)
                    break  # 這個持倉已配對，換下一個

    # === 5. 計算 ROTATE 後的 ADD 股數（供參考） ===
    rotates = [a for a in actions if a["action"] == "ROTATE"]
    if rotates and num_to_add > 0:
        rotate_proceeds = sum(
            a["sell_shares"] * a["sell_price"] * CASH_SAFETY_FACTOR
            for a in rotates
        )
        post_rotate_position_size = (projected_cash + rotate_proceeds) / num_to_add
        for a in actions:
            if a["action"] == "ADD":
                price = a.get("current_price", 0)
                if price > 0:
                    a["suggested_shares_post_rotate"] = math.floor(post_rotate_position_size / price)

    return actions


# 增持參考參數
TOPUP_MOMENTUM_MIN = 15   # 最低動能門檻 (%)
TOPUP_MAX_WEIGHT = 0.02   # 倉位比重上限（超過此比重不建議加碼）
FIXED_STOP_LOSS = 0.85    # 停損係數（-15%）


def generate_topup_suggestions(portfolio, current_prices, momentum_ranks, alpha_1y_map, trend_state_map, total_value, alpha_3y_map=None):
    """掃描現有持倉，找出倉位偏小且值得加碼的標的

    條件：非核心、動能 > 15%、趨勢轉強、倉位比重 < 2%
    關鍵指標：追高幅度、加碼後新停損 vs 原始成本

    Returns:
        list of suggestion dicts，按動能排序
    """
    if alpha_3y_map is None:
        alpha_3y_map = {}
    suggestions = []
    positions = portfolio.get("positions", {})
    momentum_map = {m["symbol"]: m for m in momentum_ranks}

    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue

        price = current_prices.get(symbol)
        if not price or price <= 0:
            continue

        # 倉位比重過濾
        current_weight = (price * pos["shares"]) / total_value if total_value > 0 else 0
        if current_weight >= TOPUP_MAX_WEIGHT:
            continue

        # 動能過濾
        m_info = momentum_map.get(symbol, {})
        momentum = m_info.get("momentum")
        if momentum is None or momentum < TOPUP_MOMENTUM_MIN:
            continue

        # 趨勢過濾（只建議轉強標的）
        trend_state = trend_state_map.get(symbol, {})
        if trend_state.get("state") != "轉強":
            continue

        # 追高風險評估
        avg_price = pos["avg_price"]
        run_up_pct = (price - avg_price) / avg_price * 100
        new_stop = round(price * FIXED_STOP_LOSS, 2)
        stop_vs_cost = new_stop - avg_price

        stop_pct_vs_cost = stop_vs_cost / avg_price * 100  # 負數=停損低於成本多少%

        if stop_vs_cost >= 0:
            safety = "🟢 安全"
            safety_note = f"停損${new_stop:.2f} 高於成本（原始獲利鎖住）"
        elif stop_pct_vs_cost >= -8:
            safety = "🟡 謹慎"
            safety_note = f"停損${new_stop:.2f} 低於成本 {stop_pct_vs_cost:.1f}%（風險可控）"
        else:
            safety = "🔴 風險高"
            safety_note = f"停損${new_stop:.2f} 低於成本 {stop_pct_vs_cost:.1f}%（原始成本曝險）"

        suggestions.append({
            "symbol": symbol,
            "current_price": price,
            "avg_price": avg_price,
            "shares": pos["shares"],
            "current_weight_pct": round(current_weight * 100, 1),
            "run_up_pct": round(run_up_pct, 1),
            "momentum": momentum,
            "momentum_rank": m_info.get("rank"),
            "trend_state": trend_state,
            "new_stop": new_stop,
            "stop_vs_cost": round(stop_vs_cost, 2),
            "safety": safety,
            "safety_note": safety_note,
            "alpha_1y": alpha_1y_map.get(symbol),
            "alpha_3y": alpha_3y_map.get(symbol),
        })

    suggestions.sort(key=lambda x: x["momentum"], reverse=True)
    return suggestions
