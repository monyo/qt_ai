import math
from datetime import date
from src.risk import check_all_exit_conditions, check_position_limit, TRANCHE_PARAMS
from src.portfolio import _ensure_tranches

VERSION = "0.10.0"  # 金字塔加碼策略（差異停損），取代 TOPUP 機制

# 汰弱留強參數（參考學術研究的定期重新排名邏輯）
ROTATE_MOMENTUM_DIFF = 10      # 動能差距門檻 (%)，從 20% 降至 10%
ROTATE_HOLDING_DAYS_MIN = 30   # 最少持有天數，從 60 天降至 30 天

# 金字塔加碼參數
MAX_PYRAMID = 5       # 最大批次數（同回測最佳參數）
MAX_PYRAMID_SHOW = 3  # 每日最多顯示的金字塔加碼建議數


def generate_actions(portfolio, current_prices, ma200_prices=None, momentum_ranks=None, alpha_1y_map=None, trend_state_map=None, alpha_3y_map=None, market_regime="BULL"):
    """盤前決策引擎（動能策略 + 三層出場 + 趨勢狀態 + 市場體制）

    Args:
        portfolio: 持倉狀態 dict
        current_prices: {symbol: price} 最新報價
        ma200_prices: {symbol: ma200_value} MA200 資料
        momentum_ranks: [{"symbol": str, "momentum": float, "rank": int}, ...]
        alpha_1y_map: {symbol: alpha_1y} 1 年超額報酬（vs SPY）
        trend_state_map: {symbol: {bounce_pct, from_high_pct, state}} 趨勢狀態
        alpha_3y_map: {symbol: alpha_3y} 3 年超額報酬（vs SPY）
        market_regime: "BULL" 或 "BEAR"（SPY vs MA200）
                       BEAR 時停止產出 ADD / ROTATE，只做 HOLD / EXIT

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
            action_id += 1
            _ensure_tranches(pos)
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
                "tranches": pos["tranches"],
            })
        elif symbol in exit_signals:
            # 逐批出場：為每個觸發批次建立 EXIT action
            for exit_item in exit_signals[symbol]:
                action_id += 1
                actions.append({
                    "id": action_id,
                    "action": "EXIT",
                    "symbol": symbol,
                    "shares": exit_item["tranche_shares"],
                    "tranche_n": exit_item["tranche_n"],
                    "current_price": price,
                    "avg_price": pos["avg_price"],
                    "high_since_entry": high_price,
                    "pnl_pct": pnl_pct,
                    "momentum": momentum,
                    "alpha_1y": alpha_1y,
                    "trend_state": trend_state,
                    "reason": exit_item["message"],
                    "source": exit_item["reason"],
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

            action_id += 1
            _ensure_tranches(pos)
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
                "tranches": pos["tranches"],
            })

    # === 3. 新增買入候選（依動能排名，BEAR 市場暫停） ===
    if market_regime == "BEAR":
        return actions  # BEAR 體制：只保留 HOLD / EXIT，不建議新增或換股

    # === 3. 新增買入候選（依動能排名） ===
    # 計算預估可用現金（假設 EXIT 全部執行，取 85% 避免價差）
    CASH_SAFETY_FACTOR = 0.85
    exit_proceeds = sum(
        current_prices.get(a["symbol"], 0) * a["shares"]
        for a in actions if a["action"] == "EXIT"
    )
    # 只有「完整出場」的 symbol 才釋放槽位（全部批次都觸發）
    exit_symbols_full = set()
    for sym, exit_list in exit_signals.items():
        pos = positions.get(sym, {})
        total_exit_shares = sum(item["tranche_shares"] for item in exit_list)
        if total_exit_shares >= pos.get("shares", 0):
            exit_symbols_full.add(sym)
    exit_count = len(exit_symbols_full)

    current_cash = portfolio.get("cash", 0)
    projected_cash = current_cash + exit_proceeds * CASH_SAFETY_FACTOR

    # 可用槽位 = 原本空位 + EXIT 完整釋放的位置
    base_slots = check_position_limit(portfolio)
    available_slots = base_slots + exit_count

    # RSI 警告參數（回測顯示飆股超買後仍可續漲，故只警告不過濾）
    RSI_OVERBOUGHT = 75  # RSI > 75 超買警告
    RSI_EXTREME = 80     # RSI > 80 極度超買警告

    num_to_add = 0  # 供後續 post-rotate 計算使用

    # Alpha 過濾：1Y > 0 AND 3Y > -30%（輕微落後視為宏觀打趴，允許進主清單）
    # 同時用於 ADD 主清單 和 ROTATE 目標過濾
    def _alpha_qualifies(sym):
        a1y = alpha_1y_map.get(sym)
        a3y = alpha_3y_map.get(sym)
        if a1y is not None and a1y <= 0:
            return False
        if a3y is not None and a3y < -30:
            return False
        return True

    if available_slots > 0 and momentum_ranks:
        # 篩選：動能 > 0 + 尚未持有 + 不在 EXIT 名單
        exit_symbols = {a["symbol"] for a in actions if a["action"] == "EXIT"}
        buy_candidates = [
            m for m in momentum_ranks
            if m.get("momentum", 0) > 0
            and m["symbol"] not in positions
            and m["symbol"] not in exit_symbols
        ]

        alpha_good = [m for m in buy_candidates if _alpha_qualifies(m["symbol"])]
        alpha_poor = [m for m in buy_candidates if not _alpha_qualifies(m["symbol"])]

        ADD_BACKUP = 3
        TARGET_PRIMARY = 5

        # 正常：取 alpha_good 前 5；不足時從 alpha_poor 按 1Y alpha 降序補足
        if len(alpha_good) >= TARGET_PRIMARY:
            primary_pool = alpha_good
            leftover_poor = alpha_poor
        else:
            supplement = sorted(
                alpha_poor,
                key=lambda m: (alpha_1y_map.get(m["symbol"]) or -999),
                reverse=True,
            )
            needed = TARGET_PRIMARY - len(alpha_good)
            primary_pool = alpha_good + supplement[:needed]
            leftover_poor = supplement[needed:]

        num_to_add = min(TARGET_PRIMARY, available_slots, len(primary_pool))
        primary_candidates = primary_pool[:num_to_add]
        backup_src = primary_pool[num_to_add:] + leftover_poor
        valid_backups = [
            m for m in backup_src
            if alpha_3y_map.get(m["symbol"]) is None or alpha_3y_map.get(m["symbol"]) >= 0
        ][:ADD_BACKUP]

        # === 金字塔加碼候選（持倉中可加碼的） ===
        exit_symbols_set = {a["symbol"] for a in actions if a["action"] == "EXIT"}
        pyramid_candidates = []
        for sym, p in positions.items():
            if p.get("core") or sym in exit_symbols_set:
                continue
            _ensure_tranches(p)
            if len(p["tranches"]) >= MAX_PYRAMID:
                continue
            mom = momentum_map.get(sym, {}).get("momentum")
            if mom is None or mom <= 0:
                continue
            price_sym = current_prices.get(sym, 0)
            if price_sym <= 0:
                continue
            latest = p["tranches"][-1]
            direction = "up" if price_sym > latest["entry_price"] else "down"
            n_next = len(p["tranches"]) + 1
            stop_type = ("tight_2" if n_next == 2 else "tight_3") if direction == "up" else "standard"
            pyramid_candidates.append({
                "symbol": sym,
                "momentum": mom,
                "direction": direction,
                "tranche_n": n_next,
                "stop_type": stop_type,
                "current_tranche_count": len(p["tranches"]),
                "rank": momentum_map.get(sym, {}).get("rank"),
            })
        pyramid_candidates.sort(key=lambda x: x["momentum"], reverse=True)
        pyramid_show = pyramid_candidates[:MAX_PYRAMID_SHOW]

        # 計算 position_size：新倉 + 金字塔共用等額分配
        total_buying_slots = num_to_add + len(pyramid_show)
        if total_buying_slots > 0:
            position_size = projected_cash / total_buying_slots
        else:
            position_size = 0

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
            is_supplemented = not _alpha_qualifies(symbol)
            if not is_backup and is_supplemented:
                reason += " ⚠️ 補位（alpha 不符主清單標準）"
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

        # === 金字塔加碼 ADD actions ===
        for pc in pyramid_show:
            action_id += 1
            sym = pc["symbol"]
            price = current_prices.get(sym, 0)
            stop_params = TRANCHE_PARAMS[pc["stop_type"]]
            direction_arrow = "↑" if pc["direction"] == "up" else "↓"
            suggested_shares = math.floor(position_size / price) if price > 0 and position_size > 0 else 0
            if pc["direction"] == "up":
                reason = (f"[持倉{direction_arrow}第{pc['tranche_n']}批] "
                          f"動能 +{pc['momentum']:.1f}%  "
                          f"差異停損({stop_params['fixed']*100:.0f}%/{stop_params['trailing']*100:.0f}%)")
            else:
                reason = (f"[持倉{direction_arrow}第{pc['tranche_n']}批] "
                          f"動能 +{pc['momentum']:.1f}%  撿便宜加碼（標準停損）")
            rsi = momentum_map.get(sym, {}).get("rsi")
            if rsi is not None and rsi > RSI_EXTREME:
                reason += f" 🔴 RSI {rsi:.0f} 極度超買"
            elif rsi is not None and rsi > RSI_OVERBOUGHT:
                reason += f" 🟡 RSI {rsi:.0f} 超買"
            actions.append({
                "id": action_id,
                "action": "ADD",
                "is_pyramid": True,
                "tranche_n": pc["tranche_n"],
                "stop_type": pc["stop_type"],
                "direction": pc["direction"],
                "symbol": sym,
                "suggested_shares": suggested_shares,
                "current_price": price,
                "momentum": pc["momentum"],
                "momentum_rank": pc["rank"],
                "rsi": rsi,
                "alpha_1y": alpha_1y_map.get(sym),
                "alpha_3y": alpha_3y_map.get(sym),
                "is_backup": False,
                "reason": reason,
                "source": "pyramid",
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

    # 可換股的持倉：排除核心、偏愛、已出場、保護期內、轉強中
    rotatable_positions = [
        (sym, pos, momentum_map.get(sym, {}).get("momentum"), get_holding_days(pos))
        for sym, pos in positions.items()
        if not pos.get("core", False)
        and not pos.get("favorite", False)  # 排除偏愛標的
        and sym not in exit_symbols
        and momentum_map.get(sym, {}).get("momentum") is not None
        and get_holding_days(pos) >= ROTATE_HOLDING_DAYS_MIN
        and trend_state_map.get(sym, {}).get("state") != "轉強"  # 轉強中不換，動能即將回升
    ]

    # 按動能排序（最弱的在前）
    rotatable_positions.sort(key=lambda x: x[2] if x[2] is not None else 999)

    # 找出所有強勢候選（不只是現金不足的）：同樣套用 alpha 過濾，排除結構衰退標的
    strong_candidates = [
        m for m in momentum_ranks
        if m.get("momentum", 0) > 0
        and m["symbol"] not in positions
        and m["symbol"] not in exit_symbols
        and _alpha_qualifies(m["symbol"])
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
