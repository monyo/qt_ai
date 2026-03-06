import argparse
import json
import math
import os
from datetime import datetime, date

import pandas as pd
import yfinance as yf

from src.portfolio import (
    load_portfolio, save_portfolio, get_individual_count,
    load_watchlist, save_watchlist, add_to_watchlist,
    update_high_prices, initialize_high_prices,
)
from src.data_loader import get_sp500_tickers, fetch_current_prices, get_tw50_tickers, TW_STOCK_NAMES
from src.tw_scanner import get_tw_liquid_tickers, scan_tw_market
from src.risk import check_position_limit
from src.premarket import generate_actions, generate_topup_suggestions, VERSION
from src.sector_monitor import get_sector_summary, check_holdings_sector_exposure
from src.snapshot import load_snapshot, calculate_yearly_pnl, create_year_start_snapshot, save_snapshot
from src.momentum import rank_by_momentum, print_momentum_report, calculate_alpha_batch, calculate_alpha_3y_batch, calculate_trend_state_batch
from src.notifier import GmailNotifier


def get_spy_regime():
    """檢查 SPY 是否高於 200 日均線，判斷當前市場體制

    Returns:
        dict: {
            "regime": "BULL" or "BEAR",
            "is_bull": bool,
            "spy_price": float,
            "ma200": float,
            "pct_vs_ma200": float,   # SPY 高於/低於 MA200 的百分比
        }
    """
    try:
        df = yf.Ticker("SPY").history(period="1y")
        if len(df) < 200:
            return {"regime": "BULL", "is_bull": True, "spy_price": None, "ma200": None, "pct_vs_ma200": None}
        spy_price = round(df["Close"].iloc[-1], 2)
        ma200 = round(df["Close"].rolling(200).mean().iloc[-1], 2)
        pct = round((spy_price - ma200) / ma200 * 100, 2)
        is_bull = bool(spy_price > ma200)
        return {
            "regime": "BULL" if is_bull else "BEAR",
            "is_bull": is_bull,
            "spy_price": float(spy_price),
            "ma200": float(ma200),
            "pct_vs_ma200": float(pct),
        }
    except Exception:
        return {"regime": "BULL", "is_bull": True, "spy_price": None, "ma200": None, "pct_vs_ma200": None}


def fetch_ma200_prices(symbols):
    """取得多檔標的的 MA200 值

    Args:
        symbols: 股票代碼列表

    Returns:
        dict: {symbol: ma200_value}
    """
    ma200_prices = {}
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period="1y")
            if len(df) >= 200:
                ma200 = df['Close'].rolling(200).mean().iloc[-1]
                if not pd.isna(ma200):
                    ma200_prices[symbol] = round(ma200, 2)
        except Exception:
            pass
    return ma200_prices


def run_init():
    """互動式建立初始 portfolio.json"""
    print("=== 初始化投資組合 ===\n")

    cash = float(input("請輸入可用現金 (USD): "))
    portfolio = {
        "cash": cash,
        "updated": str(date.today()),
        "positions": {},
        "transactions": [],
    }

    # VOO 核心持倉
    has_voo = input("\n是否持有 VOO？(y/n): ").strip().lower()
    if has_voo == "y":
        shares = int(input("  VOO 股數: "))
        avg_price = float(input("  VOO 平均成本: "))
        entry_date = input("  VOO 首次買入日期 (YYYY-MM-DD): ").strip()
        portfolio["positions"]["VOO"] = {
            "shares": shares,
            "avg_price": avg_price,
            "cost_basis": round(avg_price * shares, 2),
            "first_entry": entry_date,
            "core": True,
        }
        portfolio["transactions"].append({
            "date": entry_date, "symbol": "VOO", "action": "ADD",
            "shares": shares, "price": avg_price,
        })

    # 其他個股
    print("\n輸入現有個股持倉（輸入空白 symbol 結束）：")
    while True:
        symbol = input("\n  Symbol (留空結束): ").strip().upper()
        if not symbol:
            break
        shares = int(input(f"  {symbol} 股數: "))
        avg_price = float(input(f"  {symbol} 平均成本: "))
        entry_date = input(f"  {symbol} 首次買入日期 (YYYY-MM-DD): ").strip()

        portfolio["positions"][symbol] = {
            "shares": shares,
            "avg_price": avg_price,
            "cost_basis": round(avg_price * shares, 2),
            "first_entry": entry_date,
            "core": False,
        }
        portfolio["transactions"].append({
            "date": entry_date, "symbol": symbol, "action": "ADD",
            "shares": shares, "price": avg_price,
        })

    save_portfolio(portfolio)
    individual = get_individual_count(portfolio)
    print(f"\n投資組合已建立：{len(portfolio['positions'])} 檔持倉（個股 {individual} 檔），現金 ${cash:,.2f}")
    print(f"已儲存至 data/portfolio.json")


def run_premarket(scan_tw=False):
    """產出盤前建議（動能策略 + 三層出場）

    Args:
        scan_tw: 是否掃描台股（預設 False）
    """
    os.makedirs("data", exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    # 1. 載入持倉
    portfolio = load_portfolio()
    positions = portfolio.get("positions", {})

    if not positions and portfolio.get("cash", 0) == 0:
        print("尚未建立投資組合。請先執行: python premarket_main.py --init")
        return

    print(f"=== 盤前分析 {date.today()} （三層出場策略）===\n")
    individual = get_individual_count(portfolio)
    print(f"持倉：{len(positions)} 檔（個股 {individual}/30），現金 ${portfolio.get('cash', 0):,.2f}\n")

    # 1.5 市場體制偵測（SPY vs MA200）
    print("正在檢查市場體制（SPY MA200）...")
    regime = get_spy_regime()
    if regime["spy_price"] is not None:
        regime_label = f"SPY ${regime['spy_price']} vs MA200 ${regime['ma200']} ({regime['pct_vs_ma200']:+.1f}%)"
    else:
        regime_label = "資料無法取得，預設 BULL"
    print(f"  市場體制: {regime['regime']}  {regime_label}")

    # 1.6 板塊相對強弱檢查
    print("正在檢查板塊相對強弱...")
    sector_summary = get_sector_summary(lookback_days=5)
    held_symbols = list(positions.keys())
    sector_exposure = check_holdings_sector_exposure(held_symbols)

    # 2. 組合候選池：SP500 前 100 + 白名單 + 持倉
    sp500 = get_sp500_tickers()
    watchlist = load_watchlist()
    wl_symbols = watchlist.get("symbols", [])
    all_tickers = list(dict.fromkeys(sp500 + wl_symbols + held_symbols))

    print(f"\n正在計算 {len(all_tickers)} 檔標的動能（SP500 前100 + 白名單 {len(wl_symbols)} 檔 + 持倉）...")

    # 3. 計算動能排名
    momentum_ranks = rank_by_momentum(all_tickers, period=21)

    # 4. 取得報價（動能前 35 名 + 持倉，涵蓋主要候選+備選）
    top_symbols = [m["symbol"] for m in momentum_ranks[:35]]
    symbols_for_price = list(set(top_symbols + held_symbols))
    print(f"正在取得 {len(symbols_for_price)} 檔報價...")
    current_prices = fetch_current_prices(symbols_for_price)

    # 4.5 取得持倉的 MA200 資料（用於出場判斷）
    print(f"正在取得 {len(held_symbols)} 檔持倉的 MA200...")
    ma200_prices = fetch_ma200_prices(held_symbols)

    # 4.6 初始化/更新最高價追蹤
    initialize_high_prices(portfolio, current_prices)
    high_updated = update_high_prices(portfolio, current_prices)
    if high_updated:
        save_portfolio(portfolio)
        print("已更新持倉最高價記錄")

    # 4.7 計算 1 年超額報酬（ADD 候選 + 持倉，用於長期績效參考）
    # 取前 35 名確保涵蓋主要候選（5個）+ 備選（3個），前幾名可能已持有
    add_candidates = [m["symbol"] for m in momentum_ranks[:35] if m["symbol"] not in positions]
    alpha_symbols = list(set(add_candidates + held_symbols))
    print(f"正在計算 {len(alpha_symbols)} 檔標的的 1 年超額報酬...")
    alpha_1y_map = calculate_alpha_batch(alpha_symbols)
    print(f"正在計算 {len(add_candidates)} 檔 ADD 候選的 3 年超額報酬...")
    alpha_3y_map = calculate_alpha_3y_batch(add_candidates)

    # 4.8 計算持倉趨勢狀態（V轉/倒V/盤整）
    print(f"正在計算 {len(held_symbols)} 檔持倉的趨勢狀態...")
    trend_state_map = calculate_trend_state_batch(held_symbols)

    # 5. 產出 actions（使用動能排名 + 三層出場 + 趨勢狀態 + 市場體制）
    actions = generate_actions(portfolio, current_prices, ma200_prices, momentum_ranks, alpha_1y_map, trend_state_map, alpha_3y_map, market_regime=regime["regime"])

    # 5.5 增持參考（倉位偏小 + 動能強 + 趨勢轉強）
    total_value_for_topup = portfolio.get("cash", 0) + sum(
        current_prices.get(sym, pos["avg_price"]) * pos["shares"]
        for sym, pos in positions.items()
    )
    topup_suggestions = generate_topup_suggestions(
        portfolio, current_prices, momentum_ranks, alpha_1y_map, trend_state_map, total_value_for_topup, alpha_3y_map
    )

    # 5.6 篩選「安全」TOPUP，計算補到等權重所需股數
    CASH_SAFETY_FACTOR = 0.85
    equal_weight = total_value_for_topup / 30
    safe_topups = []
    for s in topup_suggestions:
        if "安全" not in s.get("safety", ""):
            continue
        current_val = s["current_price"] * s["shares"]
        needed = max(0, equal_weight - current_val)
        topup_shares = math.floor(needed / s["current_price"]) if s["current_price"] > 0 else 0
        topup_cost = topup_shares * s["current_price"]
        if topup_shares > 0:
            safe_topups.append({**s, "topup_shares": topup_shares, "topup_cost": topup_cost})

    # 5.7 重算 ROTATE 後的 ADD 股數（扣除安全 TOPUP 預算）
    adds_list = [a for a in actions if a["action"] == "ADD"]
    rotates_list = [a for a in actions if a["action"] == "ROTATE"]
    rotate_proceeds = sum(a["sell_shares"] * a["sell_price"] * CASH_SAFETY_FACTOR for a in rotates_list)
    post_rotate_cash = portfolio.get("cash", 0) + rotate_proceeds
    topup_total_cost = sum(s["topup_cost"] for s in safe_topups)
    remaining_for_add = max(0, post_rotate_cash - topup_total_cost)
    if adds_list:
        post_rotate_per_slot = remaining_for_add / len(adds_list)
        for a in adds_list:
            price = a.get("current_price", 0)
            if price > 0:
                a["suggested_shares_post_rotate"] = math.floor(post_rotate_per_slot / price)

    # 6. 計算投組總值
    total_value = portfolio.get("cash", 0)
    for symbol, pos in positions.items():
        price = current_prices.get(symbol, pos["avg_price"])
        total_value += price * pos["shares"]

    # 7. 載入年度快照並計算年度 P&L（用於儲存）
    current_year = date.today().year
    snapshot = load_snapshot(current_year)
    yearly_pnl = calculate_yearly_pnl(total_value, snapshot) if snapshot else None

    # 7.5 儲存 actions
    actions_output = {
        "date": str(date.today()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": VERSION,
        "portfolio_snapshot": {
            "total_value": round(total_value, 2),
            "cash": portfolio.get("cash", 0),
            "individual_count": individual,
            "yearly_pnl": yearly_pnl,
        },
        "regime_status": regime,
        "sector_status": {
            "status": sector_summary["status"],
            "alerts": [a["message"] for a in sector_summary["alerts"]],
            "tech_ratio": sector_exposure["tech_ratio"],
        },
        "actions": actions,
        "topup_suggestions": topup_suggestions,
        "safe_topups": safe_topups,
    }

    actions_path = f"data/actions_{today_str}.json"
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(actions_output, f, indent=2, ensure_ascii=False)

    # 8. 印出盤前報告
    print(f"\n{'='*60}")
    print(f"  盤前報告 {date.today()}  |  版本 {VERSION}")
    print(f"{'='*60}")
    print(f"  投組總值: ${total_value:>12,.2f}")
    print(f"  現金:     ${portfolio.get('cash', 0):>12,.2f}")
    print(f"  個股:     {individual}/30 檔")
    if yearly_pnl:
        pnl_sign = "+" if yearly_pnl["pnl_amount"] >= 0 else ""
        print(f"  {current_year}年度:  {pnl_sign}${yearly_pnl['pnl_amount']:>10,.2f} ({pnl_sign}{yearly_pnl['pnl_pct']:.1f}%)")
    else:
        print(f"  {current_year}年度:  (尚無快照，執行 --snapshot 建立)")
    print(f"{'='*60}")

    # 市場體制警告
    if regime["is_bull"]:
        bull_str = f"+{regime['pct_vs_ma200']:.1f}%" if regime["pct_vs_ma200"] is not None else ""
        print(f"\n🟢 市場體制: BULL  SPY ${regime['spy_price']} > MA200 ${regime['ma200']} ({bull_str})")
    else:
        bear_str = f"{regime['pct_vs_ma200']:.1f}%" if regime["pct_vs_ma200"] is not None else ""
        print(f"\n{'='*60}")
        print(f"  🔴 市場體制: BEAR  SPY ${regime['spy_price']} < MA200 ${regime['ma200']} ({bear_str})")
        print(f"  ⚠️  ADD 與 ROTATE 建議已暫停 — 等 SPY 站回 MA200 再開放")
        print(f"{'='*60}")

    # 板塊健康狀態
    print(f"\n--- 板塊相對強弱 (過去5日) {sector_summary['status_emoji']} ---")
    if sector_summary.get("benchmark"):
        print(f"  大盤 SPY: {sector_summary['benchmark']['return']*100:+.1f}%")
    for sym, data in sector_summary.get("sectors", {}).items():
        rel = data["relative"]
        emoji = "🔴" if rel < -0.05 else ("🟡" if rel < 0 else "🟢")
        print(f"  {emoji} {data['name']:<6} {data['return']*100:+.1f}% (vs SPY: {rel*100:+.1f}%)")

    if sector_summary["alerts"]:
        print(f"\n  ⚠️  板塊警告：")
        for alert in sector_summary["alerts"]:
            print(f"     - {alert['message']}")

    if sector_exposure["warning"]:
        print(f"\n  🚨 注意：你的持股 {sector_exposure['tech_ratio']*100:.0f}% 是科技相關，而科技板塊正在走弱！")

    print()

    # 分類印出
    exits = [a for a in actions if a["action"] == "EXIT"]
    holds = [a for a in actions if a["action"] == "HOLD"]
    adds = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup")]
    backup_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_backup")]

    if exits:
        print("--- EXIT (建議出場) ---")
        for a in exits:
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            print(f"  [{a['source']}] {a['symbol']:<6} {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {pnl}")
            print(f"         原因: {a['reason']}")
        print()

    if holds:
        print("--- HOLD (繼續持有) ---")
        for a in holds:
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            momentum = f"動能: {a['momentum']:+.1f}%" if a.get("momentum") is not None else ""
            alpha = a.get('alpha_1y')
            if alpha is not None:
                alpha_emoji = "🟢" if alpha > 0 else ("🟡" if alpha > -20 else "🔴")
                alpha_str = f"  1Y: {alpha:+.0f}% {alpha_emoji}"
            else:
                alpha_str = ""
            # 趨勢狀態
            ts = a.get('trend_state')
            if ts:
                ts_emoji = {"轉強": "↗️", "轉弱": "↘️", "盤整": "→"}.get(ts["state"], "")
                ts_str = f"  {ts_emoji}{ts['state']}"
                if ts["state"] == "轉弱":
                    ts_str += f"(距高{ts['from_high_pct']:+.0f}%)"
                elif ts["state"] == "轉強":
                    ts_str += f"(反彈{ts['bounce_pct']:+.0f}%)"
            else:
                ts_str = ""
            # 追蹤停損接近度（高點 -25% 觸發）
            trail_str = ""
            high_price = a.get('high_since_entry')
            price = a.get('current_price') or 0
            if high_price and price and high_price > 0:
                from_high = (price - high_price) / high_price * 100
                if from_high <= -20:
                    trail_str = f"  🔴追蹤{from_high:.0f}%"
                elif from_high <= -10:
                    trail_str = f"  🟡追蹤{from_high:.0f}%"
            tag = "[core]" if a["source"] == "core_hold" else "      "
            print(f"  {tag} {a['symbol']:<6} {a['shares']} 股 @ ${price:.2f}  P&L: {pnl}  {momentum}{alpha_str}{ts_str}{trail_str}")
        print()

    if adds or safe_topups or backup_adds:
        print("--- ADD / TOPUP 建議 ---")
        for a in adds:
            momentum_str = f"動能: +{a.get('momentum', 0):.1f}%"
            alpha_1y = a.get('alpha_1y')
            alpha_3y = a.get('alpha_3y')
            if alpha_1y is not None:
                alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                alpha_str = f"  1Y: {alpha_1y:+.0f}% {alpha_emoji}"
            else:
                alpha_str = ""
            if alpha_3y is not None:
                alpha_3y_emoji = "🟢" if alpha_3y > 0 else ("🟡" if alpha_3y > -20 else "🔴")
                alpha_str += f"  3Y: {alpha_3y:+.0f}% {alpha_3y_emoji}"
            shares_str = str(a['suggested_shares'])
            post_rotate_shares = a.get('suggested_shares_post_rotate')
            if post_rotate_shares is not None and post_rotate_shares != a['suggested_shares']:
                shares_str += f" (ROTATE後 {post_rotate_shares} 股)"
            print(f"  [#{a.get('momentum_rank', '?')}] {a['symbol']:<6} 建議 {shares_str} @ ${a.get('current_price', 0):.2f}  {momentum_str}{alpha_str}")
            print(f"         原因: {a['reason']}")
        if backup_adds:
            print("  --- 備選（可替換 1Y/3Y alpha 差的主要候選）---")
            for a in backup_adds:
                momentum_str = f"動能: +{a.get('momentum', 0):.1f}%"
                alpha_1y = a.get('alpha_1y')
                alpha_3y = a.get('alpha_3y')
                if alpha_1y is not None:
                    alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                    alpha_str = f"  1Y: {alpha_1y:+.0f}% {alpha_emoji}"
                else:
                    alpha_str = ""
                if alpha_3y is not None:
                    alpha_3y_emoji = "🟢" if alpha_3y > 0 else ("🟡" if alpha_3y > -20 else "🔴")
                    alpha_str += f"  3Y: {alpha_3y:+.0f}% {alpha_3y_emoji}"
                print(f"  [備#{a.get('momentum_rank', '?')}] {a['symbol']:<6} @ ${a.get('current_price', 0):.2f}  {momentum_str}{alpha_str}")
                print(f"           原因: {a['reason']}")
        for s in safe_topups:
            weight_after = (s["current_price"] * (s["shares"] + s["topup_shares"])) / total_value_for_topup * 100
            momentum_str = f"動能: +{s['momentum']:.1f}%(#{s['momentum_rank']})"
            alpha = s.get("alpha_1y")
            alpha_str = f"  1Y: {alpha:+.0f}%" if alpha is not None else ""
            print(f"  [增持] {s['symbol']:<6} +{s['topup_shares']} 股 @ ${s['current_price']:.2f}  {momentum_str}  倉位 {s['current_weight_pct']:.1f}%→{weight_after:.1f}%  {s['safety']}{alpha_str}")
        print()

    # ROTATE 建議（汰弱留強）
    rotates = [a for a in actions if a["action"] == "ROTATE"]
    if rotates:
        print("--- ROTATE (汰弱留強) ---")
        for a in rotates:
            sell_pnl = f"{a['sell_pnl_pct']:+.1f}%" if a.get("sell_pnl_pct") is not None else "N/A"
            buy_alpha_1y = a.get('buy_alpha_1y')
            buy_alpha_3y = a.get('buy_alpha_3y')
            if buy_alpha_1y is not None:
                alpha_emoji = "🟢" if buy_alpha_1y > 0 else ("🟡" if buy_alpha_1y > -20 else "🔴")
                alpha_str = f"1Y: {buy_alpha_1y:+.0f}% {alpha_emoji}"
            else:
                alpha_str = ""
            if buy_alpha_3y is not None:
                alpha_3y_emoji = "🟢" if buy_alpha_3y > 0 else ("🟡" if buy_alpha_3y > -20 else "🔴")
                alpha_str += f"  3Y: {buy_alpha_3y:+.0f}% {alpha_3y_emoji}"
            print(f"  賣 {a['sell_symbol']:<6} {a['sell_shares']} 股 (動能: {a['sell_momentum']:+.1f}%, P&L: {sell_pnl})")
            print(f"  → 買 {a['buy_symbol']:<6} {a['buy_shares']} 股 (動能: +{a['buy_momentum']:.1f}%, {alpha_str})")
            print(f"       動能差: +{a['momentum_diff']:.0f}%  {a['reason']}")
            print()

    # 增持參考（非安全，僅供參考）
    non_safe_topups = [s for s in topup_suggestions if "安全" not in s.get("safety", "")]
    if non_safe_topups:
        print("--- TOPUP 增持參考（風險較高，僅供參考）---")
        for s in non_safe_topups:
            ts = s["trend_state"]
            ts_str = f"↗️轉強(反彈{ts['bounce_pct']:+.0f}%)" if ts else ""
            alpha = s.get("alpha_1y")
            alpha_str = f"  1Y: {alpha:+.0f}%" if alpha is not None else ""
            print(f"  {s['symbol']:<6} 倉位{s['current_weight_pct']:.1f}%  動能+{s['momentum']:.1f}%(#{s['momentum_rank']})  {ts_str}")
            print(f"         現價${s['current_price']:.2f}  成本${s['avg_price']:.2f}  追高{s['run_up_pct']:+.1f}%  {s['safety']}  {s['safety_note']}{alpha_str}")
        print()

    # === 台股觀察（全市場掃描，需加 --tw 開啟）===
    if scan_tw:
        print("正在載入高流動性台股清單...")
        tw_liquid = scan_tw_market(min_volume=1000)  # 使用快取，日均量 > 1000 張
        tw_symbols = [s["symbol"] for s in tw_liquid]
        tw_name_map = {s["symbol"]: s["name"] for s in tw_liquid}

        print(f"正在計算 {len(tw_symbols)} 檔台股動能...")
        tw_momentum = rank_by_momentum(tw_symbols, period=21)

        # 計算台股 1Y Alpha（vs 0050）
        tw_alpha_symbols = [m["symbol"] for m in tw_momentum[:10]]
        from src.momentum import calculate_alpha_1y
        tw_alpha_map = {}
        for sym in tw_alpha_symbols:
            alpha = calculate_alpha_1y(sym, benchmark="0050.TW")
            if alpha is not None:
                tw_alpha_map[sym] = alpha

        # 顯示台股建議（前10名 ADD、後5名觀察）
        tw_adds = [m for m in tw_momentum if m.get("momentum", 0) > 5][:10]
        tw_weak = [m for m in tw_momentum if m.get("momentum", 0) < 0][-5:]

        print()
        print(f"--- 🇹🇼 台股觀察（{len(tw_symbols)} 檔高流動性股）---")
        if tw_adds:
            print("  動能領先（建議觀察）:")
            for m in tw_adds:
                name = tw_name_map.get(m["symbol"], "")
                alpha = tw_alpha_map.get(m["symbol"])
                alpha_str = ""
                if alpha is not None:
                    alpha_emoji = "🟢" if alpha > 0 else ("🟡" if alpha > -10 else "🔴")
                    alpha_str = f"  1Y vs 0050: {alpha:+.0f}% {alpha_emoji}"
                print(f"    #{m['rank']:<3} {m['symbol']:<10} {name:<8} 動能: +{m['momentum']:.1f}%{alpha_str}")

        if tw_weak:
            print("  動能落後（注意風險）:")
            for m in tw_weak:
                name = tw_name_map.get(m["symbol"], "")
                print(f"    #{m['rank']:<3} {m['symbol']:<10} {name:<8} 動能: {m['momentum']:.1f}%")
        print()

        # 將台股資訊加入 actions_output
        actions_output["tw_stocks"] = {
            "scan_count": len(tw_symbols),
            "leaders": [{"symbol": m["symbol"], "name": tw_name_map.get(m["symbol"], ""),
                         "momentum": m["momentum"], "rank": m["rank"],
                         "alpha_1y": tw_alpha_map.get(m["symbol"])} for m in tw_adds],
            "laggards": [{"symbol": m["symbol"], "name": tw_name_map.get(m["symbol"], ""),
                          "momentum": m["momentum"], "rank": m["rank"]} for m in tw_weak],
        }

        # 重新儲存（含台股）
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions_output, f, indent=2, ensure_ascii=False)

    # 9. 發送 Email 通知
    notifier = GmailNotifier()
    if notifier.is_configured():
        print("正在發送 Email 通知...")
        if notifier.send_premarket_report(actions_output):
            print(f"Email 已發送至 {notifier.recipient}")
        else:
            print("Email 發送失敗，請檢查 .env 設定")

    print(f"\nActions 已儲存至: {actions_path}")
    print(f"確認執行請執行: python confirm_main.py {date.today()}")


def run_watch(symbols):
    """新增白名單標的"""
    wl = add_to_watchlist(symbols)
    print(f"白名單已更新：{wl['symbols']}")


def run_momentum(top_n: int = 20):
    """顯示動能排名"""
    portfolio = load_portfolio()
    positions = portfolio.get("positions", {})
    held_symbols = list(positions.keys())

    # 候選池
    sp500 = get_sp500_tickers()[:100]
    watchlist = load_watchlist()
    wl_symbols = watchlist.get("symbols", [])
    all_tickers = list(dict.fromkeys(sp500 + wl_symbols + held_symbols))

    print_momentum_report(all_tickers, period=21, top_n=top_n)

    # 標記持倉
    print("  持倉標記: ", end="")
    ranks = rank_by_momentum(all_tickers, period=21)
    held_in_top = [r for r in ranks[:top_n] if r["symbol"] in positions]
    if held_in_top:
        for r in held_in_top:
            print(f"{r['symbol']}(#{r['rank']}) ", end="")
    else:
        print("無持倉在前 {} 名".format(top_n), end="")
    print()


def run_snapshot(year: int = None):
    """建立年度快照"""
    if year is None:
        year = date.today().year

    portfolio = load_portfolio()
    if not portfolio.get("positions"):
        print("尚未建立投資組合。請先執行: python premarket_main.py --init")
        return

    # 檢查是否已存在
    existing = load_snapshot(year)
    if existing:
        print(f"警告: {year} 年快照已存在 (建立於 {existing.get('created_at')})")
        print(f"  年初總值: ${existing['total_value']:,.2f}")
        confirm = input("是否覆蓋？(y/n): ").strip().lower()
        if confirm != "y":
            print("已取消")
            return

    snapshot = create_year_start_snapshot(portfolio, year)
    path = save_snapshot(snapshot, year)

    print(f"\n{'='*50}")
    print(f"  {year} 年度快照已建立")
    print(f"{'='*50}")
    print(f"  基準日期: {snapshot['date']}")
    print(f"  年初總值: ${snapshot['total_value']:,.2f}")
    print(f"  現金:     ${snapshot['cash']:,.2f}")
    print(f"  持倉數:   {len(snapshot['positions'])} 檔")
    print(f"{'='*50}")
    print(f"\n已儲存至: {path}")


def main():
    parser = argparse.ArgumentParser(description="盤前建議系統（動能策略版）")
    parser.add_argument("--init", action="store_true", help="互動式建立初始投資組合")
    parser.add_argument("--watch", nargs="+", metavar="SYMBOL", help="新增白名單標的")
    parser.add_argument("--snapshot", nargs="?", const=date.today().year, type=int,
                        metavar="YEAR", help="建立年度快照（預設當年）")
    parser.add_argument("--momentum", nargs="?", const=20, type=int,
                        metavar="N", help="查看動能排名（預設前20名）")
    parser.add_argument("--tw", action="store_true", help="掃描台股（預設關閉）")
    args = parser.parse_args()

    if args.init:
        run_init()
    elif args.watch:
        run_watch(args.watch)
    elif args.snapshot:
        run_snapshot(args.snapshot)
    elif args.momentum:
        run_momentum(args.momentum)
    else:
        run_premarket(scan_tw=args.tw)


if __name__ == "__main__":
    main()
