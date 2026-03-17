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
from src.data_loader import get_sp500_tickers, fetch_current_prices, get_tw50_tickers, TW_STOCK_NAMES, get_sp500_sector_map
from src.tw_scanner import get_tw_liquid_tickers, scan_tw_market
from src.risk import check_position_limit, TRANCHE_PARAMS
from src.premarket import generate_actions, VERSION
from src.sector_monitor import get_sector_summary, check_holdings_sector_exposure
from src.market_environment import get_market_environment
from src.ml_scorer import MLScorer
from src.snapshot import load_snapshot, calculate_yearly_pnl, create_year_start_snapshot, save_snapshot
from src.momentum import rank_by_momentum, print_momentum_report, calculate_alpha_batch, calculate_alpha_3y_batch, calculate_trend_state_batch
from src.notifier import GmailNotifier
from src.wave_scanner import scan_waves
from src.breadth_monitor import get_breadth_status


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


def _check_triple_warning(breadth_status, market_env, actions):
    """
    三重警告：廣度偏弱 + 市場環境差 + ADD 候選 ML% 全面低迷。
    同時找出持倉中「動能轉負但尚未觸停損」的防禦性減倉候選。

    Returns dict:
        triggered           bool
        conditions          list[str]
        defensive_candidates list[dict]  持倉弱勢標的
    """
    conditions = []

    # 條件 1：廣度偏弱（< 40%）
    breadth = breadth_status.get("stock_breadth")
    if breadth is not None and breadth < 0.40:
        pct = breadth * 100
        conditions.append(f"市場廣度 {pct:.0f}% < 40%（{breadth_status.get('level', '弱')}）")

    # 條件 2：市場環境 🔴
    regime_emoji = market_env.get("regime_emoji", "")
    regime_label = market_env.get("regime_label", "")
    if regime_emoji == "🔴":
        conditions.append(f"市場環境 {regime_label}（VIX + 油價雙升）")

    # 條件 3：ADD 候選 ML% 平均 < 50%
    add_probs = [
        a["ml_prob"] for a in actions
        if a["action"] == "ADD" and not a.get("is_backup") and a.get("ml_prob") is not None
    ]
    if add_probs:
        avg_ml = sum(add_probs) / len(add_probs) * 100
        if avg_ml < 50:
            conditions.append(f"ADD 候選 ML% 平均 {avg_ml:.0f}%（< 50%，模型整體不看好）")

    triggered = len(conditions) >= 2  # 至少兩項才觸發，避免過敏

    # 防禦性減倉候選：已持有、動能 < 0 或趨勢轉弱、尚未在 EXIT 清單
    exit_syms = {a["symbol"] for a in actions if a["action"] == "EXIT"}
    defensive = []
    for a in actions:
        if a["action"] != "HOLD":
            continue
        if a.get("source") == "core_hold":
            continue
        if a["symbol"] in exit_syms:
            continue

        mom   = a.get("momentum")
        pnl   = a.get("pnl_pct")
        ts    = (a.get("trend_state") or {}).get("state", "")

        # 計算距高點 %
        high_price = a.get("high_since_entry")
        price      = a.get("current_price") or 0
        from_high  = (price - high_price) / high_price * 100 if (high_price and price and high_price > 0) else None

        is_weak = (
            (mom is not None and mom < 0) or
            (ts == "轉弱" and (mom or 0) < 5) or
            (from_high is not None and from_high <= -15 and (pnl or 0) < 0)
        )

        if is_weak:
            defensive.append({
                "symbol":       a["symbol"],
                "momentum":     mom,
                "pnl_pct":      pnl,
                "trend_state":  a.get("trend_state"),
                "from_high_pct": from_high,
            })

    # 依動能升序（最弱的排前面）
    defensive.sort(key=lambda x: (x.get("momentum") or 0))

    return {
        "triggered":            triggered,
        "conditions":           conditions,
        "defensive_candidates": defensive,
    }


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

    # 1.6 板塊相對強弱 + 市場環境（VIX + 石油）
    print("正在檢查板塊相對強弱 + 市場環境...")
    sector_summary = get_sector_summary(lookback_days=5)
    held_symbols = list(positions.keys())
    sector_exposure = check_holdings_sector_exposure(held_symbols)
    market_env = get_market_environment()

    # 1.7 ML 評分器（快取存在直接讀取，否則自動訓練）
    ml_scorer = MLScorer()
    ml_scorer.ensure_trained()

    # 2. 組合候選池：SP500 前 100 + 白名單 + 持倉
    sp500 = get_sp500_tickers()
    watchlist = load_watchlist()
    wl_symbols = watchlist.get("symbols", [])
    all_tickers = list(dict.fromkeys(sp500 + wl_symbols + held_symbols))

    print(f"\n正在計算 {len(all_tickers)} 檔標的動能（SP500 前100 + 白名單 {len(wl_symbols)} 檔 + 持倉）...")

    # 3. 計算動能排名
    momentum_ranks = rank_by_momentum(all_tickers, period=21)

    # 4. 取得報價（動能前 45 名 + 持倉，涵蓋主要候選+備選，緩衝持倉佔位）
    top_symbols = [m["symbol"] for m in momentum_ranks[:45]]
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
    # 取前 45 名確保涵蓋主要候選（5個）+ 備選（3個），前幾名可能已持有
    add_candidates = [m["symbol"] for m in momentum_ranks[:45] if m["symbol"] not in positions]
    alpha_symbols = list(set(add_candidates + held_symbols))
    print(f"正在計算 {len(alpha_symbols)} 檔標的的 1 年超額報酬...")
    alpha_1y_map = calculate_alpha_batch(alpha_symbols)
    alpha_3y_symbols = list(set(add_candidates + held_symbols))
    print(f"正在計算 {len(alpha_3y_symbols)} 檔標的的 3 年超額報酬（ADD 候選 + 持倉）...")
    alpha_3y_map = calculate_alpha_3y_batch(alpha_3y_symbols)

    # 4.8 計算持倉趨勢狀態（V轉/倒V/盤整）
    print(f"正在計算 {len(held_symbols)} 檔持倉的趨勢狀態...")
    trend_state_map = calculate_trend_state_batch(held_symbols)

    # 4.9 波浪偵測掃描（量縮量增突破，需下載全市場量能，首次較慢）
    print("正在執行波浪偵測掃描（量縮量增突破）...")
    wave_alerts = scan_waves(verbose=True)

    # 4.95 市場廣度（使用波浪掃描快取，不需額外下載）
    breadth_status = get_breadth_status()

    # 5. 產出 actions（使用動能排名 + 三層出場 + 趨勢狀態 + 市場體制）
    actions = generate_actions(portfolio, current_prices, ma200_prices, momentum_ranks, alpha_1y_map, trend_state_map, alpha_3y_map, market_regime=regime["regime"])

    # 5.5 重算 ROTATE 後的 ADD 股數（新倉 + 金字塔共用，備選不占位）
    CASH_SAFETY_FACTOR = 0.85
    new_adds_list = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup", False) and not a.get("is_pyramid", False)]
    pyramid_adds_list = [a for a in actions if a["action"] == "ADD" and a.get("is_pyramid", False)]
    rotates_list = [a for a in actions if a["action"] == "ROTATE"]
    rotate_proceeds = sum(a["sell_shares"] * a["sell_price"] * CASH_SAFETY_FACTOR for a in rotates_list)
    post_rotate_cash = portfolio.get("cash", 0) + rotate_proceeds
    buying_slots = new_adds_list + pyramid_adds_list
    if buying_slots:
        post_rotate_per_slot = post_rotate_cash / len(buying_slots)
        for a in buying_slots:
            price = a.get("current_price", 0)
            if price > 0:
                a["suggested_shares_post_rotate"] = math.floor(post_rotate_per_slot / price)

    # 5.8 補充板塊資訊到所有 actions（共用 Wikipedia 快取，無額外 HTTP 請求）
    sector_map = get_sp500_sector_map()
    for action in actions:
        if action["action"] in ("HOLD", "EXIT", "ADD"):
            action["sector"] = sector_map.get(action["symbol"])
        elif action["action"] == "ROTATE":
            action["buy_sector"] = sector_map.get(action["buy_symbol"])
            action["sell_sector"] = sector_map.get(action["sell_symbol"])

    # 6. 計算投組總值
    total_value = portfolio.get("cash", 0)
    for symbol, pos in positions.items():
        price = current_prices.get(symbol, pos["avg_price"])
        total_value += price * pos["shares"]

    # 7. 載入年度快照並計算年度 P&L（用於儲存）
    current_year = date.today().year
    snapshot = load_snapshot(current_year)
    yearly_pnl = calculate_yearly_pnl(total_value, snapshot) if snapshot else None

    # 5.9 ML 評分（ADD 候選 + 金字塔）
    add_syms_for_ml = list({
        a["symbol"] for a in actions
        if a["action"] == "ADD"
    })
    # 取廣度值供 ML 特徵使用
    raw_breadth = breadth_status.get("raw_breadth", 0.6)
    ml_scores = ml_scorer.score(add_syms_for_ml, breadth=raw_breadth) if ml_scorer._ready else {}
    for a in actions:
        if a["action"] == "ADD" and a["symbol"] in ml_scores:
            a["ml_prob"]     = ml_scores[a["symbol"]]["prob"]
            a["ml_shap_top"] = ml_scores[a["symbol"]]["shap_top"]

    # 7.6 三重警告檢查
    _triple = _check_triple_warning(breadth_status, market_env, actions)

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
        "wave_alerts": wave_alerts,
        "breadth_status": {
            "stock_breadth":      breadth_status["stock_breadth"],
            "suggested_max_adds": breadth_status["suggested_max_adds"],
            "level":              breadth_status["level"],
        },
        "market_env": {
            "vix_level":    market_env.get("vix_level"),
            "vix_ma63":     market_env.get("vix_ma63"),
            "oil_ret_21d":  market_env.get("oil_ret_21d"),
            "oil_ma_ratio": market_env.get("oil_ma_ratio"),
            "regime_label": market_env.get("regime_label"),
            "regime_emoji": market_env.get("regime_emoji"),
            "regime_note":  market_env.get("regime_note"),
        },
        "triple_warning": _triple,
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

    # 市場廣度
    print(f"\n  {breadth_status['display_line']}")

    # 市場環境（VIX + 石油）
    print()
    print(market_env["display_block"])
    print()

    # 三重警告
    if _triple["triggered"]:
        print("=" * 60)
        print("  🚨 三重警告：市場環境不利於新增部位")
        print("=" * 60)
        for c in _triple["conditions"]:
            print(f"  ⚠️  {c}")
        print()
        print("  建議：優先守住現有部位，暫緩新 ADD。")
        print("  以下持倉動能轉負、尚未觸停損，可考慮防禦性減倉：")
        if _triple["defensive_candidates"]:
            for d in _triple["defensive_candidates"]:
                mom_str  = f"動能 {d['momentum']:+.1f}%" if d.get("momentum") is not None else ""
                pnl_str  = f"P&L {d['pnl_pct']:+.1f}%" if d.get("pnl_pct") is not None else ""
                ts_str   = f"↘️轉弱" if (d.get("trend_state") or {}).get("state") == "轉弱" else ""
                trail_str = f"距高 {d['from_high_pct']:+.0f}%" if d.get("from_high_pct") is not None else ""
                tags = "  ".join(x for x in [mom_str, pnl_str, ts_str, trail_str] if x)
                print(f"    {d['symbol']:<7}  {tags}")
        else:
            print("    （目前持倉無明顯弱勢標的）")
        print("=" * 60)
        print()

    # 波浪偵測警報（量縮量增突破）
    if wave_alerts:
        high_alerts  = [a for a in wave_alerts if a["alert_level"] == "HIGH"]
        watch_alerts = [a for a in wave_alerts if a["alert_level"] == "WATCH"]
        print("=" * 60)
        print("  🚨 波浪偵測警報（量縮量增突破）")
        print("     F+A 雙信號精準度 ~67%，歷史命中 META(2023)、ANET(2023)")
        print("=" * 60)
        for a in high_alerts:
            voo_note = ""
            if individual >= 28 and "VOO" in positions:
                voo_note = "  ← ⚠️ 槽位接近上限，考慮賣出 VOO 換入"
            print(f"  🔴 HIGH  {a['sym']:<6} 排名 #{a['rank_now']}  信號: {a['signals']}{voo_note}")
            if a.get("mom_pct") is not None:
                rank_str = f"{a.get('rank_prev','?')} → {a['rank_now']}" if a.get("rank_prev") else f"#{a['rank_now']}"
                print(f"          動能: {a['mom_pct']:+.1f}%  排名變化: {rank_str}")
        for a in watch_alerts:
            print(f"  🟡 WATCH {a['sym']:<6} 排名 #{a['rank_now']}  信號: {a['signals']}")
        print("=" * 60)
        print()

    # 分類印出
    exits = [a for a in actions if a["action"] == "EXIT"]
    holds = [a for a in actions if a["action"] == "HOLD"]
    new_adds = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup") and not a.get("is_pyramid")]
    pyramid_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_pyramid")]
    backup_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_backup")]
    adds = new_adds  # 向下相容

    if exits:
        print("--- EXIT (建議出場) ---")
        for a in exits:
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
            print(f"  [{a['source']}] {a['symbol']:<6}{tranche_str} {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {pnl}")
            print(f"         原因: {a['reason']}")
        print()

    if holds:
        print("--- HOLD (繼續持有) ---")
        today = date.today()
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
            # 保護期（單批次：inline；多批次：分行）
            tranches = a.get("tranches", [])
            protect_inline = ""
            if len(tranches) == 1:
                t = tranches[0]
                try:
                    days_held = (today - date.fromisoformat(t["entry_date"])).days
                except (ValueError, KeyError):
                    days_held = 999
                params = TRANCHE_PARAMS.get(t.get("stop_type", "standard"), TRANCHE_PARAMS["standard"])
                days_left = params["protect"] - days_held
                protect_inline = f"  [保護中{days_left}天]" if days_left > 0 else "  [可出場]"
            tag = "[core]" if a["source"] == "core_hold" else "      "
            sector_tag = f"[{a['sector']}]" if a.get('sector') else ""
            print(f"  {tag} {a['symbol']}{sector_tag}  {a['shares']} 股 @ ${price:.2f}  P&L: {pnl}  {momentum}{alpha_str}{ts_str}{trail_str}{protect_inline}")
            # 多批次：逐批顯示保護期
            if len(tranches) > 1:
                stop_labels = {"standard": "標準", "tight_2": "↑緊", "tight_3": "↑↑緊"}
                for i, t in enumerate(tranches):
                    prefix = "  └" if i == len(tranches) - 1 else "  ├"
                    try:
                        days_held = (today - date.fromisoformat(t["entry_date"])).days
                    except (ValueError, KeyError):
                        days_held = 999
                    params = TRANCHE_PARAMS.get(t.get("stop_type", "standard"), TRANCHE_PARAMS["standard"])
                    days_left = params["protect"] - days_held
                    protect_str = f"保護中{days_left}天" if days_left > 0 else "可出場"
                    stop_label = stop_labels.get(t.get("stop_type", "standard"), "")
                    t_pnl = (price - t["entry_price"]) / t["entry_price"] * 100 if price and t["entry_price"] else None
                    t_pnl_str = f"{t_pnl:+.1f}%" if t_pnl is not None else "N/A"
                    print(f"         {prefix} 批{t['n']}({stop_label}) {t['shares']:>3}股 @${t['entry_price']:.0f}  P&L:{t_pnl_str:>7}  {protect_str}")
        print()

    if new_adds or pyramid_adds or backup_adds:
        max_adds = breadth_status["suggested_max_adds"]
        breadth_note = ""
        if breadth_status["warning"]:
            actual_new = len(new_adds)
            if actual_new > max_adds:
                breadth_note = f"  {breadth_status['emoji']} 廣度{breadth_status['level']}，建議只執行前 {max_adds} 支"
            else:
                breadth_note = f"  {breadth_status['emoji']} 廣度{breadth_status['level']}"
        print(f"--- ADD 建議{breadth_note} ---")
        for a in new_adds:
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
            sector_tag = f"[{a['sector']}]" if a.get('sector') else ""
            # ML 評分
            ml_prob = a.get('ml_prob')
            ml_str  = f"  ML: {ml_prob*100:.0f}%" if ml_prob is not None else ""
            print(f"  [#{a.get('momentum_rank', '?')}] {a['symbol']}{sector_tag}  建議 {shares_str} @ ${a.get('current_price', 0):.2f}  {momentum_str}{alpha_str}{ml_str}")
            # SHAP 解釋（若有）
            shap_top = a.get('ml_shap_top', [])
            if shap_top:
                shap_parts = [f"{arrow}{label}" for label, sv, arrow in shap_top]
                print(f"         ML因素: {' | '.join(shap_parts)}")
            print(f"         原因: {a['reason']}")
        if pyramid_adds:
            print("  --- 金字塔加碼（持倉加碼）---")
            for a in pyramid_adds:
                momentum_str = f"動能: +{a.get('momentum', 0):.1f}%"
                alpha_1y = a.get('alpha_1y')
                alpha_str = f"  1Y: {alpha_1y:+.0f}%" if alpha_1y is not None else ""
                direction_arrow = "↑" if a.get("direction") == "up" else "↓"
                stop_type = a.get("stop_type", "standard")
                sector_tag = f"[{a.get('sector', '')}]" if a.get('sector') else ""
                post_rotate_shares = a.get('suggested_shares_post_rotate')
                shares_str = str(a['suggested_shares'])
                if post_rotate_shares is not None and post_rotate_shares != a['suggested_shares']:
                    shares_str += f" (ROTATE後 {post_rotate_shares} 股)"
                ml_prob = a.get('ml_prob')
                ml_str  = f"  ML: {ml_prob*100:.0f}%" if ml_prob is not None else ""
                print(f"  [{direction_arrow}第{a['tranche_n']}批] {a['symbol']}{sector_tag}  建議 {shares_str} @ ${a.get('current_price', 0):.2f}  {momentum_str}{alpha_str}{ml_str}")
                shap_top = a.get('ml_shap_top', [])
                if shap_top:
                    shap_parts = [f"{arrow}{label}" for label, sv, arrow in shap_top]
                    print(f"         ML因素: {' | '.join(shap_parts)}")
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
                sector_tag = f"[{a['sector']}]" if a.get('sector') else ""
                print(f"  [備#{a.get('momentum_rank', '?')}] {a['symbol']}{sector_tag}  @ ${a.get('current_price', 0):.2f}  {momentum_str}{alpha_str}")
                print(f"           原因: {a['reason']}")
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
            sell_sector_tag = f"[{a['sell_sector']}]" if a.get('sell_sector') else ""
            buy_sector_tag = f"[{a['buy_sector']}]" if a.get('buy_sector') else ""
            print(f"  賣 {a['sell_symbol']}{sell_sector_tag}  {a['sell_shares']} 股 (動能: {a['sell_momentum']:+.1f}%, P&L: {sell_pnl})")
            print(f"  → 買 {a['buy_symbol']}{buy_sector_tag}  {a['buy_shares']} 股 (動能: +{a['buy_momentum']:.1f}%, {alpha_str})")
            print(f"       動能差: +{a['momentum_diff']:.0f}%  {a['reason']}")
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

    # 候選池（全 S&P 500，與完整盤前一致）
    sp500 = get_sp500_tickers()
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
