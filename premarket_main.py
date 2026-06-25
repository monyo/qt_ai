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
from src.data_loader import get_sp500_tickers, fetch_current_prices, fetch_volumes, get_tw50_tickers, TW_STOCK_NAMES, get_sp500_sector_map
from src.tw_scanner import get_tw_liquid_tickers, scan_tw_market
from src.risk import (check_position_limit, TRANCHE_PARAMS, update_dynamic_trailing,
                      update_winner_cycle_highs, check_winner_cycle_exits,
                      load_winner_cycle_watch, save_winner_cycle_watch,
                      update_winner_cycle_watch_lows, check_winner_cycle_reentries,
                      WINNER_CYCLE_PULLBACK)
from src.premarket import generate_actions, VERSION
from src.sector_monitor import get_sector_summary, check_holdings_sector_exposure
from src.market_environment import get_market_environment
from src.ml_scorer import MLScorer
from src.snapshot import load_snapshot, calculate_yearly_pnl, create_year_start_snapshot, save_snapshot
from src.momentum import rank_by_momentum, print_momentum_report, calculate_alpha_batch, calculate_alpha_3y_batch, calculate_trend_state_batch
from src.notifier import GmailNotifier
from src.wave_scanner import scan_waves
from src.breadth_monitor import get_breadth_status
from src.deviation_tracker import print_deviation_report


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


def _get_stop_update_reminders(portfolio, current_prices, vol_map=None):
    """找出需要更新 Firstrade 停損單的持倉批次。

    每個批次的有效停損 = max(固定停損, 追蹤停損)
    - 固定停損 = 批次成本價 × (1 - fixed_pct)  [波動率分層調整]
    - 追蹤停損 = 批次最高點 × (1 - trailing_pct)

    觸發條件：標的中至少有一個批次的追蹤停損 > 固定停損。
    多批次標的：觸發後輸出全部批次，從最緊（批N）到最寬（批1），
    讓使用者知道 Firstrade 要依序掛哪個價格（越緊的越先觸發）。

    Returns:
        list of dict: symbol, tranche_n, shares, entry_price, high_price,
                      fixed_stop, trailing_stop, effective_stop,
                      trailing_pct, tightened, use_fixed, vol_tier
    """
    from src.risk import TRANCHE_PARAMS, vol_adjusted_stops
    reminders = []
    positions = portfolio.get("positions", {})

    for symbol, pos in positions.items():
        if pos.get("core"):
            continue

        sym_vol = (vol_map or {}).get(symbol)

        tranches = pos.get("tranches", [])
        if not tranches:
            tranches = [{
                "n": 1,
                "shares": pos.get("shares", 0),
                "entry_price": pos["avg_price"],
                "high": pos.get("high_since_entry", pos["avg_price"]),
                "stop_type": "standard",
            }]

        multi = len(tranches) > 1

        # 計算每個批次的停損資訊
        tranche_infos = []
        any_trailing_above_fixed = False

        for t in tranches:
            stop_type    = t.get("stop_type", "standard")
            params       = TRANCHE_PARAMS.get(stop_type, TRANCHE_PARAMS["standard"])
            entry_price  = t.get("entry_price", pos["avg_price"])
            high_price   = t.get("high") or pos.get("high_since_entry") or entry_price
            tightened    = "trailing_pct" in t

            # 波動率分層調整固定停損距離（standard 批次）
            base_fixed_dist, base_trail_dist = vol_adjusted_stops(stop_type, sym_vol)
            eff_trailing  = t.get("trailing_pct", base_trail_dist)

            fixed_stop     = entry_price * (1 - base_fixed_dist)
            trailing_stop  = high_price  * (1 - eff_trailing)
            effective_stop = max(fixed_stop, trailing_stop)
            use_fixed      = fixed_stop >= trailing_stop

            # vol tier 標示
            if sym_vol is None:
                vol_tier = None
            elif sym_vol < 0.35:
                vol_tier = "低"
            elif sym_vol < 0.60:
                vol_tier = "中"
            else:
                vol_tier = "高"

            if trailing_stop > fixed_stop:
                any_trailing_above_fixed = True

            tranche_infos.append({
                "symbol":         symbol,
                "tranche_n":      t["n"] if multi else None,
                "shares":         t.get("shares", pos.get("shares", 0)),
                "entry_price":    entry_price,
                "high_price":     high_price,
                "fixed_stop":     fixed_stop,
                "trailing_stop":  trailing_stop,
                "effective_stop": effective_stop,
                "trailing_pct":   eff_trailing * 100,
                "tightened":      tightened,
                "use_fixed":      use_fixed,
                "vol_tier":       vol_tier,
                "vol":            sym_vol,
            })

        if not any_trailing_above_fixed:
            continue  # 此標的無任何批次需要更新

        # 多批次：從最緊（批N）到最寬（批1）排序輸出
        if multi:
            tranche_infos = sorted(tranche_infos, key=lambda x: -(x["tranche_n"] or 0))

        reminders.extend(tranche_infos)

    return reminders


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


TW_SLOTS      = 5       # 台股持倉上限（槽數）
TW_FIXED_STOP = 0.15    # 固定停損 -15%
TW_TRAIL_STOP = 0.25    # 追蹤停損 -25%
TW_CACHE      = "data/_tw_bt_prices.pkl"


TW_MKTCAP_PATH = "data/_tw_mktcap.pkl"
TW_MKTCAP_MIN  = 500  # 億台幣，回測驗證：>500億中位報酬 +2.4% vs 無門檻


def _tw_momentum_from_cache(exclude_syms=None):
    """從快取計算台股動能排名，回傳 [{symbol, name, momentum, trend_state, price}]"""
    import json as _json
    import pickle as _pickle
    if not os.path.exists(TW_CACHE):
        return []
    prices = pd.read_pickle(TW_CACHE)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    with open("data/tw_liquid_stocks.json") as f:
        tw_data = _json.load(f)
    name_map = {s["symbol"]: s["name"] for s in tw_data["stocks"]}
    exclude = set(exclude_syms or [])

    # 市值快取（每週更新一次）
    mktcap_map = {}
    if os.path.exists(TW_MKTCAP_PATH):
        with open(TW_MKTCAP_PATH, "rb") as f:
            mktcap_map = _pickle.load(f)

    results = []
    for sym in prices.columns:
        if sym in exclude:
            continue
        # Filter 3：市值門檻（回測驗證：>500億中位報酬 +2.4%）
        if mktcap_map.get(sym, 0) < TW_MKTCAP_MIN:
            continue
        s = prices[sym].dropna()
        if len(s) < 260:
            continue
        p0   = float(s.iloc[-1])
        p21  = float(s.iloc[-22])
        p252 = float(s.iloc[-253]) if len(s) > 253 else None
        p63  = float(s.iloc[-64])  if len(s) > 64  else None
        if not (p21 and p252 and p21 > 0 and p252 > 0):
            continue
        mom = 0.5 * (p0/p21 - 1) * 100 + 0.5 * (p0/p252 - 1) * 100
        # 風險過濾（回測驗證：排除後中位報酬 +0.73%）
        # Filter 1：5日急漲 >40%（炒作特徵，如聯致 70→190）
        ret5 = (p0 / float(s.iloc[-6]) - 1) if len(s) > 6 else 0
        if ret5 > 0.40:
            continue
        # Filter 2：連續漲停 ≥ 2 天（追漲停陷阱）
        daily_rets = s.iloc[-7:].pct_change().dropna().values
        consec_up = 0
        for dr in reversed(daily_rets):
            if dr >= 0.095:
                consec_up += 1
            else:
                break
        if consec_up >= 2:
            continue
        # 趨勢狀態
        h40 = float(s.iloc[-40:].max())
        l40 = float(s.iloc[-40:].min())
        bounce  = (p0/l40 - 1) * 100
        from_h  = (p0/h40 - 1) * 100
        if bounce > 20 and from_h > -5:
            state = "↗️轉強"
        elif from_h < -15:
            state = "↘️轉弱"
        else:
            state = "→盤整"
        results.append({
            "symbol": sym, "name": name_map.get(sym, ""),
            "momentum": round(mom, 1), "price": round(p0, 1),
            "trend_state": state, "from_high_pct": round(from_h, 1),
        })

    results.sort(key=lambda x: x["momentum"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def _run_tw_section(portfolio, actions_output):
    """台股持倉管理：HOLD/EXIT/ADD，寫入 actions_output['tw_actions']"""
    tw_positions = portfolio.get("tw_positions", {})
    tw_cash      = portfolio.get("tw_cash", 0.0)

    # ── 取得持倉現價 ──
    tw_syms = list(tw_positions.keys())
    tw_prices = {}
    if tw_syms:
        try:
            raw = yf.download(tw_syms, period="3d", auto_adjust=True, progress=False)
            close = raw["Close"] if "Close" in raw else raw
            if isinstance(close, pd.Series):
                close = close.to_frame(name=tw_syms[0])
            for sym in tw_syms:
                if sym in close.columns:
                    s = close[sym].dropna()
                    if not s.empty:
                        tw_prices[sym] = float(s.iloc[-1])
        except Exception:
            pass

    # ── 計算台股持倉總值 ──
    tw_pos_value = sum(
        tw_prices.get(sym, pos["avg_price"]) * pos["shares"]
        for sym, pos in tw_positions.items()
    )
    tw_total = tw_cash + tw_pos_value

    print()
    print("─" * 60)
    print(f"  🇹🇼 台股部位")
    print(f"  現金: NT${tw_cash:>10,.0f}  持倉: NT${tw_pos_value:>10,.0f}  合計: NT${tw_total:>10,.0f}")
    print(f"  持倉: {len(tw_positions)}/{TW_SLOTS} 槽")
    print("─" * 60)

    today_str = date.today().isoformat()
    tw_actions = []

    # ── HOLD / EXIT 檢查 ──
    exit_syms = []
    hold_lines = []

    for sym, pos in tw_positions.items():
        px = tw_prices.get(sym, pos["avg_price"])
        shares     = pos["shares"]
        avg_price  = pos["avg_price"]
        high       = pos.get("high_since_entry", avg_price)
        first_entry = pos.get("first_entry", today_str)
        holding_days = (date.today() - date.fromisoformat(first_entry)).days

        # 更新高點
        if px > high:
            high = px
            portfolio["tw_positions"][sym]["high_since_entry"] = round(px, 2)

        pnl_pct = (px - avg_price) / avg_price * 100
        fixed_px = avg_price * (1 - TW_FIXED_STOP)
        trail_px = high * (1 - TW_TRAIL_STOP)
        stop_px  = max(fixed_px, trail_px)
        from_high_pct = (px - high) / high * 100

        if px < stop_px:
            reason = f"固定停損 -15%" if fixed_px >= trail_px else f"追蹤停損 -25%"
            exit_syms.append(sym)
            tw_actions.append({
                "action": "TW_EXIT", "symbol": sym, "shares": shares,
                "current_price": px, "avg_price": avg_price,
                "pnl_pct": round(pnl_pct, 2), "reason": reason,
                "stop_price": round(stop_px, 2), "status": "pending",
            })
            print(f"  🔴 EXIT  {sym}  {shares}股 @ NT${px:.0f}  P&L: {pnl_pct:+.1f}%  {reason}")
        else:
            # 停損單提醒
            stop_flag = ""
            if from_high_pct < -20:
                stop_flag = "  🔴追蹤-{:.0f}%".format(abs(from_high_pct))
            elif from_high_pct < -10:
                stop_flag = "  🟡追蹤{:.0f}%".format(from_high_pct)

            hold_lines.append(
                f"  HOLD  {sym:<10}  {shares}股 @ NT${avg_price:.0f}"
                f"  P&L: {pnl_pct:+.1f}%  停損: NT${stop_px:.0f}{stop_flag}"
                f"  ({holding_days}天)"
            )

    if not tw_positions:
        print("  （尚無台股持倉）")
    else:
        for line in hold_lines:
            print(line)

    # ── ADD 建議（動能前 TW_SLOTS，排除持倉）──
    occupied = len(tw_positions) - len(exit_syms)
    open_slots = TW_SLOTS - occupied
    print()

    mom_ranks = _tw_momentum_from_cache(exclude_syms=list(tw_positions.keys()))
    # 只取轉強或動能 >10% 的標的
    candidates = [r for r in mom_ranks if r["momentum"] > 10][:TW_SLOTS * 3]

    if open_slots > 0 and candidates:
        per_slot_ntd = tw_cash / open_slots if open_slots > 0 else 0
        print(f"  --- ADD 建議（{open_slots} 個空槽，每槽約 NT${per_slot_ntd:,.0f}）---")
        shown = 0
        for r in candidates:
            if shown >= open_slots:
                print(f"  [備選 #{r['rank']}]  {r['symbol']:<10} {r['name']:<8} "
                      f"動能: {r['momentum']:+.1f}%  {r['trend_state']}  NT${r['price']:.1f}")
                continue
            suggested_shares = int(per_slot_ntd / r["price"]) if r["price"] > 0 else 0
            print(f"  [#{r['rank']}]  {r['symbol']:<10} {r['name']:<8} "
                  f"建議 {suggested_shares} 股 @ NT${r['price']:.1f}"
                  f"  動能: {r['momentum']:+.1f}%  {r['trend_state']}")
            tw_actions.append({
                "action": "TW_ADD", "symbol": r["symbol"], "name": r["name"],
                "current_price": r["price"], "suggested_shares": suggested_shares,
                "momentum": r["momentum"], "trend_state": r["trend_state"],
                "rank": r["rank"], "status": "pending",
            })
            shown += 1
    elif open_slots == 0:
        print("  台股持倉已滿（5 槽），無新增建議")
    else:
        print("  ⚠ 無符合條件的台股候補（需先更新 data/_tw_bt_prices.pkl）")

    # ── 停損單提醒 ──
    stop_updates = []
    for sym, pos in tw_positions.items():
        if sym in exit_syms:
            continue
        px     = tw_prices.get(sym, pos["avg_price"])
        high   = pos.get("high_since_entry", pos["avg_price"])
        trail_px = round(high * (1 - TW_TRAIL_STOP), 0)
        fixed_px = round(pos["avg_price"] * (1 - TW_FIXED_STOP), 0)
        eff    = max(trail_px, fixed_px)
        stop_updates.append((sym, eff, high))

    if stop_updates:
        print()
        print("  📌 台股停損單（Firstrade 無法掛，請在台灣券商設定）")
        for sym, stop, high in stop_updates:
            print(f"     {sym}  停損: NT${stop:.0f}  （高點 NT${high:.0f}）")

    # ── 寫入 actions_output ──
    actions_output["tw_actions"] = tw_actions
    actions_output["tw_cash"] = tw_cash
    actions_output["tw_total"] = tw_total

    return tw_actions


def run_premarket(scan_tw=False, send_email=True):
    """產出盤前建議（動能策略 + 三層出場）

    Args:
        scan_tw: 是否掃描台股（預設 False）
        send_email: 是否發送 Email（預設 True，--no-email 時為 False）
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

    # 4.6b 動態收緊追蹤停損（持倉獲利 ≥ +25% 時，追蹤停損從原始值收緊）
    newly_tightened = update_dynamic_trailing(portfolio, current_prices)
    if newly_tightened:
        save_portfolio(portfolio)
        print(f"🔒 追蹤停損已收緊：{', '.join(newly_tightened)}")

    # 4.6c 取得持倉成交量（供停損確認過濾使用）
    print(f"正在取得 {len(held_symbols)} 檔持倉成交量...")
    held_volumes = fetch_volumes(held_symbols)

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

    # 4.96 計算持倉年化波動率（供波動率分層停損使用）
    vol_map = {}
    try:
        import yfinance as yf
        import numpy as np
        if held_symbols:
            _hist = yf.download(held_symbols, period="90d", auto_adjust=True, progress=False)["Close"]
            if hasattr(_hist, "columns"):
                for _sym in held_symbols:
                    if _sym not in _hist.columns:
                        continue
                    _px = _hist[_sym].dropna()
                    if len(_px) >= 20:
                        _lr = np.log(_px / _px.shift(1)).dropna()
                        if len(_lr) >= 15:
                            vol_map[_sym] = float(_lr.std() * np.sqrt(252))
            elif len(held_symbols) == 1:
                _px = _hist.dropna()
                if len(_px) >= 20:
                    _lr = np.log(_px / _px.shift(1)).dropna()
                    if len(_lr) >= 15:
                        vol_map[held_symbols[0]] = float(_lr.std() * np.sqrt(252))
    except Exception:
        pass

    # 4.97 Winner Cycle：特殊池高點更新（用 alpha_1y_map 判斷股票本身是否超強，不依賴進場時機）
    wc_new_pool = update_winner_cycle_highs(portfolio, current_prices, alpha_1y_map=alpha_1y_map)
    if wc_new_pool:
        print(f"🏆 新進特殊池（1Y alpha>100%）：{', '.join(wc_new_pool)}")
        save_portfolio(portfolio)
    elif any("winner_cycle_high" in pos for pos in portfolio.get("positions", {}).values()):
        save_portfolio(portfolio)  # 更新週期高點
    # 抓近 10 日歷史，用於過濾「出場窗口已過」的 EXIT 訊號
    _wc_pool_syms = [s for s, pos in portfolio.get("positions", {}).items() if pos.get("winner_cycle_high")]
    _wc_hist: dict = {}
    if _wc_pool_syms:
        try:
            import yfinance as _yf
            _h = _yf.download(_wc_pool_syms, period="10d", auto_adjust=True, progress=False)["Close"]
            _h = _h.ffill()
            if len(_wc_pool_syms) == 1:
                _wc_hist[_wc_pool_syms[0]] = _h
            else:
                for _s in _wc_pool_syms:
                    if _s in _h.columns:
                        _wc_hist[_s] = _h[_s]
        except Exception:
            pass
    wc_exits = check_winner_cycle_exits(portfolio, current_prices, hist_prices=_wc_hist)
    wc_watch = load_winner_cycle_watch()
    update_winner_cycle_watch_lows(wc_watch, current_prices)
    save_winner_cycle_watch(wc_watch)
    wc_reentries = check_winner_cycle_reentries(wc_watch, current_prices)

    # 5. 產出 actions（使用動能排名 + 三層出場 + 趨勢狀態 + 市場體制）
    actions = generate_actions(portfolio, current_prices, ma200_prices, momentum_ranks, alpha_1y_map, trend_state_map, alpha_3y_map, market_regime=regime["regime"],
                               vix=market_env.get("vix_level", 20.0), volumes=held_volumes, vol_map=vol_map)

    # 5.1 儲存 pending 停損狀態（stop_pending_since 已寫入 portfolio tranches）
    has_pending = any(a.get("stop_pending") for a in actions if a["action"] == "HOLD")
    if has_pending:
        save_portfolio(portfolio)

    # 5.2 注入 Winner Cycle EXIT 訊號（特殊池標的從週期高點回落 -10%）
    existing_exit_syms = {a["symbol"] for a in actions if a["action"] == "EXIT"}
    _wc_action_id = max((a.get("id", 0) for a in actions), default=0)
    for sym, wc in wc_exits.items():
        if sym in existing_exit_syms:
            continue  # 已有停損信號，不重複
        _wc_action_id += 1
        actions.insert(0, {
            "id": _wc_action_id,
            "action": "EXIT",
            "symbol": sym,
            "shares": wc["shares"],
            "current_price": wc["current_price"],
            "avg_price": wc["avg_price"],
            "pnl_pct": wc["pnl_pct"],
            "reason": f"特殊池輪動出場（從週期高點 ${wc['cycle_high']:.2f} 回落 {wc['from_high_pct']:.1f}%）",
            "source": "winner_cycle",
            "cycle_high": wc["cycle_high"],
            "wc_stop_px": round(wc["cycle_high"] * (1 - WINNER_CYCLE_PULLBACK), 2),
            "status": "pending",
        })

    # 5.2b 標記 winner cycle 出場標的的 ADD（顯示時壓制，不在建議清單出現）
    wc_exit_syms = set(wc_exits.keys())
    for a in actions:
        if a["action"] == "ADD" and a.get("symbol") in wc_exit_syms:
            a["wc_suppressed"] = True

    # 5.3 注入 Winner Cycle 回補訊號（已出場標的從低點反彈 +10%）
    existing_add_syms = {a["symbol"] for a in actions if a["action"] == "ADD"}
    existing_hold_syms = {a["symbol"] for a in actions if a["action"] == "HOLD"}
    for sym, wc in wc_reentries.items():
        if sym in existing_add_syms or sym in existing_hold_syms:
            continue  # 已在持倉或 ADD 清單中
        _wc_action_id += 1
        price = wc["current_price"]
        suggested = math.floor(portfolio.get("cash", 0) / price) if price > 0 else 0
        suggested = min(suggested, wc["shares"])  # 不超過原本持有股數
        actions.append({
            "id": _wc_action_id,
            "action": "ADD",
            "symbol": sym,
            "current_price": price,
            "suggested_shares": suggested,
            "reason": (f"特殊池回補（賣出 ${wc['exit_price']:.2f}，"
                       f"低點 ${wc['post_exit_low']:.2f}，"
                       f"反彈 {wc['recovery_pct']:.1f}%）"),
            "source": "winner_cycle_reentry",
            "status": "pending",
            "is_winner_cycle": True,
        })

    # 5.5 重算 ROTATE 後的 ADD 股數（新倉 + 金字塔共用，備選不占位）
    CASH_SAFETY_FACTOR = 0.85
    new_adds_list = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup", False) and not a.get("is_pyramid", False) and not a.get("is_winner_cycle", False)]
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
            "tw_positions": portfolio.get("tw_positions", {}),
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
    wc_reentry_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_winner_cycle")]
    new_adds = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup") and not a.get("is_pyramid") and not a.get("is_winner_cycle") and not a.get("wc_suppressed")]
    pyramid_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_pyramid") and not a.get("wc_suppressed")]
    backup_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_backup")]
    adds = new_adds  # 向下相容

    if exits:
        print("--- EXIT (建議出場) ---")
        for a in exits:
            pnl = f"{a['pnl_pct']:+.2f}%" if a.get("pnl_pct") is not None else "N/A"
            tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
            wc_tag = " 🏆" if a.get("source") == "winner_cycle" else ""
            print(f"  [{a['source']}]{wc_tag} {a['symbol']:<6}{tranche_str} {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {pnl}")
            print(f"         原因: {a['reason']}")
        print()

    # ── 特殊池狀態（獲利>100% 且在觀察中）──────────────────────────
    pool_positions = {sym: pos for sym, pos in portfolio.get("positions", {}).items()
                      if pos.get("winner_cycle_high")}
    if pool_positions or wc_watch or wc_reentry_adds:
        print("--- 🏆 特殊池（大贏家輪動）---")
        for sym, pos in pool_positions.items():
            price = current_prices.get(sym, 0)
            cycle_high = pos["winner_cycle_high"]
            from_high = (price / cycle_high - 1) * 100 if cycle_high else 0
            pnl_pct = (price / pos.get("avg_price", price) - 1) * 100 if pos.get("avg_price") else 0
            alert = "  ⚠️ 觸發賣出" if sym in wc_exits else ""
            print(f"  持有 {sym:<6}  P&L: {pnl_pct:+.0f}%  週期高點 ${cycle_high:.2f}  距高 {from_high:+.1f}%{alert}")
        for sym, entry in wc_watch.items():
            price = current_prices.get(sym, 0)
            low = entry.get("post_exit_low", 0)
            rec = (price / low - 1) * 100 if low else 0
            print(f"  觀察 {sym:<6}  賣出 ${entry['exit_price']:.2f}  低點 ${low:.2f}  現在 ${price:.2f}  反彈 {rec:+.1f}%  冷卻至 {entry.get('cooldown_end', '?')}")
        for a in wc_reentry_adds:
            print(f"  🔁 買回 {a['symbol']:<6}  建議 {a['suggested_shares']} 股 @ ${a.get('current_price', 0):.2f}  {a['reason']}")
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
                    if ts.get("strong_bounce_pct"):
                        ts_str += f" 🔄強彈+{ts['strong_bounce_pct']:.0f}%"
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
            # 停損待確認警告
            stop_pending = a.get("stop_pending")
            if stop_pending:
                for sp in stop_pending:
                    vol_r = sp.get("vol_ratio")
                    vol_tag = f"  量比={vol_r:.1f}x" if vol_r is not None else ""
                    vix_m = sp.get("vix_mult", 1.0)
                    vix_tag = f"  VIX展寬×{vix_m:.1f}" if vix_m > 1 else ""
                    print(f"         ⚠️  停損待確認（第{sp['tranche_n']}批）{vix_tag}{vol_tag}  {sp['message']}  → 明日仍觸發則出場")
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
                    tight_mark = " 🔒" if "trailing_pct" in t else ""
                    print(f"         {prefix} 批{t['n']}({stop_label}) {t['shares']:>3}股 @${t['entry_price']:.0f}  P&L:{t_pnl_str:>7}  {protect_str}{tight_mark}")
        print()

    # 停損單更新提醒（多批次從最緊到最寬排序，Firstrade 依序掛單）
    stop_reminders = _get_stop_update_reminders(portfolio, current_prices, vol_map=vol_map)
    if stop_reminders:
        print("--- 📌 停損單需更新（Firstrade 依序掛，最緊批次先）---")
        for r in stop_reminders:
            batch_str  = f" 第{r['tranche_n']}批" if r["tranche_n"] is not None else ""
            tight_mark = " 🔒收緊" if r.get("tightened") else ""
            if r.get("use_fixed"):
                reason = f"固定 -{100 - r['fixed_stop'] / r['entry_price'] * 100:.0f}% from ${r['entry_price']:.2f}"
            else:
                reason = f"高點 ${r['high_price']:.2f} 追蹤{r['trailing_pct']:.0f}%{tight_mark}"
            shares_str = f"{r['shares']}股" if r["tranche_n"] is not None else ""
            print(f"  {r['symbol']:<6}{batch_str:<5}  {shares_str:<5}"
                  f"→ 改掛 ${r['effective_stop']:>8.2f}  （{reason}）")
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

        # 現金部署節奏控制：現金 > 20% 投組時，建議分批進場而非單日全投
        cash_now = portfolio.get("cash", 0)
        cash_ratio = cash_now / total_value if total_value > 0 else 0
        if cash_ratio > 0.20 and (new_adds or pyramid_adds):
            batch_amt = cash_now / 3
            print(f"  💰 部署節奏提醒：現金 ${cash_now:,.0f}（{cash_ratio*100:.0f}% 投組）超過 20%")
            print(f"     建議分 3 批投入，每批約 ${batch_amt:,.0f}，間隔 ≥2 個交易日")
            print(f"     今日只執行第一批額度，暫停下一批的條件：")
            print(f"       · 市場環境轉為恐慌（目前：{market_env.get('regime_label', '?')}，VIX {market_env.get('vix_level', 0):.1f}）")
            print(f"       · 市場廣度降級（目前：{breadth_status.get('level', '?')}）")
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

    # === 台股持倉管理（每次自動執行）===
    tw_actions = _run_tw_section(portfolio, actions_output)
    # 補存（含台股資料）
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(actions_output, f, indent=2, ensure_ascii=False)

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

    # 9. 今日待辦清單（操作摘要，執行優先順序）
    exit_actions   = [a for a in actions if a["action"] == "EXIT"]
    wc_exit_actions = [a for a in exit_actions if a.get("source") == "winner_cycle"]
    add_actions    = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup") and not a.get("wc_suppressed")]
    rotate_actions = [a for a in actions if a["action"] == "ROTATE"]
    stop_reminders = _get_stop_update_reminders(portfolio, current_prices, vol_map=vol_map)

    has_todo = exit_actions or stop_reminders or add_actions or rotate_actions or wc_reentry_adds
    if has_todo:
        width = 52
        print("╔" + "═" * width + "╗")
        print(f"║  {'📋 今日待辦清單（開盤前確認）':<{width-2}}║")
        print("╠" + "═" * width + "╣")

        if exit_actions:
            normal_exits = [a for a in exit_actions if a.get("source") != "winner_cycle"]
            if normal_exits:
                print(f"║  {'🔴 EXIT — 立即執行（停損已觸發）':<{width-2}}║")
                for a in normal_exits:
                    tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
                    line = f"     {a['symbol']}{tranche_str}  {a['shares']} 股 @ ${a.get('current_price', 0):.2f}"
                    print(f"║  {line:<{width-2}}║")
            if wc_exit_actions:
                if normal_exits:
                    print("╠" + "─" * width + "╣")
                print(f"║  {'🏆 特殊池出場 — 週期高點回落 -10%':<{width-2}}║")
                for a in wc_exit_actions:
                    line = f"     {a['symbol']}  {a['shares']} 股 @ ${a.get('current_price', 0):.2f}  P&L: {a.get('pnl_pct', 0):+.0f}%"
                    print(f"║  {line:<{width-2}}║")
                print(f"║  {'   📌 Firstrade Stop-Market 掛單止損價':<{width-2}}║")
                for a in wc_exit_actions:
                    stop_px = a.get("cycle_high", 0) * (1 - WINNER_CYCLE_PULLBACK)
                    line = f"     {a['symbol']}  → Stop ${stop_px:.2f}"
                    print(f"║  {line:<{width-2}}║")

        if stop_reminders:
            if exit_actions:
                print("╠" + "─" * width + "╣")
            print(f"║  {'📌 停損單需更新（最緊批次先掛）':<{width-2}}║")
            for r in stop_reminders:
                batch_str  = f" 第{r['tranche_n']}批" if r["tranche_n"] is not None else ""
                shares_str = f" {r['shares']}股" if r["tranche_n"] is not None else ""
                line = f"     {r['symbol']}{batch_str}{shares_str}  → 改掛 ${r['effective_stop']:.2f}"
                print(f"║  {line:<{width-2}}║")

        if rotate_actions:
            if exit_actions or stop_reminders:
                print("╠" + "─" * width + "╣")
            print(f"║  {'🔄 ROTATE — 汰弱留強（擇機執行）':<{width-2}}║")
            for a in rotate_actions:
                line = f"     賣 {a['sell_symbol']} → 買 {a['buy_symbol']}  {a['buy_shares']} 股"
                print(f"║  {line:<{width-2}}║")

        if wc_reentry_adds:
            if exit_actions or stop_reminders or rotate_actions:
                print("╠" + "─" * width + "╣")
            print(f"║  {'🔁 特殊池回補 — 低點反彈 +10%':<{width-2}}║")
            for a in wc_reentry_adds:
                line = f"     {a['symbol']}  {a.get('suggested_shares', 0)} 股 @ ${a.get('current_price', 0):.2f}"
                print(f"║  {line:<{width-2}}║")

        if add_actions:
            if exit_actions or stop_reminders or rotate_actions or wc_reentry_adds:
                print("╠" + "─" * width + "╣")
            print(f"║  {'🟢 ADD — 新倉/加碼（視現金與判斷執行）':<{width-2}}║")
            for a in add_actions:
                if a.get("is_winner_cycle"):
                    continue
                pyramid_str = f" 第{a['tranche_n']}批加碼" if a.get("is_pyramid") else ""
                line = f"     {a['symbol']}{pyramid_str}  {a.get('suggested_shares', 0)} 股 @ ${a.get('current_price', 0):.2f}"
                print(f"║  {line:<{width-2}}║")

        print("╚" + "═" * width + "╝")
        print()

    # 9.5 偏離成本追蹤（每週一自動顯示，其他日子用 --deviation 查看）
    if date.today().weekday() == 0:
        try:
            print_deviation_report(days=30)
        except Exception as e:
            print(f"⚠ 偏離成本追蹤失敗：{e}")

    # 10. 發送 Email 通知
    if send_email:
        notifier = GmailNotifier()
        if notifier.is_configured():
            print("正在發送 Email 通知...")
            if notifier.send_premarket_report(actions_output):
                print(f"Email 已發送至 {notifier.recipient}")
            else:
                print("Email 發送失敗，請檢查 .env 設定")
    else:
        print("（--no-email：跳過 Email 發送）")

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
    parser.add_argument("--deviation", nargs="?", const=30, type=int,
                        metavar="DAYS", help="偏離成本追蹤（預設近30天）")
    parser.add_argument("--no-email", action="store_true", help="跳過 Email 發送（測試用）")
    args = parser.parse_args()

    if args.init:
        run_init()
    elif args.watch:
        run_watch(args.watch)
    elif args.snapshot:
        run_snapshot(args.snapshot)
    elif args.momentum:
        run_momentum(args.momentum)
    elif args.deviation:
        print_deviation_report(days=args.deviation)
    else:
        run_premarket(scan_tw=args.tw, send_email=not args.no_email)


if __name__ == "__main__":
    main()
