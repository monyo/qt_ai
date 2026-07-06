"""台股回調提醒 + 持倉停損監控

每日執行，自動偵測：
  - 觀察清單的五雄等強勢股是否回調到目標進場價
  - 持倉是否跌破 -10% 停損線

使用方式:
    python tw_monitor.py               # 有警示才寄信
    python tw_monitor.py --force       # 強制寄出每日狀態信
    python tw_monitor.py --check-only  # 只顯示，不寄信

    # 買進後加入持倉監控:
    python tw_monitor.py --buy 5347.TWO 世界先進 188 1000
    python tw_monitor.py --buy 3707.TWO 漢磊 90 2000

    # 賣出後移除:
    python tw_monitor.py --sell 5347.TWO
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yfinance as yf

from src.notifier import GmailNotifier

WATCHLIST_PATH = Path("data/tw_watchlist.json")
STOP_LOSS_PCT = 0.10   # 持倉停損：跌 -10% 寄警告


# ─── Watchlist I/O ────────────────────────────────────────────────────────────

def load_watchlist():
    if not WATCHLIST_PATH.exists():
        return {"pullback_watch": [], "holding": []}
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_watchlist(data):
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Price Fetching ───────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str]) -> dict[str, float | None]:
    prices = {}
    if not symbols:
        return prices
    try:
        raw = yf.download(symbols, period="2d", auto_adjust=True,
                          progress=False, threads=True)
        close = raw["Close"] if len(symbols) > 1 else raw["Close"].to_frame(symbols[0])
        for sym in symbols:
            if sym in close.columns:
                series = close[sym].dropna()
                prices[sym] = round(float(series.iloc[-1]), 2) if not series.empty else None
            else:
                prices[sym] = None
    except Exception as e:
        print(f"[warn] 價格抓取失敗: {e}")
        for sym in symbols:
            prices[sym] = None
    return prices


# ─── Alert Logic ─────────────────────────────────────────────────────────────

def check(watchlist: dict, prices: dict) -> tuple[list, list, list]:
    """回傳 (entry_alerts, stop_alerts, all_status)"""
    entry_alerts, stop_alerts, all_status = [], [], []

    for stock in watchlist.get("pullback_watch", []):
        sym = stock["symbol"]
        price = prices.get(sym)
        target = stock["target_price"]
        ref = stock.get("ref_price", target)

        row = {**stock, "price": price, "category": "watch"}
        if price is not None:
            row["pct_vs_target"] = (price - target) / target * 100
            row["pct_vs_ref"] = (price - ref) / ref * 100
            if price <= target:
                row["alert"] = True
                entry_alerts.append(row)
            else:
                row["alert"] = False
        else:
            row["alert"] = False
        all_status.append(row)

    for stock in watchlist.get("holding", []):
        sym = stock["symbol"]
        price = prices.get(sym)
        entry = stock["entry_price"]
        stop = round(entry * (1 - STOP_LOSS_PCT), 1)

        row = {**stock, "price": price, "stop_price": stop, "category": "holding"}
        if price is not None:
            row["pct_vs_entry"] = (price - entry) / entry * 100
            if price <= stop:
                row["alert"] = True
                stop_alerts.append(row)
            else:
                row["alert"] = False
        else:
            row["alert"] = False
        all_status.append(row)

    return entry_alerts, stop_alerts, all_status


# ─── Console Output ───────────────────────────────────────────────────────────

def print_status(entry_alerts, stop_alerts, all_status):
    today = date.today().isoformat()
    print(f"\n{'─'*60}")
    print(f"  台股監控報告  {today}")
    print(f"{'─'*60}")

    if entry_alerts:
        print("\n🔔 進場機會（回調到目標價）：")
        for r in entry_alerts:
            print(f"  ✅ {r['name']} ({r['symbol']})  現價 {r['price']}  目標 {r['target_price']}  [{r['note']}]")

    if stop_alerts:
        print("\n⚠️  停損警告：")
        for r in stop_alerts:
            print(f"  🔴 {r['name']} ({r['symbol']})  現價 {r['price']}  停損線 {r['stop_price']}  進場 {r['entry_price']}")

    if not entry_alerts and not stop_alerts:
        print("\n  ✓ 無警示，繼續等候")

    watch_items = [r for r in all_status if r["category"] == "watch"]
    hold_items  = [r for r in all_status if r["category"] == "holding"]

    if watch_items:
        print("\n📋 觀察清單（等回調）：")
        print(f"  {'名稱':<8} {'代號':<12} {'現價':>7} {'目標':>7} {'距目標':>8} {'參考':>7}")
        print(f"  {'─'*58}")
        for r in watch_items:
            price_str = f"{r['price']:7.1f}" if r['price'] else "    N/A"
            pct_str = f"{r.get('pct_vs_target', 0):+7.1f}%" if r['price'] else "    N/A"
            flag = "🔔" if r["alert"] else "  "
            print(f"  {flag} {r['name']:<8} {r['symbol']:<12} {price_str} {r['target_price']:7.1f} {pct_str} {r.get('ref_price', 0):7.1f}")

    if hold_items:
        print("\n📦 持倉監控：")
        print(f"  {'名稱':<8} {'代號':<12} {'現價':>7} {'進場':>7} {'損益%':>8} {'停損線':>7}")
        print(f"  {'─'*58}")
        for r in hold_items:
            price_str = f"{r['price']:7.1f}" if r['price'] else "    N/A"
            pct_str = f"{r.get('pct_vs_entry', 0):+7.1f}%" if r['price'] else "    N/A"
            flag = "🔴" if r["alert"] else "  "
            print(f"  {flag} {r['name']:<8} {r['symbol']:<12} {price_str} {r['entry_price']:7.1f} {pct_str} {r['stop_price']:7.1f}")

    print(f"\n{'─'*60}\n")


# ─── Email Formatting ─────────────────────────────────────────────────────────

def _row_color(alert: bool, category: str) -> str:
    if alert and category == "holding":
        return "background:#f8d7da;"
    if alert and category == "watch":
        return "background:#d4edda;"
    return ""


def format_html(entry_alerts, stop_alerts, all_status, today: str) -> str:
    alert_count = len(entry_alerts) + len(stop_alerts)
    header_color = "#dc3545" if stop_alerts else ("#28a745" if entry_alerts else "#6c757d")

    alert_section = ""
    if entry_alerts:
        rows = "".join(
            f"<tr style='background:#d4edda;'>"
            f"<td><b>{r['name']}</b></td><td>{r['symbol']}</td>"
            f"<td style='text-align:right;'>{r['price']}</td>"
            f"<td style='text-align:right;'>{r['target_price']}</td>"
            f"<td style='text-align:right;color:#28a745;'>{r.get('pct_vs_target',0):+.1f}%</td>"
            f"<td style='font-size:11px;color:#555;'>{r['note']}</td></tr>"
            for r in entry_alerts
        )
        alert_section += f"""
<h3 style="color:#28a745;">🔔 進場機會</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#f0f0f0;">
    <th style="text-align:left;padding:4px 8px;">名稱</th>
    <th>代號</th><th>現價</th><th>目標價</th><th>距目標</th><th>備註</th>
  </tr>{rows}
</table>"""

    if stop_alerts:
        rows = "".join(
            f"<tr style='background:#f8d7da;'>"
            f"<td><b>{r['name']}</b></td><td>{r['symbol']}</td>"
            f"<td style='text-align:right;'>{r['price']}</td>"
            f"<td style='text-align:right;'>{r['entry_price']}</td>"
            f"<td style='text-align:right;color:#dc3545;'>{r.get('pct_vs_entry',0):+.1f}%</td>"
            f"<td style='text-align:right;'>{r['stop_price']}</td></tr>"
            for r in stop_alerts
        )
        alert_section += f"""
<h3 style="color:#dc3545;">⚠️ 停損警告</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#f0f0f0;">
    <th style="text-align:left;padding:4px 8px;">名稱</th>
    <th>代號</th><th>現價</th><th>進場價</th><th>損益</th><th>停損線</th>
  </tr>{rows}
</table>"""

    watch_rows = ""
    for r in all_status:
        if r["category"] != "watch":
            continue
        bg = "background:#d4edda;" if r["alert"] else ""
        pct = f"{r.get('pct_vs_target',0):+.1f}%" if r["price"] else "N/A"
        dist_color = "color:#28a745;font-weight:bold;" if r["alert"] else ""
        watch_rows += (
            f"<tr style='{bg}'>"
            f"<td style='padding:3px 8px;'><b>{r['name']}</b></td>"
            f"<td>{r['symbol']}</td>"
            f"<td style='text-align:right;'>{r['price'] or 'N/A'}</td>"
            f"<td style='text-align:right;'>{r['target_price']}</td>"
            f"<td style='text-align:right;{dist_color}'>{pct}</td>"
            f"<td style='font-size:11px;color:#555;'>{r['note']}</td></tr>"
        )

    hold_rows = ""
    for r in all_status:
        if r["category"] != "holding":
            continue
        bg = "background:#f8d7da;" if r["alert"] else ""
        pct = f"{r.get('pct_vs_entry',0):+.1f}%" if r["price"] else "N/A"
        pct_color = "color:#dc3545;font-weight:bold;" if r["alert"] else (
            "color:#28a745;" if r.get("pct_vs_entry", 0) > 0 else "")
        hold_rows += (
            f"<tr style='{bg}'>"
            f"<td style='padding:3px 8px;'><b>{r['name']}</b></td>"
            f"<td>{r['symbol']}</td>"
            f"<td style='text-align:right;'>{r['price'] or 'N/A'}</td>"
            f"<td style='text-align:right;'>{r['entry_price']}</td>"
            f"<td style='text-align:right;{pct_color}'>{pct}</td>"
            f"<td style='text-align:right;'>{r['stop_price']}</td>"
            f"<td style='font-size:11px;color:#555;'>{r.get('note','')}</td></tr>"
        )

    watch_section = f"""
<h3 style="color:#333;border-bottom:1px solid #ddd;padding-bottom:4px;">📋 觀察清單（等回調進場）</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#f0f0f0;">
    <th style="text-align:left;padding:4px 8px;">名稱</th>
    <th>代號</th><th>現價</th><th>目標價</th><th>距目標</th><th>備註</th>
  </tr>{watch_rows or '<tr><td colspan="6" style="color:#888;padding:6px;">（空）</td></tr>'}
</table>""" if any(r["category"] == "watch" for r in all_status) else ""

    hold_section = f"""
<h3 style="color:#333;border-bottom:1px solid #ddd;padding-bottom:4px;">📦 持倉監控（-10% 停損）</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#f0f0f0;">
    <th style="text-align:left;padding:4px 8px;">名稱</th>
    <th>代號</th><th>現價</th><th>進場價</th><th>損益</th><th>停損線</th><th>備註</th>
  </tr>{hold_rows or '<tr><td colspan="7" style="color:#888;padding:6px;">（尚未設定持倉）</td></tr>'}
</table>""" if True else ""

    subject_tag = f"🔔 {alert_count} 個警示" if alert_count else "✓ 無警示"

    return f"""<html>
<body style="font-family:Arial,sans-serif;max-width:680px;margin:0 auto;padding:20px;color:#333;">
  <h2 style="color:{header_color};margin-bottom:4px;">台股監控報告 {today}</h2>
  <p style="color:#888;font-size:13px;margin-top:0;">{subject_tag}　停損設定：進場價 ×{1-STOP_LOSS_PCT:.0%}</p>
  {alert_section}
  {watch_section}
  {hold_section}
  <hr style="margin:24px 0;border:none;border-top:1px solid #eee;">
  <p style="color:#aaa;font-size:11px;">執行：python tw_monitor.py　　加倉：python tw_monitor.py --buy 代號 名稱 進場價 股數</p>
</body>
</html>"""


# ─── CLI Commands ─────────────────────────────────────────────────────────────

def cmd_buy(args):
    watchlist = load_watchlist()
    holding = watchlist.setdefault("holding", [])
    symbol = args.symbol
    existing = next((h for h in holding if h["symbol"] == symbol), None)
    if existing:
        print(f"已存在 {symbol}，更新進場價與股數")
        existing.update({"name": args.name, "entry_price": args.price,
                         "shares": args.shares, "note": args.note or ""})
    else:
        holding.append({"symbol": symbol, "name": args.name,
                         "entry_price": args.price, "shares": args.shares,
                         "note": args.note or ""})
    save_watchlist(watchlist)
    stop = round(args.price * (1 - STOP_LOSS_PCT), 1)
    print(f"✅ 加入持倉：{args.name} ({symbol})  進場 {args.price}  停損線 {stop}  {args.shares} 股")


def cmd_sell(args):
    watchlist = load_watchlist()
    before = len(watchlist.get("holding", []))
    watchlist["holding"] = [h for h in watchlist.get("holding", [])
                             if h["symbol"] != args.symbol]
    if len(watchlist["holding"]) < before:
        print(f"✅ 已從持倉移除：{args.symbol}")
    else:
        print(f"找不到持倉：{args.symbol}")
    save_watchlist(watchlist)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="台股回調提醒 + 持倉停損監控")
    parser.add_argument("--force", action="store_true", help="強制寄出每日狀態信（即使無警示）")
    parser.add_argument("--check-only", action="store_true", help="只顯示狀態，不寄信")

    sub = parser.add_subparsers(dest="cmd")

    buy_p = sub.add_parser("--buy", help="加入持倉")
    buy_p.add_argument("symbol")
    buy_p.add_argument("name")
    buy_p.add_argument("price", type=float)
    buy_p.add_argument("shares", type=int)
    buy_p.add_argument("--note", default="")

    sell_p = sub.add_parser("--sell", help="移除持倉")
    sell_p.add_argument("symbol")

    # 手動解析 --buy / --sell（argparse subparser 在這種格式下需要特殊處理）
    argv = sys.argv[1:]
    if "--buy" in argv:
        idx = argv.index("--buy")
        rest = argv[idx+1:]
        note_val = ""
        if "--note" in rest:
            ni = rest.index("--note")
            note_val = rest[ni+1] if ni+1 < len(rest) else ""
            rest = rest[:ni]
        if len(rest) < 4:
            print("用法: python tw_monitor.py --buy 代號 名稱 進場價 股數 [--note 備註]")
            sys.exit(1)

        class BuyArgs:
            symbol = rest[0]
            name = rest[1]
            price = float(rest[2])
            shares = int(rest[3])
            note = note_val

        cmd_buy(BuyArgs())
        return

    if "--sell" in argv:
        idx = argv.index("--sell")
        rest = argv[idx+1:]
        if not rest:
            print("用法: python tw_monitor.py --sell 代號")
            sys.exit(1)

        class SellArgs:
            symbol = rest[0]

        cmd_sell(SellArgs())
        return

    args = parser.parse_args(argv)

    watchlist = load_watchlist()
    all_symbols = (
        [s["symbol"] for s in watchlist.get("pullback_watch", [])] +
        [s["symbol"] for s in watchlist.get("holding", [])]
    )

    print(f"抓取 {len(all_symbols)} 支股票報價...")
    prices = fetch_prices(all_symbols)

    entry_alerts, stop_alerts, all_status = check(watchlist, prices)
    print_status(entry_alerts, stop_alerts, all_status)

    if args.check_only:
        return

    has_alerts = bool(entry_alerts or stop_alerts)
    if not has_alerts and not args.force:
        print("無警示，今日不寄信。（--force 可強制寄出）")
        return

    notifier = GmailNotifier()
    if not notifier.is_configured():
        print("Email 未設定（檢查 .env GMAIL_* 欄位）")
        return

    today = date.today().isoformat()
    alert_count = len(entry_alerts) + len(stop_alerts)
    subject = (
        f"台股監控 {today} 🔔 {alert_count} 個警示" if has_alerts
        else f"台股監控 {today} ✓ 無警示"
    )
    html = format_html(entry_alerts, stop_alerts, all_status, today)
    text = f"台股監控 {today}\n進場機會: {len(entry_alerts)}\n停損警告: {len(stop_alerts)}"

    if notifier._send_email(subject, text, html):
        print(f"Email 已發送至 {notifier.recipient}")
    else:
        print("Email 發送失敗")


if __name__ == "__main__":
    main()
