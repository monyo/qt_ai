"""偏離成本追蹤器

比較「系統建議」vs「實際執行」，量化偏離系統的成本：
  1. 沒買的 ADD（pending/skipped）→ 機會成本 = 股數 ×（現價 − 建議時價）
  2. 延遲進場（先跳過、之後才買）→ 成本 = 實際股數 ×（實際成交價 − 首次建議價）
  3. 沒執行的 EXIT → 成本 = 股數 ×（建議時價 − 現價）
  4. 沒執行的 ROTATE → 成本 =（該買的漲幅）−（該賣的漲幅）
  5. 手動賣出（不在系統建議內，如恐慌平倉）→ 成本 = 股數 ×（現價 − 賣出日價）

正值 = 偏離讓你少賺/多虧，負值 = 偏離反而省錢。
"""
import json
import glob
import os
from datetime import date, timedelta

ACTIONS_GLOB = "data/actions_*.json"


def _load_actions_files(days):
    """載入時間窗內的 actions 檔，依日期排序。回傳 [(date_str, data), ...]"""
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    out = []
    for path in sorted(glob.glob(ACTIONS_GLOB)):
        fname = os.path.basename(path)
        dstr = fname.replace("actions_", "").replace(".json", "")
        if not dstr.isdigit() or dstr < cutoff:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                out.append((dstr, json.load(f)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def collect_deviations(days=30):
    """掃描時間窗內的偏離事件。回傳 (deviations, symbols_needed)

    deviations: list of dict {type, date, symbol, shares, ref_price, ...}
    symbols_needed: 需要現價的標的集合
    """
    files = _load_actions_files(days)
    skipped_adds = {}      # symbol -> first occurrence
    confirmed_adds = {}    # symbol -> first confirmed after skip
    skipped_exits = {}     # symbol -> first occurrence
    skipped_rotates = {}   # (sell, buy) -> first occurrence
    manual_sells = []

    prev_holds = None      # {symbol: (shares, price)} from previous file
    prev_date = None
    prev_actions = []

    for dstr, data in files:
        actions = data.get("actions", [])
        holds = {a["symbol"]: (a.get("shares", 0), a.get("current_price", 0))
                 for a in actions if a["action"] == "HOLD"}

        # 手動賣出偵測：前一天 HOLD、今天消失，且前一天沒有 confirmed EXIT/ROTATE
        if prev_holds is not None:
            confirmed_out = set()
            for a in prev_actions:
                if a.get("status") != "confirmed":
                    continue
                if a["action"] == "EXIT":
                    confirmed_out.add(a.get("symbol"))
                elif a["action"] == "ROTATE":
                    confirmed_out.add(a.get("sell_symbol"))
            for sym, (sh, px) in prev_holds.items():
                if sym not in holds and sym not in confirmed_out and sh and px:
                    manual_sells.append({
                        "type": "manual_sell", "date": prev_date,
                        "symbol": sym, "shares": sh, "ref_price": px,
                    })

        for a in actions:
            act, st = a["action"], a.get("status")
            if act == "ADD":
                sym = a.get("symbol")
                if a.get("is_backup") or not a.get("suggested_shares"):
                    continue
                if st in ("pending", "skipped"):
                    skipped_adds.setdefault(sym, {
                        "type": "skipped_add", "date": dstr, "symbol": sym,
                        "shares": a["suggested_shares"],
                        "ref_price": a.get("current_price", 0),
                    })
                elif st == "confirmed" and sym in skipped_adds and sym not in confirmed_adds:
                    confirmed_adds[sym] = {
                        "date": dstr,
                        "actual_shares": a.get("actual_shares") or a.get("suggested_shares"),
                        "actual_price": a.get("actual_price") or a.get("current_price", 0),
                    }
            elif act == "EXIT" and st in ("pending", "skipped"):
                sym = a.get("symbol")
                skipped_exits.setdefault(sym, {
                    "type": "skipped_exit", "date": dstr, "symbol": sym,
                    "shares": a.get("shares", 0),
                    "ref_price": a.get("current_price", 0),
                })
            elif act == "ROTATE" and st in ("pending", "skipped"):
                key = (a.get("sell_symbol"), a.get("buy_symbol"))
                skipped_rotates.setdefault(key, {
                    "type": "skipped_rotate", "date": dstr,
                    "sell_symbol": key[0], "buy_symbol": key[1],
                    "sell_shares": a.get("sell_shares", 0),
                    "sell_price": a.get("sell_price", 0),
                    "buy_shares": a.get("buy_shares", 0),
                    "buy_price": a.get("buy_price", 0),
                })

        prev_holds, prev_date, prev_actions = holds, dstr, actions

    deviations = []
    symbols = set()

    for sym, ev in skipped_adds.items():
        if sym in confirmed_adds:
            c = confirmed_adds[sym]
            deviations.append({
                "type": "delayed_add", "date": ev["date"], "symbol": sym,
                "shares": c["actual_shares"], "ref_price": ev["ref_price"],
                "exec_price": c["actual_price"], "exec_date": c["date"],
            })
        else:
            deviations.append(ev)
            symbols.add(sym)

    for ev in skipped_exits.values():
        deviations.append(ev)
        symbols.add(ev["symbol"])

    for ev in skipped_rotates.values():
        deviations.append(ev)
        symbols.add(ev["sell_symbol"])
        symbols.add(ev["buy_symbol"])

    for ev in manual_sells:
        deviations.append(ev)
        symbols.add(ev["symbol"])

    return deviations, symbols


def _sane(ref, now):
    """30 天窗內價格比例超出 [1/3, 3] 視為資料異常（如分割未調整），排除"""
    return ref > 0 and now > 0 and (1 / 3) <= now / ref <= 3


def compute_costs(deviations, prices):
    """計算每筆偏離成本。回傳 list of dict（含 cost 欄位），無法取價或資料異常的跳過。"""
    out = []
    for ev in deviations:
        t = ev["type"]
        if t == "delayed_add":
            if not _sane(ev["ref_price"], ev["exec_price"]):
                continue  # 建議時報價或成交價異常（如分割/資料錯誤），跳過
            cost = ev["shares"] * (ev["exec_price"] - ev["ref_price"])
            desc = (f"{ev['symbol']} 延遲進場（{ev['date'][4:6]}/{ev['date'][6:]} 建議 "
                    f"${ev['ref_price']:.2f} → {ev['exec_date'][4:6]}/{ev['exec_date'][6:]} "
                    f"買 ${ev['exec_price']:.2f}）")
        elif t == "skipped_add":
            now = prices.get(ev["symbol"])
            if now is None or not _sane(ev["ref_price"], now):
                continue
            cost = ev["shares"] * (now - ev["ref_price"])
            desc = (f"{ev['symbol']} 未買（{ev['date'][4:6]}/{ev['date'][6:]} 建議 "
                    f"${ev['ref_price']:.2f}，現 ${now:.2f}）")
        elif t == "skipped_exit":
            now = prices.get(ev["symbol"])
            if now is None or not _sane(ev["ref_price"], now):
                continue
            cost = ev["shares"] * (ev["ref_price"] - now)
            desc = (f"{ev['symbol']} 未停損（{ev['date'][4:6]}/{ev['date'][6:]} 建議 "
                    f"${ev['ref_price']:.2f}，現 ${now:.2f}）")
        elif t == "skipped_rotate":
            now_s = prices.get(ev["sell_symbol"])
            now_b = prices.get(ev["buy_symbol"])
            if now_s is None or now_b is None:
                continue
            if not _sane(ev["sell_price"], now_s) or not _sane(ev["buy_price"], now_b):
                continue
            cost = (ev["buy_shares"] * (now_b - ev["buy_price"])
                    - ev["sell_shares"] * (now_s - ev["sell_price"]))
            desc = (f"未換股 {ev['sell_symbol']}→{ev['buy_symbol']}"
                    f"（{ev['date'][4:6]}/{ev['date'][6:]}）")
        elif t == "manual_sell":
            now = prices.get(ev["symbol"])
            if now is None or not _sane(ev["ref_price"], now):
                continue
            cost = ev["shares"] * (now - ev["ref_price"])
            desc = (f"{ev['symbol']} 手動賣出（{ev['date'][4:6]}/{ev['date'][6:]} "
                    f"約 ${ev['ref_price']:.2f}，現 ${now:.2f}）")
        else:
            continue
        out.append({**ev, "cost": cost, "desc": desc})
    return out


def print_deviation_report(days=30, top_n=8):
    """印出偏離成本摘要（供盤前報告每週嵌入或 --deviation 手動呼叫）"""
    deviations, symbols = collect_deviations(days)
    if not deviations:
        print(f"\n📊 偏離成本追蹤（近 {days} 天）：無偏離紀錄，完全照系統執行 ✅")
        return

    from src.data_loader import fetch_current_prices
    prices = fetch_current_prices(sorted(symbols)) if symbols else {}
    costed = compute_costs(deviations, prices)
    if not costed:
        print(f"\n📊 偏離成本追蹤（近 {days} 天）：無法取得報價，略過")
        return

    total = sum(e["cost"] for e in costed)
    n_pos = sum(1 for e in costed if e["cost"] > 0)

    print()
    print("=" * 64)
    print(f"📊 偏離成本追蹤（近 {days} 天，{len(costed)} 筆偏離）")
    print("-" * 64)
    print(f"   總偏離成本：${total:+,.0f}（正值 = 偏離讓你少賺/多虧）")
    print(f"   {n_pos}/{len(costed)} 筆偏離是虧的")
    print("-" * 64)
    for e in sorted(costed, key=lambda x: -abs(x["cost"]))[:top_n]:
        print(f"   {e['cost']:>+9,.0f}  {e['desc']}")
    if len(costed) > top_n:
        rest = sum(e["cost"] for e in sorted(costed, key=lambda x: -abs(x["cost"]))[top_n:])
        print(f"   {rest:>+9,.0f}  …其餘 {len(costed) - top_n} 筆")
    print("=" * 64)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print_deviation_report(days=days)
