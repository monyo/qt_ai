"""盤前質化分析候選整理與 Hermes 產生器。"""
import json
import os
import re
import subprocess

from dotenv import load_dotenv

load_dotenv()

_VERDICTS = ("✅ 積極建議", "🟡 可考慮", "⚠️ 等待", "❌ 不建議")
_OVERRIDES = ("upgrade", "confirm", "confirm_but_wait", "downgrade", "reject", "hold_sell")


def collect_qualitative_candidates(actions_data):
    """依盤前規則建立完整、去重的質化分析候選清單。"""
    candidates = {}
    actions = actions_data.get("actions", [])

    def add(symbol, trigger, strategy_type, action_data):
        if not symbol:
            return
        item = candidates.setdefault(symbol, {
            "symbol": symbol,
            "triggers": [],
            "strategy_type": strategy_type,
            "metrics": {},
        })
        if trigger not in item["triggers"]:
            item["triggers"].append(trigger)
        for key in ("current_price", "momentum", "rsi", "alpha_1y", "alpha_3y", "trend_state", "reason",
                    "forward_pe", "peg", "target_mean", "upside_pct", "revenue_growth",
                    "earnings_growth", "gross_margin", "operating_margin", "free_cashflow",
                    "operating_cashflow", "total_debt", "total_cash", "roic",
                    "shares_outstanding", "market_cap", "enterprise_value", "sector",
                    "screen_quality_score", "screen_quality_coverage", "screen_source"):
            if action_data.get(key) is not None:
                item["metrics"][key] = action_data[key]

    for action in actions:
        symbol = action.get("symbol")
        trend = action.get("trend_state")
        trend_state = trend.get("state") if isinstance(trend, dict) else trend
        if isinstance(action.get("rsi"), (int, float)) and action["rsi"] > 75:
            add(symbol, f"RSI > 75 ({action['rsi']:.1f})", "Momentum", action)
        if trend_state == "轉弱":
            add(symbol, "趨勢狀態為轉弱", "Momentum", action)
        if action.get("is_backup") is True:
            add(symbol, "ADD 備選", "Momentum", action)

        if action.get("action") == "ROTATE":
            sell = action.get("sell_symbol")
            buy = action.get("buy_symbol")
            add(buy, "ROTATE 換入", "ROTATE", action)
            add(sell, "ROTATE 賣出", "ROTATE", action)
            if buy:
                candidates[buy]["metrics"]["rotate_from"] = sell
            if sell:
                candidates[sell]["metrics"]["rotate_to"] = buy

    for action in actions_data.get("value_actions", []):
        if action.get("action") == "VALUE_ADD":
            value_type = action.get("value_type")
            label = "Early Value" if value_type == "early_value" else "Value With Confirmation"
            add(action.get("symbol"), f"VALUE_ADD ({label})", label, action)
            candidates[action["symbol"]]["metrics"].update({
                key: action.get(key) for key in
                ("forward_pe", "peg", "target_mean", "upside_pct", "revenue_growth", "earnings_growth",
                 "gross_margin", "operating_margin", "free_cashflow", "operating_cashflow",
                 "total_debt", "total_cash", "roic", "shares_outstanding", "market_cap",
                 "enterprise_value", "sector")
                if action.get(key) is not None
            })

    for candidate in actions_data.get("fundamental_screen_candidates", []):
        symbol = candidate.get("symbol")
        add(symbol, "S&P 500 獨立基本面/估值初篩", "Fundamental Screen", candidate)

    return list(candidates.values())


def _build_research_packets(actions_data):
    """將量化候選整理成可追溯的研究資料包；不替 LLM 補造事實。"""
    candidates = collect_qualitative_candidates(actions_data)
    packets = []
    for candidate in candidates:
        metrics = candidate.get("metrics", {})
        packets.append({
            "symbol": candidate["symbol"],
            "research_archetype": candidate["strategy_type"],
            "triggers": candidate["triggers"],
            "known_metrics": metrics,
            "data_quality": {
                "source": "premarket quant pipeline / yfinance snapshot",
                "as_of": actions_data.get("date"),
                "limitations": [
                    "未提供 point-in-time 財務序列",
                    "未提供完整財報、電話會議與獨立來源證據",
                    "分析師目標價不可視為內在價值證明",
                ],
            },
        })
    return packets


def _decode_json(text):
    """從 Hermes 輸出擷取 JSON，並保留可診斷的解析錯誤。"""
    match = re.search(r"\{", text or "")
    if not match:
        return None, "回覆中找不到 JSON object"
    candidate = text[match.start():]
    # 模型偶爾把分數輸出成英文字面值；只正規化數值欄位的已知偏差。
    for word, number in (("fifty", "50"), ("forty", "40"), ("thirty", "30")):
        candidate = re.sub(rf"(:\s*){word}(?=\s*[,}}])", rf"\g<1>{number}", candidate)
    try:
        return json.JSONDecoder().raw_decode(candidate)[0], None
    except json.JSONDecodeError as exc:
        start = max(0, exc.pos - 60)
        end = min(len(candidate), exc.pos + 60)
        context = candidate[start:end].replace("\n", "\\n")
        return None, f"JSON 格式錯誤 line={exc.lineno} col={exc.colno}: {context!r}"


def _extract_json(text):
    """相容介面：只回傳 JSON object；詳細錯誤由 _decode_json 提供。"""
    return _decode_json(text)[0]


def _build_hermes_command(prompt, env=None):
    """Build the child command using an explicit caller route or Hermes config."""
    env = os.environ if env is None else env
    command = [
        env.get("HERMES_BIN", "hermes"), "chat", "-q", prompt, "-t", "web", "-Q",
        "--source", "qt-qualitative",
    ]
    provider = env.get("QT_QUALITATIVE_PROVIDER", "").strip()
    model = env.get("QT_QUALITATIVE_MODEL", "").strip()
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    return command


def generate_qualitative_data(actions_data):
    """透過 Hermes 目前設定的模型產生研究分析，不直接呼叫第二個模型 API。"""
    packets = _build_research_packets(actions_data)
    if not packets:
        return None

    prompt = f"""
你是嚴謹的股票投資研究員。請對每個 research packet 做完整研究，不得漏任何標的。
VALUE_ADD 是獨立價值研究，優先檢查「低估但動能尚未反應」的可能性；不要因為動能為負就否定，也不要因為 PE/PEG 低就直接推薦。
你可以使用 web 工具查找最新公司公告、財報與電話會議；每個外部事實必須附 source、published_at、type、source_type。
證據欄位嚴格分開：type 只能表示認知類型 fact/inference/assumption/unknown；source_type 才表示文件類型（例如 SEC filing、company earnings release、earnings call transcript、quant snapshot）。絕不可把文件類型放進 type。
若無法查證，填入 unknown，不可捏造數字或新聞。

每檔輸出完整欄位：
symbol, research_archetype, core_thesis, business_model, evidence,
market_expectation, catalysts, risks, counter_case, invalidation_conditions,
scenarios, scores, action, position_guidance, next_review_triggers,
quant_action_override, final_recommendation。

規則：
- evidence.type 只能是 fact、inference、assumption、unknown。
- catalyst 必須包含 event、transmission、time_horizon、confirmation_signal。
- risk 必須包含 event、financial_transmission、warning_signal。
- scenarios 必須包含 bear、base、bull，分別寫假設與結果，不必捏造精確價格。
- scores 分別是 business_quality、thesis_strength、valuation_margin、catalyst_quality、risk_asymmetry，0-100。
- action 只能是 strong_buy、buy、watchlist、hold、reduce、reject、insufficient_evidence。
- quant_action_override 只能是 upgrade、confirm、confirm_but_wait、downgrade、reject、hold_sell。
- 對 ROTATE 賣出端，必須判斷賣出合理、部分調整、續抱或等待確認。
- 分析師目標價只是市場預期資料，不是內在價值證明。
- 最後的 recommendation 必須由證據、反證與估值假設推導，不可只重述量化指標。

research packets:
{json.dumps(packets, ensure_ascii=False)}

嚴格只輸出 JSON：{{"date":"{actions_data.get('date')}","analyses":[...]}}
"""
    try:
        command = _build_hermes_command(prompt)
        provider = os.environ.get("QT_QUALITATIVE_PROVIDER", "").strip()
        model = os.environ.get("QT_QUALITATIVE_MODEL", "").strip()
        route = f"{provider or 'Hermes profile default'}{('/' + model) if model else ''}"
        print(f"質化子程序模型路由：{route}")
        result = subprocess.run(
            command,
            # 逐檔 web evidence 研究可能超過一般單次 CLI timeout；
            # 保持 fail-closed，但給完整質化分析足夠完成時間。
            cwd=os.getcwd(), text=True, capture_output=True, timeout=900,
        )
        if result.returncode != 0:
            print(f"⚠ Hermes 質化分析失敗：{result.stderr[-200:]}")
            return None
        data, parse_error = _decode_json(result.stdout)
        if not data:
            print(f"⚠ 質化分析輸出無效：{parse_error or 'JSON 為空'}")
            return None
        analyses = data.get("analyses", [])
        by_symbol = {item.get("symbol"): item for item in analyses if item.get("symbol")}
        expected_symbols = {item["symbol"] for item in packets}
        if set(by_symbol) != expected_symbols:
            print(
                "⚠ 質化分析標的缺漏："
                f"missing={sorted(expected_symbols - set(by_symbol))}, "
                f"extra={sorted(set(by_symbol) - expected_symbols)}"
            )
            return None
        for item in by_symbol.values():
            try:
                raw_score = item.get("score")
                if raw_score is None:
                    score_parts = item.get("scores", {})
                    raw_score = sum(
                        float(score_parts.get(key, 50))
                        for key in ("business_quality", "thesis_strength", "valuation_margin", "catalyst_quality", "risk_asymmetry")
                    ) / 5
                item["score"] = max(0, min(100, int(float(raw_score))))
            except (TypeError, ValueError):
                item["score"] = 50
            item["verdict"] = (
                "✅ 積極建議" if item["score"] >= 80 else
                "🟡 可考慮" if item["score"] >= 60 else
                "⚠️ 等待" if item["score"] >= 40 else
                "❌ 不建議"
            )
            if item.get("quant_action_override") not in _OVERRIDES:
                print(f"⚠ {item.get('symbol')} 質化分析缺少有效 quant_action_override")
                return None
            if not str(item.get("final_recommendation", "")).strip():
                print(f"⚠ {item.get('symbol')} 質化分析缺少 final_recommendation")
                return None
            if not item.get("core_thesis") or not item.get("counter_case"):
                print(f"⚠ {item.get('symbol')} 質化分析缺少 thesis/counter_case")
                return None
            if not isinstance(item.get("evidence"), list):
                print(f"⚠ {item.get('symbol')} 質化分析 evidence 欄位格式錯誤")
                return None
            for evidence in item["evidence"]:
                if not isinstance(evidence, dict) or evidence.get("type") not in {
                    "fact", "inference", "assumption", "unknown",
                }:
                    print(f"⚠ {item.get('symbol')} evidence.type 必須是 fact/inference/assumption/unknown")
                    return None
                if evidence["type"] in {"fact", "inference"} and not all(
                    str(evidence.get(key, "")).strip()
                    for key in ("source", "published_at", "source_type")
                ):
                    print(f"⚠ {item.get('symbol')} 外部證據缺少 source/published_at/source_type")
                    return None
            if not isinstance(item.get("invalidation_conditions"), list):
                print(f"⚠ {item.get('symbol')} 質化分析 invalidation_conditions 欄位格式錯誤")
                return None
        data["date"] = actions_data.get("date")
        data["research_method"] = "Hermes model + web evidence + structured bear review"
        data["research_packets"] = packets
        data["analyses"] = [by_symbol[item["symbol"]] for item in packets]
        return data
    except Exception as exc:
        print(f"⚠ 自動質化分析失敗：{str(exc)[:120]}")
        return None
