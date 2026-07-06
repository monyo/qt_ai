執行盤前分析系統，產出今日 ADD/EXIT/HOLD 建議，並對矛盾訊號的標的加入質化分析。

步驟：

## 1. 執行盤前系統（不寄信）

```
conda run -n qt_env python premarket_main.py --no-email
```

## 2. 讀取 actions JSON

讀取 `data/actions_YYYYMMDD.json`（今天日期），確認 ADD/EXIT/ROTATE 建議。

## 3. 識別需要質化分析的候選標的

從 ADD actions 中找出符合以下任一條件的標的：
- RSI > 75（`rsi` 欄位）
- 趨勢狀態為「轉弱」（`trend_state == "轉弱"`）
- 備選清單（`is_backup == true`）
- ROTATE 換入目標（action == "ROTATE" 的 `buy_symbol`）

## 4. 質化分析

對每個識別出的標的，以 Claude 在 context 內做分析，評估：

**觸發說明**：為何觸發質化分析（RSI 過高、趨勢轉弱、備選清單等）

**催化劑**（1-2句）：近期有什麼事件或趨勢支撐這支股票？產業地位/競爭優勢？

**主要風險**（1-2句）：目前最大的不確定因素？技術面過熱？財報風險？

**建議**（1句）：現在是好的入場時機嗎？等什麼信號？

**評分**（0-100）：
- 80-100：✅ 積極建議（基本面強 + 技術面合理）
- 60-79：🟡 可考慮（有疑慮但可少量建倉）
- 40-59：⚠️ 等待（暫不建議，等信號改善）
- 0-39：❌ 不建議（明確的反對理由）

## 5. 寫入質化分析 JSON

將分析結果寫入 `data/qualitative_YYYYMMDD.json`（今天日期），格式如下：

```json
{
  "date": "YYYY-MM-DD",
  "analyses": [
    {
      "symbol": "XXX",
      "trigger": "RSI 91",
      "score": 62,
      "verdict": "🟡 可考慮",
      "catalyst": "...",
      "risk": "...",
      "recommendation": "..."
    }
  ]
}
```

verdict 必須是以下四個之一：`✅ 積極建議` / `🟡 可考慮` / `⚠️ 等待` / `❌ 不建議`

## 6. 寄送報告

```
conda run -n qt_env python send_report.py YYYY-MM-DD
```

## 7. 摘要輸出

用繁體中文摘要：
- 今日市場體制（BULL/BEAR）
- EXIT/ROTATE/ADD 建議重點
- 質化分析結論（哪些有顧慮，哪些可以放心買）
