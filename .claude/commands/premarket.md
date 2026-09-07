執行盤前分析系統，產出今日 ADD/EXIT/HOLD 建議，並對矛盾訊號的標的加入質化分析。

步驟：

## 1. 執行盤前系統（不寄信）

```
conda run -n qt_env python premarket_main.py --no-email
```

## 2. 讀取 actions JSON

讀取 `data/actions_YYYYMMDD.json`（今天日期），確認 ADD/EXIT/ROTATE 建議。

## 3. 識別需要質化分析的候選標的

從 ADD 與 ROTATE actions 中找出符合以下任一條件的標的：
- RSI > 75（`rsi` 欄位）
- 趨勢狀態為「轉弱」（`trend_state == "轉弱"`）
- 備選清單（`is_backup == true`）
- ROTATE 換入目標（action == "ROTATE" 的 `buy_symbol`）
- ROTATE 賣出標的（action == "ROTATE" 的 `sell_symbol`）——**必做**：現行邏輯已加上「自身動能須比 2 個月前更差」的量化檢查（見 `src/premarket.py` `_self_momentum_not_declining`），但量化過濾不出「這次轉弱是基本面確實惡化，還是財報型錯殺/短期雜訊」，賣出端跟買入端一樣值得把關，不要預設賣出一定合理

## 4. 質化分析

對每個識別出的標的，以 Claude 在 context 內做分析，評估：

**觸發說明**：為何觸發質化分析（RSI 過高、趨勢轉弱、備選清單、ROTATE 賣出標的等）

**催化劑**（1-2句）：近期有什麼事件或趨勢支撐這支股票？產業地位/競爭優勢？
　（ROTATE 賣出標的：改問「近期是否有基本面利空（財報/展望/產業結構）足以支撐轉弱訊號？」）

**主要風險**（1-2句）：目前最大的不確定因素？技術面過熱？財報風險？
　（ROTATE 賣出標的：改問「賣出的主要風險是什麼？是否可能只是短期雜訊/錯殺，賣掉會不會太早？」）

**建議**（1句）：現在是好的入場時機嗎？等什麼信號？
　（ROTATE 賣出標的：改問「這次賣出合理嗎？還是應該續抱、等訊號更明確再賣？」）

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
