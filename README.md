# 盤前建議 + 部位管理系統

美股量化掃描系統，結合技術分析（MA60+RSI）、AI 情緒分析（Gemini）、板塊監控，每日盤前產出持倉建議。

---

## 安裝

```bash
# 使用 conda（推薦）
conda activate qt_env

# 或使用 pip
pip install -r requirements.txt

# 設定 API Key（用於 AI 情緒分析）
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## 快速開始

```bash
# 1. 建立你的持倉資料
python premarket_main.py --init

# 2. 每日盤前跑一次，查看建議
python premarket_main.py

# 3. 盤後回報哪些有執行
python confirm_main.py 2026-02-05
```

---

## 功能列表

| 功能 | 說明 |
|------|------|
| **盤前建議** | 載入持倉 → 風控檢查 → 掃描候選 → AI 情緒 → 輸出 actions |
| **板塊監控** | 追蹤科技/軟體/半導體 vs 大盤，板塊走弱時警告 |
| **曝險警告** | 當科技股佔比高 + 板塊走弱時特別提醒 |
| **停損檢查** | 持倉跌破 -35% 時建議 EXIT |
| **技術訊號** | MA60 + RSI 判斷買賣點 |
| **AI 情緒** | 用 Gemini 分析新聞標題，給出 -1.0 ~ +1.0 分數 |
| **持倉追蹤** | 記錄 avg_price、cost_basis、交易紀錄 |
| **年度 P&L** | 建立年度快照，追蹤年度績效 |
| **停損回測** | 比較不同停損策略的歷史表現 |

---

## 每日工作流程

```
盤前（美股開盤前）
├── python premarket_main.py
├── 查看板塊健康狀態
├── 查看 EXIT / HOLD / ADD 建議
└── 決定今天要做什麼

盤中
└── 手動執行交易（系統只給建議，不自動下單）

盤後
├── python confirm_main.py YYYY-MM-DD
├── 標記哪些建議有實際執行
└── 系統自動更新持倉和交易紀錄
```

---

## 指令參考

### 盤前建議
```bash
# 產出今日建議
python premarket_main.py

# 首次使用：互動式建立持倉
python premarket_main.py --init

# 新增白名單標的
python premarket_main.py --watch PLTR COIN

# 建立年度快照（用於追蹤年度 P&L）
python premarket_main.py --snapshot 2026
```

### 確認執行
```bash
# 回報哪些建議有執行，更新持倉
python confirm_main.py 2026-02-05
```

### 停損策略回測
```bash
# 比較不同停損策略
python stop_loss_compare.py NVDA SHOP TSLA GOOG MU
```

### 獨立工具
```bash
# 獨立掃描器（不需要持倉）
python scanner_main.py

# 歷史壓力測試
python main.py

# 板塊相對強弱獨立檢查
python -c "from src.sector_monitor import print_sector_report; print_sector_report()"
```

---

## 資料檔案

所有資料存放在 `data/` 目錄：

| 檔案 | 說明 |
|------|------|
| `portfolio.json` | 你的持倉狀態（股數、成本、交易紀錄） |
| `watchlist.json` | 白名單標的 |
| `actions_YYYYMMDD.json` | 每日盤前建議（含 status: pending/confirmed/skipped） |
| `snapshot_YYYY.json` | 年度快照（用於計算年度 P&L） |
| `*.csv` | 股票歷史數據快取 |

### portfolio.json 結構
```json
{
  "cash": 10000,
  "positions": {
    "VOO": {
      "shares": 20,
      "avg_price": 500.0,
      "cost_basis": 10000,
      "first_entry": "2025-01-01",
      "core": true
    },
    "NVDA": {
      "shares": 10,
      "avg_price": 130.0,
      "cost_basis": 1300,
      "first_entry": "2025-06-01",
      "core": false
    }
  },
  "transactions": [
    {"date": "2025-01-01", "symbol": "VOO", "action": "ADD", "shares": 20, "price": 500.0}
  ]
}
```

---

## 約束條件

- **VOO 保護**：`core: true` 的持倉永遠只會 HOLD，不會建議賣出
- **個股上限**：最多 30 檔（不含 VOO）
- **硬停損**：跌破 -35%（以 avg_price 計算）時強制建議 EXIT
- **掃描範圍**：S&P 500 前 50 檔 + 白名單

---

## 盤前報告範例

```
============================================================
  盤前報告 2026-02-05  |  版本 0.2.0
============================================================
  投組總值: $   38,355.17
  現金:     $   10,000.00
  個股:     5/30 檔
  2026年度:  +$  1,234.56 (+3.2%)
============================================================

--- 板塊相對強弱 (過去5日) 🔴 ---
  大盤 SPY: -1.3%
  🔴 軟體     -13.6% (vs SPY: -12.3%)
  🔴 半導體    -8.3% (vs SPY: -7.0%)
  🟢 金融     +1.8% (vs SPY: +3.2%)

  🚨 注意：你的持股 83% 是科技相關，而科技板塊正在走弱！

--- EXIT (建議出場) ---
  [strategy_signal] SHOP   15 股 @ $114.02  P&L: +14.02%
         原因: 技術面賣出訊號（MA60/RSI）

--- HOLD (繼續持有) ---
  [core] VOO    20 股 @ $630.91  P&L: +26.18%
         NVDA   10 股 @ $174.19  P&L: +33.99%

--- ADD (建議買入) ---
  [scanner] AMT    建議 2 股 @ $179.46
         原因: 技術面買入訊號 + AI 情緒 0.0（中立）
```
