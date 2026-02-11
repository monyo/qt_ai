# 盤前建議 + 部位管理系統

美股量化掃描系統，結合**動能策略**、AI 情緒分析（Gemini）、板塊監控，每日盤前產出持倉建議。

## 策略核心（v0.3.0）

| 項目 | 規則 |
|------|------|
| **進場** | 動能排名前 5 名（過去 21 天報酬最高） |
| **出場** | 只有硬停損 -35%（讓獲利奔跑） |
| **候選池** | S&P 500 成分股 + 白名單 + 現有持倉 |

---

## 安裝

```bash
# 使用 conda（推薦）
conda activate qt_env

# 或使用 pip
pip install -r requirements.txt

# 設定 API Key 與 Email
touch .env
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "EMAIL_ENABLED=true" >> .env
echo "GMAIL_SENDER=your_gmail@gmail.com" >> .env
echo "GMAIL_APP_PASSWORD=your_gmail_app_password" >> .env
echo "GMAIL_RECIPIENT=your_recipient_email@gmail.com" >> .env
```
> **注意**: `GMAIL_APP_PASSWORD` 需要在你的 Google 帳戶中產生「應用程式密碼」，而不是你的登入密碼。

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
| **盤前建議** | 載入持倉 → 動能排名 → 三層出場檢查 → 輸出 actions |
| **動能策略** | 追漲：買過去 21 天表現最好的股票 |
| **三層出場** | 移動停利 -15% / MA200 停損 / 極端停損 -35% |
| **板塊監控** | 追蹤科技/軟體/半導體 vs 大盤，板塊走弱時警告 |
| **曝險警告** | 當科技股佔比高 + 板塊走弱時特別提醒 |
| **郵件通知** | 每日盤前自動發送 HTML 格式的分析報告 |
| **持倉追蹤** | 記錄 avg_price、cost_basis、high_since_entry |
| **年度 P&L** | 建立年度快照，追蹤年度績效 |

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

# 查看動能排名（不執行完整分析）
python premarket_main.py --momentum 20
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
- **掃描範圍**：完整的 S&P 500 + 白名單

---

## 盤前報告範例

```
============================================================
  盤前報告 2026-02-11  |  版本 0.4.0
============================================================
  投組總值: $  125,947.75
  現金:     $      155.00
  個股:     12/30 檔
  2026年度:  $ -1,023.18 (-0.8%)
============================================================

--- 板塊相對強弱 (過去5日) 🟢 ---
  大盤 SPY: +0.4%
  🟡 科技     +0.3% (vs SPY: -0.1%)
  🟢 半導體    +1.8% (vs SPY: +1.4%)

--- HOLD (繼續持有) ---
  [core] VOO    97 股 @ $636.44  P&L: +49.20%  動能: -0.3%  1Y: +0% 🟢
         NVDA   10 股 @ $188.54  P&L: +80.04%  動能: +2.0%  1Y: +26% 🟢

--- ADD (建議買入) ---
  [#1] GLW    建議 6 股 @ $128.10  動能: +50.3%  1Y vs SPY: +133% 🟢
         原因: 動能排名 #1（+50.3%）
  [#2] SNDK   建議 1 股 @ $541.64  動能: +43.5%  1Y vs SPY: +1389% 🟢
         原因: 動能排名 #2（+43.5%）
  [#5] CHTR   建議 3 股 @ $248.19  動能: +17.9%  1Y vs SPY: -45% 🔴
         原因: 動能排名 #5（+17.9%）⚠️ 1年落後大盤 -45%
```
