# 盤前建議 + 部位管理系統

美股量化掃描系統，結合**混合動能策略**、趨勢狀態偵測、板塊監控，每日盤前產出持倉建議。

## 策略核心（v0.11.0）

| 項目 | 規則 |
|------|------|
| **動能排名** | 混合動能 = 50% 短期(21天) + 50% 長期(252天) |
| **進場** | 動能排名前 5 名，等權重建倉；廣度偏低時自動收縮至前 1-4 名 |
| **出場** | 波動率分層停損 → MA200 → 極端 -35%（低波動: -15%/-25%，中波動: -16%/-24%，高波動: -25%/-35%） |
| **金字塔加碼** | 持倉動能 > 0 且批次 < 5 時可加碼；差異停損越晚越緊（回測 Calmar 0.386） |
| **汰弱留強** | 持倉動能 vs 候選動能差距 >10% 且持有 >30 天，建議 ROTATE |
| **趨勢狀態** | ↗️轉強（V轉）/ ↘️轉弱（倒V）/ →盤整，回測月差 +2.14% |
| **市場廣度** | S&P500 % 股票 > MA50，5 級分類；廣度危險時建議只執行最強 1-2 支 |
| **市場環境** | VIX + WTI 油價 → 4 種體制標籤（滯脹恐慌/恐慌下跌/健康風險偏好/平靜牛市） |
| **ML% 評分** | XGBoost 預測每個 ADD 候選「打敗 SPY 的機率」（AUROC 0.726），>50% 模型看好 |
| **候選池** | S&P 500 全部成分股（505 支）+ 白名單 + 現有持倉 |

---

## 安裝

### 1. 建立 conda 環境（推薦）

```bash
# 建立環境（Python 3.12，環境名稱 qt_env）
conda create -n qt_env python=3.12 -y

# 啟動環境
conda activate qt_env

# 安裝所有依賴
pip install -r requirements.txt
```

> **注意**：每次開啟新的 terminal 都需要先 `conda activate qt_env`，
> 或者直接用 `conda run -n qt_env python premarket_main.py`。

### 2. 或使用 pip + pyenv / 系統 Python

```bash
# 需要 Python 3.12
pip install -r requirements.txt
```

### 3. 設定 API Key 與 Email

```bash
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
python confirm_main.py 2026-02-19
```

---

## 透過 Claude Code Agent 使用（推薦）

安裝完成後，在專案目錄開啟 Claude Code，可以直接用 slash commands 操作，不需要記任何腳本名稱。

### 第一步：告訴 Agent 你的持倉

用自然語言描述你目前的持倉，Agent 會幫你建立 `data/portfolio.json`：

```
我目前持有：
- VOO  20 股，成本均價 $520，核心持倉不賣
- NVDA 10 股，成本均價 $135，2025-06-01 買入
- 現金 $5,000

請幫我建立初始持倉
```

Agent 會自動呼叫 `python premarket_main.py --init` 並根據你的描述填入資料。

> 或者直接跑 `python premarket_main.py --init` 進入互動式建立流程。

### 每日工作流程（使用 slash commands）

```
/premarket      → 盤前分析，查看今日 EXIT/ADD 建議，可直接追問
/confirm DATE   → 盤後確認哪些有執行，更新持倉
/exit-check     → 快速確認有無停損觸發（不跑完整分析）
/perf           → 績效三路比較（B&H vs 實際 vs System-Perfect vs SPY）
```

---

## 功能列表

| 功能 | 說明 |
|------|------|
| **盤前建議** | 載入持倉 → 混合動能排名 → 四層出場 → 趨勢狀態 → ML% → 輸出 actions |
| **混合動能** | 50% 短期(21天) + 50% 長期(252天)，兼顧反應速度和穩定性 |
| **四層出場** | 波動率分層停損 → MA200 → 極端 -35%（低: -15%/-25%，中: -16%/-24%，高: -25%/-35%） |
| **金字塔加碼** | 持倉動能持續正 → 最多 5 批加碼，差異停損（越晚越緊），回測 Calmar 0.386 |
| **追蹤停損顯示** | HOLD 欄顯示距高回落 %：🔴 > -20%（接近觸發）、🟡 > -10% |
| **停損單更新提醒** | 追蹤停損 > 固定停損時提醒調整 Firstrade 掛單；多批次取最嚴格（最高）價格合併 |
| **今日待辦清單** | 報告末尾框線摘要：🔴 EXIT（立即執行）→ 📌 停損單調整 → 🔄 ROTATE → 🟢 ADD |
| **趨勢狀態** | 偵測 V 轉回升、倒 V 見頂，補充動能指標的盲點 |
| **汰弱留強** | 自動建議 ROTATE：賣出弱勢持倉，換入強勢候選；confirm 支援部分執行 |
| **市場廣度** | S&P500 廣度（% > MA50）5 級分類；廣度危險時自動縮減 ADD 建議數量 |
| **市場環境** | VIX + WTI 油價組合 → 4 種體制標籤，滯脹恐慌時能源族群 ML% 系統性偏高 |
| **ML% 評分** | XGBoost 對每個 ADD 候選計算「打敗 SPY 機率」，附 SHAP 前 3 驅動因素說明 |
| **板塊監控** | 追蹤科技/軟體/半導體/金融/能源/醫療 vs SPY 相對強弱，板塊走弱時警告 |
| **3Y Alpha** | ADD/ROTATE 候選同時顯示 1Y 和 3Y 超額報酬，協助判斷結構性衰退 vs 景氣循環低點 |
| **RSI 警告** | 🔴 RSI > 80 極度超買、🟡 RSI > 75 超買（只警告不過濾） |
| **郵件通知** | 每日盤前自動發送 HTML 摘要（本文）+ 完整 PDF（附件） |
| **年度 P&L** | 建立年度快照，追蹤年度績效 |

---

## 每日工作流程

```
盤前（美股開盤前）
├── python premarket_main.py
├── 查看板塊健康狀態
├── 查看 EXIT / HOLD / ADD / ROTATE 建議
├── 查看趨勢狀態（↗️轉強 / ↘️轉弱 / →盤整）
├── 查看「📌 停損單需更新」提醒（追蹤停損已超過固定停損時）
└── 查看「今日待辦清單」（報告末尾框線，EXIT 標記立即執行）

盤中
├── 執行 EXIT（停損已觸發，當日必須執行）
├── 調整 Firstrade 停損掛單價格（若有提醒）
└── 視判斷執行 ADD / ROTATE

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

# 含台股掃描
python premarket_main.py --tw

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
python confirm_main.py 2026-02-19
```

### 回測工具
```bash
# 投組層級回測（6 種策略 + SPY B&H 比較，10Y S&P500 全市場）
python portfolio_backtest.py

# 追蹤停損門檻敏感度掃描（-10% 至 -40%）
python _trailing_sensitivity.py

# 停損策略比較（單股）
python stop_loss_compare.py NVDA SHOP TSLA GOOG MU

# 趨勢狀態指標回測（V轉/倒V 的預測力驗證）
python trend_state_backtest.py

# 動能回看週期比較（21d vs 63d vs 126d vs 252d + 混合策略）
python momentum_period_backtest.py
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
| `portfolio.json` | 你的持倉狀態（股數、成本、tranches 批次紀錄、交易紀錄） |
| `watchlist.json` | 白名單標的 |
| `actions_YYYYMMDD.json` | 每日盤前建議（含 status: pending/confirmed/skipped） |
| `snapshot_YYYY.json` | 年度快照（用於計算年度 P&L） |
| `_protection_bt_prices.pkl` | S&P500 歷史價格快取（ML 訓練資料，換裝置需手動搬移） |
| `_ml_model.pkl` | 訓練好的 XGBoost 模型（自動生成，不存在時重訓） |
| `_ml_features.pkl` | ML 特徵矩陣快取（自動生成） |
| `_ml_sector_etf_prices.pkl` / `_ml_vix.pkl` / `_ml_oil.pkl` | ML 輔助快取（自動下載） |
| `sector_map.json` | 板塊分類快取（GICS） |
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
      "core": true,
      "tranches": [
        {"n": 1, "shares": 20, "entry_price": 500.0, "entry_date": "2025-01-01", "stop_type": "standard"}
      ]
    },
    "NVDA": {
      "shares": 15,
      "avg_price": 135.0,
      "cost_basis": 2025,
      "first_entry": "2025-06-01",
      "core": false,
      "favorite": true,
      "tranches": [
        {"n": 1, "shares": 10, "entry_price": 130.0, "entry_date": "2025-06-01", "stop_type": "standard"},
        {"n": 2, "shares": 5,  "entry_price": 145.0, "entry_date": "2025-07-01", "stop_type": "tight_2"}
      ]
    }
  },
  "transactions": [
    {"date": "2025-01-01", "symbol": "VOO", "action": "ADD", "shares": 20, "price": 500.0}
  ]
}
```

---

## 約束條件

- **核心持倉**：`core: true`（如 VOO）永遠只會 HOLD，不會建議賣出
- **偏愛標的**：`favorite: true`（如 TSLA, NVDA）不參與 ROTATE 換股
- **個股上限**：最多 30 檔（不含 core）
- **四層出場**：Fixed -15% → 追蹤停損（高點 -25%）→ MA200 → 極端 -35%
- **掃描範圍**：完整的 S&P 500 + 白名單

---

## 盤前報告範例

```
============================================================
  盤前報告 2026-02-19  |  版本 0.9.0
============================================================
  投組總值: $  126,177.62
  現金:     $      321.00
  個股:     21/30 檔
  2026年度:  $   -793.31 (-0.6%)
============================================================

--- 板塊相對強弱 (過去5日) 🟢 ---
  大盤 SPY: -0.8%
  🟡 科技     -1.1% (vs SPY: -0.3%)
  🟢 半導體    +1.9% (vs SPY: +2.8%)
  🟢 能源     +2.2% (vs SPY: +3.1%)

--- HOLD (繼續持有) ---
  [core] VOO    97 股 @ $631.15  P&L: +47.96%  動能: -0.8%  1Y: +0% 🟢  →盤整
         SNDK   6 股 @ $600.40  P&L: +1.65%  動能: +45.2%  1Y: +1192% 🟢  →盤整
         MU     8 股 @ $420.95  P&L: +3.32%  動能: +16.0%  1Y: +291% 🟢  ↗️轉強(反彈+58%)
         TSLA   12 股 @ $411.32  P&L: +32.53%  動能: -6.0%  1Y: +1% 🟢  ↘️轉弱(距高-16%)

--- ADD / TOPUP 建議 ---
  [#9]  CIEN   建議 2 (ROTATE後 17 股) @ $310.96  動能: +27.8%  1Y vs SPY: +244% 🟢
         原因: 動能排名 #9（+27.8%）
  [增持] SNDK  +5 股 @ $600.40  動能: +45.2%(#3)  倉位 0.6%→3.3%  🟢 安全  1Y: +1192%

--- ROTATE (汰弱留強) ---
  賣 CVS    80 股 (動能: -0.3%, P&L: +52.8%)
  → 買 CIEN   17 股 (動能: +27.8%, 1Y: +244% 🟢)
       動能差: +28%  汰弱留強：動能差 +28%（持有 414 天）
```
