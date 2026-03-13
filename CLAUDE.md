# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python 3.12 + conda 環境 `qt_env`**（首次或新機器時）

```bash
conda create -n qt_env python=3.12 -y
conda activate qt_env
pip install -r requirements.txt
```

每次執行前確認在 `qt_env` 中（`conda activate qt_env`）。

## Running the Application

```bash
# Install dependencies（已建立環境後）
pip install -r requirements.txt

# Set up API keys in .env file
GEMINI_API_KEY=<your_key>

# === 盤前建議系統（主要使用） ===
# 首次使用：互動建立投資組合
python premarket_main.py --init

# 盤前：產出 actions 建議 → data/actions_YYYYMMDD.json
python premarket_main.py

# 新增白名單標的
python premarket_main.py --watch PLTR COIN

# 盤後：確認執行了哪些 actions → 更新 portfolio.json
python confirm_main.py 2026-01-28

# === 舊工具（仍可用） ===
# 歷史壓力測試
python main.py

# 獨立掃描器
python scanner_main.py

# === 風控工具 ===
# 停損策略回測比較（單股）
python stop_loss_compare.py NVDA SHOP TSLA GOOG MU

# 投組層級回測（6 種策略 + SPY B&H，10Y S&P500 全市場）
python portfolio_backtest.py

# 追蹤停損門檻敏感度掃描（-10% 至 -40%）
python _trailing_sensitivity.py

# 板塊相對強弱獨立檢查
python -c "from src.sector_monitor import print_sector_report; print_sector_report()"
```

There is no test suite, linter, or CI/CD configured.

## Architecture

Quantitative stock scanning + position management system. Combines technical analysis (MA60+RSI) with LLM sentiment analysis.

### Daily workflow

```
盤前: premarket_main.py → actions_YYYYMMDD.json (建議)
盤後: confirm_main.py   → 標記已執行的 actions → 更新 portfolio.json
```

### Entry points

| File | Purpose |
|---|---|
| `premarket_main.py` | 盤前建議主入口。載入持倉 → 報價 → 風控 → 掃描候選 → AI 情緒 → 輸出 actions |
| `confirm_main.py` | 確認入口。讀取 actions 檔 → 逐筆確認/跳過 → 更新持倉 |
| `scanner_main.py` | 獨立掃描器（可被 premarket_main 呼叫） |
| `main.py` | 歷史壓力測試 |

### Core modules in `src/`

| Module | Role |
|---|---|
| `portfolio.py` | 持倉狀態管理。讀寫 `data/portfolio.json`（含 avg_price, cost_basis, transactions, favorite），白名單 `data/watchlist.json` |
| `risk.py` | 風控：Fixed -15% → 追蹤停損（高點 -25%）→ MA200 → 極端 -35%，持倉上限 30 檔 |
| `premarket.py` | 決策引擎。產出 HOLD/EXIT/ADD/ROTATE actions，含 source 和 version 欄位 |
| `data_loader.py` | yfinance 資料取得（含快取）、S&P 500 ticker 列表、批次最新報價 |
| `strategy.py` | 技術訊號：buy when Price > MA60 AND RSI < 70, sell when Price < MA60 OR RSI > 85 |
| `momentum.py` | 混合動能排名（50% 短期21天 + 50% 長期252天）、RSI、1Y/3Y Alpha、趨勢狀態 |
| `backtester.py` | 回測引擎。Signal → Position 狀態機，計算 Return%, MDD%, WinRate% |
| `ai_analyst.py` | 新聞抓取 + Gemini 2.0 Flash 情緒分析 (-1.0 to +1.0) |
| `indicators.py` | SMA, RSI 指標（pandas_ta） |
| `visualizer.py` | 策略 vs 大盤累積報酬圖 |
| `sector_monitor.py` | 板塊相對強弱監控。追蹤 XLK/IGV/SMH vs SPY，板塊跑輸 -5% 時警告 |
| `stop_loss_backtester.py` | 停損策略回測。支援 fixed / trailing stop-loss 比較 |

### Key design details

- **Actions 狀態流**：`pending` → `confirmed`/`skipped`，HOLD 為 `auto`
- **停損機制**（四層，依優先順序）：
  - Fixed -15%：從成本價計算，跌破即出場
  - 追蹤停損 -25%：從進場後最高點（`high_since_entry`）回落 -25% 出場，回測 Calmar 0.882
  - MA200 停損：跌破 200 日均線
  - 極端停損 -35%：最後防線
  - HOLD 欄位顯示距高接近度：🔴 > -20%、🟡 > -10%
- **持倉保護層級**：
  - `core=true`：核心持倉（如 VOO），永遠只產出 HOLD
  - `favorite=true`：偏愛標的（如 TSLA, NVDA），不參與 ROTATE 換股
- **ROTATE 汰弱留強**：
  - 觸發條件：動能差距 >10% 且持有 >30 天
  - 主動建議換股，不限於現金不足時
  - 排除 core 和 favorite 標的
  - confirm 時可輸入實際股數（支援部分賣出/買入），sell/buy 獨立設定
- **ADD / TOPUP 合併顯示**：
  - ADD 清單同時顯示「現金」與「ROTATE 後」兩種股數建議
  - `suggested_shares_post_rotate`：ROTATE 賣出收回現金後，扣除安全 TOPUP 預算，再分配給 ADD 槽位
  - 安全 TOPUP（停損高於成本）以 `[增持]` 標籤合入 ADD 清單，目標補到等權重（total_value/30）
  - 非安全 TOPUP 獨立顯示於下方「風險較高」區塊
- **RSI 警告**：🔴 RSI > 80 極度超買、🟡 RSI > 75 超買（只警告不過濾，讓使用者決定）
- **趨勢狀態**（回測驗證月差 +2.14%）：
  - ↗️ 轉強：40 日低點反彈 >20% 且距高點 <5%（V 轉格局）
  - ↘️ 轉弱：距 40 日高點 >15%（倒 V 格局）
  - → 盤整：其他
  - 特殊警告：動能正 + 轉弱 = ⚠️ 倒V警告、動能負 + 轉強 = 💡 V轉回升中
- **候選池**：S&P 500 前 100 + `data/watchlist.json` 白名單
- **Sizing**：等權重 cash / available_slots；TOPUP 增持目標為 total_value / 30（等權重）
- **報價定義**：前一交易日收盤價（盤前 yfinance 最後一筆 Close）
- Signal 是事件（1/-1/0），backtester 轉為 Position（0/1）狀態機
- AI 情緒在 API 額度用完時降級為中性 (0.0)
- **板塊監控**：盤前報告顯示科技/軟體/半導體 vs SPY 相對強弱，當板塊跑輸 >5% 時警告
- **曝險警告**：當持股科技比例高 + 科技板塊走弱時，會特別提醒

## Portfolio Baseline (2026)

| 日期 | 事件 | 數值 |
|------|------|------|
| 2026-01-01 | 年初基準值 | $126,970.93 |
| 2026-02-13 | Firstrade 實際值 | $125,150 |
| 2026-02-13 | YTD P&L | -$1,821 (-1.43%) |
| 2026-02-26 | Firstrade 實際值 | $128,587 |
| 2026-02-26 | YTD P&L | +$1,616 (+1.3%) |

**注意**：投組總值以 Firstrade 實際數字為準，yfinance 報價可能有微小差異。
