執行盤前分析系統，產出今日 ADD/EXIT/HOLD 建議。

步驟：
1. 執行：`conda run -n qt_env python premarket_main.py`
2. 執行完後，讀取 `data/actions_YYYYMMDD.json`（今天日期）
3. 用繁體中文摘要：
   - 今日市場體制（BULL/BEAR）
   - EXIT 建議：哪些要出場、原因
   - ADD 建議：哪些要買、動能排名、alpha
   - 有無 🔄強彈 或 ⚠️ 倒V警告值得特別注意
4. 等待使用者追問（例如「為什麼推 XXX？」「SNDK 今天怎樣？」）
