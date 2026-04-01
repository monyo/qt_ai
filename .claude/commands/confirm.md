確認今日已執行的交易，更新 portfolio.json。

使用方式：`/confirm YYYY-MM-DD`（例如 `/confirm 2026-04-01`）

步驟：
1. 執行：`conda run -n qt_env python confirm_main.py $ARGUMENTS`
2. 程式會逐筆列出建議的 actions，引導使用者輸入 confirmed/skipped 及實際成交價與股數
3. 完成後讀取更新後的 `data/portfolio.json`，確認：
   - 現金餘額是否合理
   - 新增持倉的 avg_price / tranches 是否正確
   - 如有異常主動提醒
