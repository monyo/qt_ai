執行績效三路比較（B&H vs Actual vs System-Perfect vs SPY）。

步驟：
1. 執行：`conda run -n qt_env python _perf_review.py`
2. 執行：`conda run -n qt_env python _system_perf_sim.py`
3. 用繁體中文摘要：
   - 三路策略各自 YTD 損益與 vs SPY 差距
   - 主動操作（Actual）vs 不動（B&H）的貢獻
   - 人為偏離系統建議造成的損益影響
   - 一句話結論：目前操作有沒有打敗大盤
