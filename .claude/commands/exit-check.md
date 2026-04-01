快速檢查目前持倉有無觸發停損條件，不跑完整盤前分析。

步驟：
1. 執行：`conda run -n qt_env python premarket_main.py --exit-only`
   （若無此參數，改執行完整 premarket_main.py 但只看 EXIT 區塊）
2. 列出所有觸發或接近觸發的停損：
   - 已觸發 Fixed Stop（-15%）
   - 已觸發 Trailing Stop（距高點 -25%）
   - 接近追蹤停損（距高點 -20% 以上標示 🔴）
   - 有 🔒 動態收緊停損的批次
3. 如有緊急出場建議，明確標示優先順序
