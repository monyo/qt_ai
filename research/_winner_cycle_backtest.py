"""
特殊池回測：對已證明自己的大贏家，輪動式賣出/買回 vs 單純持有
============================================================
假設：
  - 任一股票從進場後獲利達到 POOL_THRESHOLD（如 +100%）即進入「特殊池」
  - 進入特殊池後，測試兩種策略：
    A. 標準追蹤停損（-25% from high）→ 持有到底
    B. 輪動策略：從近期高點回落 PULLBACK% → 賣出；
                 之後從低點反彈 RECOVERY% → 買回；
                 最終以相同 -25% 追蹤停損作為最後出場
  - 比較兩策略從「進入特殊池」那天到「最終出場」的累積報酬
"""
import pickle
import numpy as np
import pandas as pd
from itertools import product

# ── 參數 ──────────────────────────────────────────────
POOL_THRESHOLD = 1.0      # 獲利 >100% 才進入特殊池
FINAL_STOP     = 0.25     # 兩策略都用 -25% 追蹤停損作最終出場
PULLBACK_TESTS = [0.08, 0.10, 0.12]   # 賣出門檻
RECOVERY_TESTS = [0.05, 0.07, 0.10]   # 買回門檻
MIN_HOLD_DAYS  = 5        # 賣出後至少等 5 天才能買回

# ── 模擬進場：用 6 個月前股價推算，漲超 100% 的起點 ──
with open('data/_protection_bt_prices.pkl', 'rb') as f:
    prices = pickle.load(f)

prices = prices.dropna(axis=1, thresh=int(len(prices)*0.8))
prices = prices.ffill().bfill()

def simulate_pool(sym_prices, pullback_pct, recovery_pct):
    """
    找出所有「進入特殊池」事件，對每個事件模擬 A vs B 策略。
    回傳 list of (days_held, ret_A, ret_B)
    """
    px = sym_prices.dropna()
    if len(px) < 500:
        return []

    results = []
    used_start = set()

    for i in range(126, len(px) - 60):   # 至少需要 6 個月前資料
        entry_price = px.iloc[i - 126]    # 假設 126 天前進場
        current     = px.iloc[i]
        ret_so_far  = current / entry_price - 1

        if ret_so_far < POOL_THRESHOLD:
            continue
        if i in used_start:
            continue
        used_start.update(range(i, i + 20))   # 同一段不重複計算

        # 從今天開始，模擬到最終追蹤停損
        pool_entry = current
        sub = px.iloc[i:]

        # ─── 策略 A：標準追蹤停損 ─────────────────
        high_A = pool_entry
        ret_A  = None
        for j, p in enumerate(sub):
            high_A = max(high_A, p)
            if p < high_A * (1 - FINAL_STOP):
                ret_A = p / pool_entry - 1
                days_A = j
                break
        if ret_A is None:
            ret_A  = sub.iloc[-1] / pool_entry - 1
            days_A = len(sub) - 1

        # ─── 策略 B：輪動 ──────────────────────────
        in_position = True
        cash_level  = pool_entry   # 記錄最後持有的買入價（相對 pool_entry 的比例）
        cumulative  = 1.0          # 累積倍數
        high_B      = pool_entry
        low_out     = None
        cooldown    = 0
        ret_B       = None

        for j, p in enumerate(sub):
            if in_position:
                high_B = max(high_B, p)
                # 最終追蹤停損
                if p < high_B * (1 - FINAL_STOP):
                    cumulative *= p / cash_level
                    ret_B = cumulative - 1
                    break
                # 輪動賣出
                if p < high_B * (1 - pullback_pct) and cooldown == 0:
                    cumulative *= p / cash_level
                    in_position = False
                    low_out     = p
                    cooldown    = MIN_HOLD_DAYS
            else:
                cooldown = max(0, cooldown - 1)
                low_out  = min(low_out, p)
                if cooldown == 0 and p > low_out * (1 + recovery_pct):
                    cash_level  = p
                    in_position = True
                    high_B      = p   # 重置追蹤高點

        if ret_B is None:
            if in_position:
                cumulative *= sub.iloc[-1] / cash_level
            ret_B = cumulative - 1

        results.append((days_A, ret_A, ret_B))

    return results


# ── 主迴圈 ─────────────────────────────────────────
print("=" * 65)
print("特殊池回測：輪動策略 vs 標準追蹤停損（進場門檻 +100%）")
print(f"資料期間：{prices.index[0].date()} ~ {prices.index[-1].date()}")
print(f"股票數：{prices.shape[1]}")
print("=" * 65)

for pullback, recovery in product(PULLBACK_TESTS, RECOVERY_TESTS):
    all_results = []
    for sym in prices.columns:
        r = simulate_pool(prices[sym], pullback, recovery)
        all_results.extend(r)

    if not all_results:
        continue

    df = pd.DataFrame(all_results, columns=['days', 'ret_A', 'ret_B'])
    n  = len(df)
    avg_A  = df['ret_A'].mean() * 100
    avg_B  = df['ret_B'].mean() * 100
    med_A  = df['ret_A'].median() * 100
    med_B  = df['ret_B'].median() * 100
    win_B  = (df['ret_B'] > df['ret_A']).mean() * 100

    print(f"\n賣出={pullback*100:.0f}%  買回={recovery*100:.0f}%  (n={n})")
    print(f"  策略A（持有）  平均 {avg_A:+.1f}%  中位 {med_A:+.1f}%")
    print(f"  策略B（輪動）  平均 {avg_B:+.1f}%  中位 {med_B:+.1f}%")
    print(f"  輪動勝率：{win_B:.1f}%  差距：{avg_B - avg_A:+.1f}%")

print("\n完成")
