"""動能策略模組

計算個股動能分數，用於排名和篩選候選標的。
"""
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_rsi(df, period: int = 14) -> float | None:
    """計算 RSI 指標

    Args:
        df: 含有 Close 欄位的 DataFrame
        period: RSI 週期（預設 14）

    Returns:
        RSI 值 (0-100)，失敗回傳 None
    """
    try:
        if len(df) < period + 1:
            return None

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi.iloc[-1], 1)
    except Exception:
        return None


def calculate_momentum_with_rsi(symbol: str, period: int = 21) -> dict | None:
    """計算單一標的的混合動能分數和 RSI

    混合動能 = 50% × 短期(21天) + 50% × 長期(252天)
    回測驗證：混合50/50 累積報酬 +153%（純21天 +111%），勝率 82.6%（純21天 73.9%）

    Args:
        symbol: 股票代碼
        period: 短期動能回看天數（預設21天≈1個月）

    Returns:
        {"momentum": float, "momentum_short": float, "momentum_long": float, "rsi": float}
        momentum 為混合分數，失敗回傳 None
    """
    LONG_PERIOD = 252
    SHORT_WEIGHT = 0.5
    LONG_WEIGHT = 0.5

    try:
        # 取得足夠的數據來計算長期動能(252天) + RSI(14天)
        # 注意：period="1y" 實測只回傳約251個交易日，比 LONG_PERIOD+1=253 少，
        # 會導致長期動能永遠算不出來、悄悄退化成純短期動能（2026-08-25 發現的 bug）。
        # 改用 "2y"（約500個交易日）確保長期動能有足夠資料。
        df = yf.Ticker(symbol).history(period="2y")
        if "Close" not in df:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < max(period, 20):
            return None

        # 計算短期動能
        df_short = df.tail(period + 1)
        momentum_short = (df_short['Close'].iloc[-1] / df_short['Close'].iloc[0] - 1) * 100

        # 計算長期動能（數據足夠時混合，不足時純用短期）
        if len(df) >= LONG_PERIOD + 1:
            df_long = df.tail(LONG_PERIOD + 1)
            momentum_long = (df_long['Close'].iloc[-1] / df_long['Close'].iloc[0] - 1) * 100
            momentum = SHORT_WEIGHT * momentum_short + LONG_WEIGHT * momentum_long
        else:
            momentum_long = None
            momentum = momentum_short

        # 計算 RSI
        rsi = calculate_rsi(df, 14)

        # 資料品質檢查：單日跳動 ≤-45% 或 ≥+90% 通常是分割未調整等資料錯誤
        # （KLAC 案例：動能 +1233%、1Y +2650%，實為錯價）
        daily_ret = df['Close'].pct_change()
        suspect = bool(daily_ret.min() <= -0.45 or daily_ret.max() >= 0.9 or momentum > 500)

        return {
            "momentum": round(momentum, 2),
            "momentum_short": round(momentum_short, 2),
            "momentum_long": round(momentum_long, 2) if momentum_long is not None else None,
            "rsi": rsi,
            "suspect": suspect,
        }
    except Exception:
        return None


def calculate_momentum_as_of(symbol: str, months_back: int = 2, period: int = 21) -> float | None:
    """計算「N個月前」當下的混合動能分數，用於檢查標的動能是否比過去更差（真的在退步）。

    重用 calculate_momentum_with_rsi 的定義（50%短期21天 + 50%長期252天），
    只是把資料截到 N個月前 為止再計算。用於 ROTATE 賣出端「自身動能衰退確認」
    （回測見 research/_rotate_decel_backtest.py：2個月回看，whipsaw 事件 4→1，報酬不變）。

    Args:
        symbol: 股票代碼
        months_back: 回看幾個月前（≈21個交易日/月）
        period: 短期動能回看天數，需與當下動能計算一致

    Returns:
        動能分數（%），資料不足（如新股）時回傳 None，呼叫端應 fallback 為不擋
    """
    LONG_PERIOD = 252
    SHORT_WEIGHT = 0.5
    LONG_WEIGHT = 0.5
    TRADING_DAYS_PER_MONTH = 21

    try:
        df = yf.Ticker(symbol).history(period="3y")
        if "Close" not in df:
            return None
        df = df.dropna(subset=["Close"])

        offset = months_back * TRADING_DAYS_PER_MONTH
        cutoff = len(df) - offset
        if cutoff < max(period, 20) + 1:
            return None  # 資料不足以回推 N 個月前（新股常見）
        df_cut = df.iloc[:cutoff]

        df_short = df_cut.tail(period + 1)
        momentum_short = (df_short['Close'].iloc[-1] / df_short['Close'].iloc[0] - 1) * 100

        if len(df_cut) >= LONG_PERIOD + 1:
            df_long = df_cut.tail(LONG_PERIOD + 1)
            momentum_long = (df_long['Close'].iloc[-1] / df_long['Close'].iloc[0] - 1) * 100
            momentum = SHORT_WEIGHT * momentum_short + LONG_WEIGHT * momentum_long
        else:
            momentum = momentum_short

        return round(momentum, 2)
    except Exception:
        return None


def calculate_momentum(symbol: str, period: int = 21) -> float | None:
    """計算單一標的的動能分數（過去N天報酬%）

    Args:
        symbol: 股票代碼
        period: 回看天數（預設21天≈1個月）

    Returns:
        動能分數（報酬%），失敗回傳 None
    """
    try:
        df = yf.Ticker(symbol).history(period=f"{period + 10}d")
        if "Close" not in df:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < period + 1:
            return None

        # 取最近 period 天
        df = df.tail(period + 1)
        momentum = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        return round(momentum, 2)
    except Exception:
        return None


def calculate_momentum_batch(symbols: list, period: int = 21, max_workers: int = 10, include_rsi: bool = False) -> dict:
    """批次計算多檔標的的動能分數（可選 RSI）

    Args:
        symbols: 股票代碼列表
        period: 回看天數
        max_workers: 最大並行數
        include_rsi: 是否同時計算 RSI

    Returns:
        dict: {symbol: momentum_score} 或 {symbol: {"momentum": float, "rsi": float}}
    """
    results = {}

    def fetch_one(sym):
        if include_rsi:
            return sym, calculate_momentum_with_rsi(sym, period)
        else:
            return sym, calculate_momentum(sym, period)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, data = future.result()
            if data is not None:
                results[sym] = data

    return results


def rank_by_momentum(symbols: list, period: int = 21, top_n: int = None, include_rsi: bool = True) -> list:
    """計算動能並排名（含 RSI）

    Args:
        symbols: 股票代碼列表
        period: 回看天數
        top_n: 只回傳前 N 名（None = 全部）
        include_rsi: 是否計算 RSI（預設 True）

    Returns:
        list of dict: [{"symbol": str, "momentum": float, "rsi": float, "rank": int}, ...]
        按動能由高到低排序
    """
    print(f"正在計算 {len(symbols)} 檔標的的動能分數...")
    scores = calculate_momentum_batch(symbols, period, include_rsi=include_rsi)

    if include_rsi:
        # 資料品質防線：排除可疑標的（分割未調整等），避免進入 ADD 候選
        suspects = {s: d for s, d in scores.items() if d.get("suspect")}
        if suspects:
            for s, d in suspects.items():
                print(f"⚠ 資料異常排除：{s}（動能 {d['momentum']:+.1f}%，疑似分割未調整或錯價，不列入排名）")
            scores = {s: d for s, d in scores.items() if not d.get("suspect")}

        # scores = {symbol: {"momentum": float, "momentum_short": float, "momentum_long": float, "rsi": float}}
        ranked = sorted(scores.items(), key=lambda x: x[1]["momentum"], reverse=True)
        results = []
        for i, (symbol, data) in enumerate(ranked):
            results.append({
                "symbol": symbol,
                "momentum": data["momentum"],
                "momentum_short": data.get("momentum_short"),
                "momentum_long": data.get("momentum_long"),
                "rsi": data.get("rsi"),
                "rank": i + 1,
            })
    else:
        # scores = {symbol: momentum}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for i, (symbol, momentum) in enumerate(ranked):
            results.append({
                "symbol": symbol,
                "momentum": momentum,
                "rank": i + 1,
            })

    if top_n:
        results = results[:top_n]

    return results


def get_momentum_leaders(symbols: list, period: int = 21, top_pct: float = 0.2) -> list:
    """取得動能領先者（前 N%）

    Args:
        symbols: 股票代碼列表
        period: 回看天數
        top_pct: 前幾%（0.2 = 前20%）

    Returns:
        list of dict: 動能領先者資訊
    """
    all_ranked = rank_by_momentum(symbols, period)
    top_n = max(1, int(len(all_ranked) * top_pct))
    return all_ranked[:top_n]


def print_momentum_report(symbols: list, period: int = 21, top_n: int = 20):
    """印出動能排名報告"""
    ranked = rank_by_momentum(symbols, period, top_n)

    print(f"\n{'='*50}")
    print(f"  動能排名 (過去 {period} 天)")
    print(f"{'='*50}")
    print(f"  {'排名':>4} {'股票':<6} {'動能':>10}")
    print(f"  {'-'*30}")

    for item in ranked:
        momentum = item['momentum']
        emoji = "🚀" if momentum > 10 else ("📈" if momentum > 0 else "📉")
        print(f"  {item['rank']:>4} {item['symbol']:<6} {momentum:>+9.1f}% {emoji}")

    print()


def calculate_alpha_1y(symbol: str, benchmark: str = "SPY") -> float | None:
    """計算標的過去 1 年相對於大盤的超額報酬

    Args:
        symbol: 股票代碼
        benchmark: 基準指數（預設 SPY）

    Returns:
        超額報酬%（symbol 報酬 - benchmark 報酬），失敗回傳 None
    """
    try:
        sym_df = yf.Ticker(symbol).history(period="1y")
        bench_df = yf.Ticker(benchmark).history(period="1y")

        if "Close" not in sym_df or "Close" not in bench_df:
            return None
        sym_df = sym_df.dropna(subset=["Close"])
        bench_df = bench_df.dropna(subset=["Close"])

        if sym_df.empty or bench_df.empty or len(sym_df) < 200 or len(bench_df) < 200:
            return None

        sym_return = (sym_df['Close'].iloc[-1] / sym_df['Close'].iloc[0] - 1) * 100
        bench_return = (bench_df['Close'].iloc[-1] / bench_df['Close'].iloc[0] - 1) * 100

        return round(sym_return - bench_return, 1)
    except Exception:
        return None


def calculate_alpha_3y(symbol: str, benchmark: str = "SPY") -> float | None:
    """計算標的過去 3 年相對於大盤的超額報酬

    Args:
        symbol: 股票代碼
        benchmark: 基準指數（預設 SPY）

    Returns:
        超額報酬%（symbol 報酬 - benchmark 報酬），失敗回傳 None
    """
    try:
        sym_df = yf.Ticker(symbol).history(period="3y")
        bench_df = yf.Ticker(benchmark).history(period="3y")

        if "Close" not in sym_df or "Close" not in bench_df:
            return None
        sym_df = sym_df.dropna(subset=["Close"])
        bench_df = bench_df.dropna(subset=["Close"])

        if sym_df.empty or bench_df.empty or len(sym_df) < 600 or len(bench_df) < 600:
            return None

        sym_return = (sym_df['Close'].iloc[-1] / sym_df['Close'].iloc[0] - 1) * 100
        bench_return = (bench_df['Close'].iloc[-1] / bench_df['Close'].iloc[0] - 1) * 100

        return round(sym_return - bench_return, 1)
    except Exception:
        return None


def calculate_alpha_batch(symbols: list, benchmark: str = "SPY", max_workers: int = 10) -> dict:
    """批次計算多檔標的的 1 年超額報酬

    Args:
        symbols: 股票代碼列表
        benchmark: 基準指數
        max_workers: 最大並行數

    Returns:
        dict: {symbol: alpha_1y}
    """
    results = {}

    def fetch_one(sym):
        return sym, calculate_alpha_1y(sym, benchmark)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, alpha = future.result()
            if alpha is not None:
                results[sym] = alpha

    return results


def calculate_alpha_3y_batch(symbols: list, benchmark: str = "SPY", max_workers: int = 10) -> dict:
    """批次計算多檔標的的 3 年超額報酬

    Args:
        symbols: 股票代碼列表
        benchmark: 基準指數
        max_workers: 最大並行數

    Returns:
        dict: {symbol: alpha_3y}
    """
    results = {}

    def fetch_one(sym):
        return sym, calculate_alpha_3y(sym, benchmark)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, alpha = future.result()
            if alpha is not None:
                results[sym] = alpha

    return results


def calculate_trend_state(symbol: str) -> dict | None:
    """計算單一標的的趨勢狀態（20日低點反彈% + 距20日高點%）

    回測驗證：bounce=25%/window=20天 gap +1.06% vs 非轉強（2026-04）
    from_high 門檻 -15%（2026-04 更新，舊值 -5%）

    Args:
        symbol: 股票代碼

    Returns:
        dict: {bounce_pct, from_high_pct, state}
        state: "轉強"(↗), "轉弱"(↘), "盤整"(→)
    """
    try:
        df = yf.Ticker(symbol).history(period="3mo")
        if "Close" not in df:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < 20:
            return None

        closes_20d = df['Close'].iloc[-20:]
        current = closes_20d.iloc[-1]
        low_20d = closes_20d.min()
        high_20d = closes_20d.max()

        if low_20d == 0 or high_20d == 0:
            return None

        bounce_pct = (current / low_20d - 1) * 100
        from_high_pct = (current / high_20d - 1) * 100

        if bounce_pct > 25 and from_high_pct > -15:
            state = "轉強"
        elif from_high_pct < -15:
            state = "轉弱"
        else:
            state = "盤整"

        result = {
            "bounce_pct": round(bounce_pct, 1),
            "from_high_pct": round(from_high_pct, 1),
            "state": state,
        }

        # 強彈偵測：轉弱格局中單日漲幅 ≥ +8%（回測：63天回前高率 44% vs 對照組 27%）
        if state == "轉弱" and len(df) >= 2:
            prev_close = float(df['Close'].iloc[-2])
            if prev_close > 0:
                day_ret = current / prev_close - 1
                if day_ret >= 0.08:
                    result["strong_bounce_pct"] = round(day_ret * 100, 1)

        return result
    except Exception:
        return None


# ── A頭（倒V）型態偵測 ──────────────────────────────────────────────────────
# 回測基礎（2019-2026，S&P500 高動能股，21日後中位報酬差距）：
#   有明顯拖累：房地產 -7.20%、工業 -1.18%、能源 -1.09%、通訊 -0.70%
#   無影響或正面：科技 +1.17%、原物料 +4.81%、金融/醫療/消費/公用事業 中性
A_TOP_WINDOW       = 21
A_TOP_SPIKE_PCT    = 0.12   # 窗口起點到峰值 ≥ +12%
A_TOP_REVERSAL_PCT = 0.12   # 當前距峰值 ≤ -12%
A_TOP_PEAK_BUFFER  = 5      # 峰值不能在窗口末尾 5 天內
A_TOP_DRAG_SECTORS = {"房地產", "工業", "能源", "通訊"}


def detect_a_top(prices: pd.Series) -> bool:
    """偵測 A頭（倒V）型態：窗口內出現明顯拉升後已明顯回落"""
    if len(prices) < A_TOP_WINDOW:
        return False
    recent = prices.iloc[-A_TOP_WINDOW:]
    peak_idx = int(recent.values.argmax())
    if peak_idx >= A_TOP_WINDOW - A_TOP_PEAK_BUFFER:
        return False
    peak_price  = float(recent.iloc[peak_idx])
    start_price = float(recent.iloc[0])
    curr_price  = float(recent.iloc[-1])
    spike    = (peak_price - start_price) / start_price
    reversal = (curr_price  - peak_price)  / peak_price
    return spike >= A_TOP_SPIKE_PCT and reversal <= -A_TOP_REVERSAL_PCT


def detect_a_top_batch(symbols: list, max_workers: int = 10) -> dict:
    """批次偵測 A頭型態

    Returns:
        dict: {symbol: {"is_a_top": bool, "peak_price": float | None}}
    """
    def fetch_one(sym):
        try:
            df = yf.Ticker(sym).history(period="2mo")
            if "Close" not in df:
                return sym, {"is_a_top": False, "peak_price": None}
            df = df.dropna(subset=["Close"])
            if len(df) < A_TOP_WINDOW:
                return sym, {"is_a_top": False, "peak_price": None}
            closes = df["Close"]
            is_a = detect_a_top(closes)
            peak_price = float(closes.iloc[-A_TOP_WINDOW:].max()) if is_a else None
            return sym, {"is_a_top": is_a, "peak_price": peak_price}
        except Exception:
            return sym, {"is_a_top": False, "peak_price": None}

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, data = future.result()
            results[sym] = data
    return results


def calculate_trend_state_batch(symbols: list, max_workers: int = 10) -> dict:
    """批次計算多檔標的的趨勢狀態

    Args:
        symbols: 股票代碼列表
        max_workers: 最大並行數

    Returns:
        dict: {symbol: {bounce_pct, from_high_pct, state}}
    """
    results = {}

    def fetch_one(sym):
        return sym, calculate_trend_state(sym)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, data = future.result()
            if data is not None:
                results[sym] = data

    return results


if __name__ == "__main__":
    # 測試
    test_symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'SHOP', 'MU', 'UEC']
    print_momentum_report(test_symbols, period=21, top_n=10)
