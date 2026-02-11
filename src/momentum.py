"""動能策略模組

計算個股動能分數，用於排名和篩選候選標的。
"""
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        if df.empty or len(df) < period:
            return None

        # 取最近 period 天
        df = df.tail(period + 1)
        momentum = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        return round(momentum, 2)
    except Exception:
        return None


def calculate_momentum_batch(symbols: list, period: int = 21, max_workers: int = 10) -> dict:
    """批次計算多檔標的的動能分數

    Args:
        symbols: 股票代碼列表
        period: 回看天數
        max_workers: 最大並行數

    Returns:
        dict: {symbol: momentum_score}
    """
    results = {}

    def fetch_one(sym):
        return sym, calculate_momentum(sym, period)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, score = future.result()
            if score is not None:
                results[sym] = score

    return results


def rank_by_momentum(symbols: list, period: int = 21, top_n: int = None) -> list:
    """計算動能並排名

    Args:
        symbols: 股票代碼列表
        period: 回看天數
        top_n: 只回傳前 N 名（None = 全部）

    Returns:
        list of dict: [{"symbol": str, "momentum": float, "rank": int}, ...]
        按動能由高到低排序
    """
    print(f"正在計算 {len(symbols)} 檔標的的動能分數...")
    scores = calculate_momentum_batch(symbols, period)

    # 排序
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


if __name__ == "__main__":
    # 測試
    test_symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'SHOP', 'MU', 'UEC']
    print_momentum_report(test_symbols, period=21, top_n=10)
