"""板塊相對強弱監控

追蹤科技相關板塊 vs 大盤的相對表現，
當板塊明顯跑輸時發出警告。
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# 監控的板塊 ETF
SECTOR_ETFS = {
    "XLK": "科技",
    "IGV": "軟體",
    "SMH": "半導體",
    "XLF": "金融",
    "XLE": "能源",
    "XLV": "醫療",
}

BENCHMARK = "SPY"

# 警告門檻：相對強弱低於此值時警告
ALERT_THRESHOLD = -0.05  # -5%


def fetch_sector_returns(lookback_days=5):
    """取得板塊和大盤的近期報酬

    Args:
        lookback_days: 回看天數

    Returns:
        dict: {symbol: {"name": str, "return": float, "relative": float}}
    """
    results = {}

    # 取得基準報酬
    benchmark_ret = _get_return(BENCHMARK, lookback_days)
    if benchmark_ret is None:
        return {}

    results[BENCHMARK] = {
        "name": "大盤",
        "return": benchmark_ret,
        "relative": 0.0,
    }

    # 取得各板塊報酬
    for symbol, name in SECTOR_ETFS.items():
        ret = _get_return(symbol, lookback_days)
        if ret is not None:
            results[symbol] = {
                "name": name,
                "return": ret,
                "relative": ret - benchmark_ret,
            }

    return results


def _get_return(symbol, days):
    """取得單一標的的近期報酬"""
    try:
        # 多抓幾天確保有足夠交易日
        df = yf.Ticker(symbol).history(period=f"{days + 10}d")
        closes = df["Close"].dropna() if "Close" in df else pd.Series(dtype=float)
        if len(closes) < days + 1:
            return None

        closes = closes.tail(days + 1)  # +1 因為要算 pct change
        ret = (closes.iloc[-1] / closes.iloc[0] - 1)
        return round(ret, 4)
    except Exception:
        return None


def get_sector_alerts(lookback_days=5, threshold=ALERT_THRESHOLD):
    """取得板塊警告

    Returns:
        alerts: list of dict，每個 dict 含 symbol, name, return, relative, message
    """
    sector_data = fetch_sector_returns(lookback_days)
    alerts = []

    for symbol, data in sector_data.items():
        if symbol == BENCHMARK:
            continue

        if data["relative"] < threshold:
            alerts.append({
                "symbol": symbol,
                "name": data["name"],
                "return": data["return"],
                "relative": data["relative"],
                "message": f"{data['name']}板塊跑輸大盤 {data['relative']*100:.1f}%",
            })

    # 按相對強弱排序（最弱的在前）
    alerts.sort(key=lambda x: x["relative"])

    return alerts


def get_sector_summary(lookback_days=5):
    """取得板塊摘要，用於盤前報告

    Returns:
        summary: dict 含 benchmark, sectors, alerts, status
    """
    sector_data = fetch_sector_returns(lookback_days)
    alerts = get_sector_alerts(lookback_days)

    # 判斷整體狀態
    if not sector_data:
        status = "unknown"
        status_emoji = "❓"
    elif len(alerts) >= 3:
        status = "danger"
        status_emoji = "🔴"
    elif len(alerts) >= 1:
        status = "warning"
        status_emoji = "🟡"
    else:
        status = "healthy"
        status_emoji = "🟢"

    return {
        "lookback_days": lookback_days,
        "benchmark": sector_data.get(BENCHMARK),
        "sectors": {k: v for k, v in sector_data.items() if k != BENCHMARK},
        "alerts": alerts,
        "status": status,
        "status_emoji": status_emoji,
    }


def print_sector_report(lookback_days=5):
    """印出板塊報告"""
    summary = get_sector_summary(lookback_days)

    print(f"\n{'='*50}")
    print(f"  板塊相對強弱  |  過去 {lookback_days} 日  |  {summary['status_emoji']} {summary['status'].upper()}")
    print(f"{'='*50}")

    if summary["benchmark"]:
        print(f"\n  基準 SPY: {summary['benchmark']['return']*100:+.1f}%")

    print(f"\n  {'板塊':<8} {'報酬':>8} {'vs SPY':>10}")
    print(f"  {'-'*30}")

    for symbol, data in summary["sectors"].items():
        rel = data["relative"]
        emoji = "🔴" if rel < -0.05 else ("🟡" if rel < 0 else "🟢")
        print(f"  {emoji} {data['name']:<6} {data['return']*100:>+7.1f}% {rel*100:>+9.1f}%")

    if summary["alerts"]:
        print(f"\n  ⚠️  警告：")
        for alert in summary["alerts"]:
            print(f"     - {alert['message']}")

    print()


def check_holdings_sector_exposure(holdings, lookback_days=5):
    """檢查持股的板塊曝險

    Args:
        holdings: list of symbol

    Returns:
        dict: 含 tech_heavy, alerts 等資訊
    """
    # 簡單分類（可以之後擴展）
    tech_related = {"NVDA", "SHOP", "GOOG", "GOOGL", "TSLA", "MU", "DASH", "ZG",
                    "AAPL", "MSFT", "META", "AMZN", "AMD", "INTC", "CRM", "ADBE"}

    holdings_upper = {s.upper() for s in holdings}
    tech_holdings = holdings_upper & tech_related
    tech_ratio = len(tech_holdings) / len(holdings) if holdings else 0

    # 取得板塊警告
    alerts = get_sector_alerts(lookback_days)
    tech_alerts = [a for a in alerts if a["symbol"] in ("XLK", "IGV", "SMH")]

    return {
        "tech_ratio": tech_ratio,
        "tech_holdings": list(tech_holdings),
        "is_tech_heavy": tech_ratio > 0.5,
        "tech_alerts": tech_alerts,
        "warning": tech_ratio > 0.5 and len(tech_alerts) > 0,
    }


if __name__ == "__main__":
    print_sector_report(5)

    # 測試持股曝險
    holdings = ["NVDA", "SHOP", "UEC", "GOOG", "CVS", "TSLA", "MU", "LLY", "DASH", "ZG"]
    exposure = check_holdings_sector_exposure(holdings)
    print(f"科技股佔比: {exposure['tech_ratio']*100:.0f}%")
    print(f"科技股持倉: {exposure['tech_holdings']}")
    if exposure['warning']:
        print("⚠️  警告：科技股佔比高且板塊走弱！")
