"""年度快照管理

建立和載入年度快照，用於計算年度 P&L。
"""
import json
import os
from datetime import date
import yfinance as yf

SNAPSHOT_DIR = "data"


def get_snapshot_path(year: int) -> str:
    """取得快照檔案路徑"""
    return os.path.join(SNAPSHOT_DIR, f"snapshot_{year}.json")


def load_snapshot(year: int) -> dict | None:
    """載入年度快照

    Returns:
        快照資料，若不存在則回傳 None
    """
    path = get_snapshot_path(year)
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(snapshot: dict, year: int):
    """儲存年度快照"""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = get_snapshot_path(year)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return path


def fetch_historical_prices(symbols: list, target_date: str) -> dict:
    """取得指定日期的收盤價

    Args:
        symbols: 股票代碼列表
        target_date: 目標日期 (YYYY-MM-DD)

    Returns:
        dict: {symbol: price}
    """
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # 取得目標日期前後的資料
            df = ticker.history(start=target_date, period="5d")
            if not df.empty:
                prices[symbol] = round(df['Close'].iloc[0], 2)
        except Exception:
            pass
    return prices


def create_year_start_snapshot(portfolio: dict, year: int) -> dict:
    """建立年初快照

    Args:
        portfolio: 當前持倉資料
        year: 年份

    Returns:
        快照資料
    """
    positions = portfolio.get("positions", {})
    symbols = list(positions.keys())

    # 取得年初第一個交易日的價格 (約 1/2 或 1/3)
    target_date = f"{year}-01-02"
    print(f"正在取得 {year} 年初價格 ({len(symbols)} 檔)...")
    prices = fetch_historical_prices(symbols, target_date)

    # 建立快照
    snapshot_positions = {}
    total_value = portfolio.get("cash", 0)

    for symbol, pos in positions.items():
        price = prices.get(symbol)
        if price is None:
            # 若無法取得年初價格，使用成本價
            price = pos.get("avg_price", 0)
            print(f"  警告: {symbol} 無法取得年初價格，使用成本價 ${price:.2f}")

        shares = pos.get("shares", 0)
        value = round(price * shares, 2)
        total_value += value

        snapshot_positions[symbol] = {
            "shares": shares,
            "price": price,
            "value": value,
        }

    snapshot = {
        "year": year,
        "date": target_date,
        "created_at": str(date.today()),
        "cash": portfolio.get("cash", 0),
        "total_value": round(total_value, 2),
        "positions": snapshot_positions,
    }

    return snapshot


def calculate_yearly_pnl(current_value: float, snapshot: dict) -> dict:
    """計算年度 P&L

    Args:
        current_value: 當前投組總值
        snapshot: 年初快照

    Returns:
        dict: {pnl_amount, pnl_pct}
    """
    if not snapshot:
        return None

    start_value = snapshot.get("total_value", 0)
    if start_value <= 0:
        return None

    pnl_amount = current_value - start_value
    pnl_pct = (pnl_amount / start_value) * 100

    return {
        "start_value": start_value,
        "current_value": current_value,
        "pnl_amount": round(pnl_amount, 2),
        "pnl_pct": round(pnl_pct, 2),
    }


if __name__ == "__main__":
    from portfolio import load_portfolio

    portfolio = load_portfolio()
    snapshot = create_year_start_snapshot(portfolio, 2026)

    print(f"\n快照資料:")
    print(f"  日期: {snapshot['date']}")
    print(f"  總值: ${snapshot['total_value']:,.2f}")
    print(f"  持倉數: {len(snapshot['positions'])}")

    path = save_snapshot(snapshot, 2026)
    print(f"\n已儲存至: {path}")
