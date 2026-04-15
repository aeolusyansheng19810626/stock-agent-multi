"""Data Agent: 负责获取股票实时数据和历史走势图"""
from tools import get_stock_data, get_stock_history


def run(tickers: list, need_history: bool = False, periods: list = None) -> dict:
    """
    Args:
        tickers: 股票代码列表，如 ["AAPL", "TSLA"]
        need_history: 是否需要历史走势图
        periods: 每只股票对应的历史周期，默认 6mo

    Returns:
        {"tool_calls": [...], "content": str}
    """
    if periods is None:
        periods = ["6mo"] * len(tickers)

    tool_calls = []
    results = []

    for ticker in tickers:
        result = get_stock_data.invoke({"ticker": ticker})
        tool_calls.append({"tool_name": "get_stock_data", "tool_args": {"ticker": ticker}})
        results.append(f"[实时数据 · {ticker}]\n{result}")

    if need_history:
        for i, ticker in enumerate(tickers):
            period = periods[i] if i < len(periods) else "6mo"
            result = get_stock_history.invoke({"ticker": ticker, "period": period})
            tool_calls.append({
                "tool_name": "get_stock_history",
                "tool_args": {"ticker": ticker, "period": period},
            })
            results.append(f"[历史走势 · {ticker} · {period}]\n{result}")

    return {
        "tool_calls": tool_calls,
        "content": "\n\n".join(results),
    }
