import streamlit as st
import yfinance as yf
from datetime import datetime


def _fetch_prices(tickers: list[str]) -> dict:
    """批量拉取股价，返回 {ticker: info_dict}"""
    if not tickers:
        return {}
    try:
        data = yf.Tickers(" ".join(tickers))
        result = {}
        for t in tickers:
            try:
                info = data.tickers[t].fast_info
                price = info.last_price
                prev  = info.previous_close
                if price is None or prev is None:
                    result[t] = None
                    continue
                change     = price - prev
                change_pct = change / prev * 100
                result[t] = {
                    "price":      price,
                    "change":     change,
                    "change_pct": change_pct,
                }
            except Exception:
                result[t] = None
        return result
    except Exception:
        return {}


@st.fragment(run_every=30)
def render_stock_ticker():
    """侧边栏实时股价卡片，每30秒局部刷新"""

    # ── 用户输入股票列表 ──
    raw = st.text_input(
        "📌 自选股票（逗号分隔）",
        value=st.session_state.get("ticker_input", ""),
        placeholder="AAPL, TSLA, 7203.T",
        key="ticker_raw_input",
    )
    st.session_state["ticker_input"] = raw

    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    if not tickers:
        st.markdown(
            '<div style="font-size:0.75rem;color:#9CA3AF;padding:4px 0;">输入股票代码后显示实时价格</div>',
            unsafe_allow_html=True,
        )
        return

    prices = _fetch_prices(tickers)

    for ticker, data in prices.items():
        if data is None:
            st.markdown(
                f'<div style="font-size:0.78rem;color:#9CA3AF;padding:2px 0;">{ticker} · 数据获取失败</div>',
                unsafe_allow_html=True,
            )
            continue

        price      = data["price"]
        change     = data["change"]
        change_pct = data["change_pct"]
        is_up      = change >= 0
        color      = "#00C087" if is_up else "#F5475B"
        arrow      = "▲" if is_up else "▼"
        sign       = "+" if is_up else ""
        currency   = "¥" if ticker.endswith(".T") else "$"

        st.markdown(f"""
<div style="
    background:#fff;
    border:1px solid #E5E7EB;
    border-left:3px solid {color};
    border-radius:6px;
    padding:8px 12px;
    margin:4px 0;
">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.78rem;font-weight:700;color:#1D2129;">{ticker}</span>
        <span style="font-size:0.88rem;font-weight:700;color:{color};">
            {arrow} {sign}{change_pct:.2f}%
        </span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">
        <span style="font-size:0.95rem;font-weight:800;color:#1D2129;">{currency}{price:.2f}</span>
        <span style="font-size:0.72rem;color:{color};">{sign}{change:.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    updated = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f'<div style="font-size:0.65rem;color:#9CA3AF;text-align:right;margin-top:2px;">更新于 {updated}</div>',
        unsafe_allow_html=True,
    )
