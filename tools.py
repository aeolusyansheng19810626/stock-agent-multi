import json
import pickle
import os
import yfinance as yf
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from datetime import datetime
from tavily import TavilyClient
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

VECTORSTORE_DIR = "./vectorstore"
_embeddings = None
_vectorstore = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embeddings

def get_vectorstore():
    """返回缓存的 Chroma 实例，向量库有更新时调用此函数重新加载"""
    global _vectorstore
    if _vectorstore is None and os.path.exists(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR)):
        from langchain_community.vectorstores import Chroma
        _vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=get_embeddings(),
            collection_name="stockai_docs",
        )
    return _vectorstore

def invalidate_vectorstore():
    """上传新文档后调用，让下次检索重新加载"""
    global _vectorstore
    _vectorstore = None


from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

def get_gmail_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('gmail', 'v1', credentials=creds)

@tool
def get_stock_data(ticker: str) -> str:
    """获取美股实时数据，包括价格、涨跌幅、52周高低点、市盈率。ticker 是股票代码如 AAPL、TSLA。"""
    stock = yf.Ticker(ticker)
    info = stock.info
    data = {
        "ticker": ticker.upper(),
        "name": info.get("longName", "N/A"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
        "change_pct": info.get("regularMarketChangePercent", 0),
        "week52_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "week52_low": info.get("fiftyTwoWeekLow", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "volume": info.get("regularMarketVolume", "N/A"),
    }
    return json.dumps(data)

@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息，遇到不确定的概念、名词、新闻时使用。"""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    enhanced_query = f"{query} {today}"
    result = tavily.search(query=enhanced_query, max_results=3)
    contents = [r["content"] for r in result["results"]]
    return "\n".join(contents)

@tool
def get_stock_history(ticker: str, period: str = "6mo") -> str:
    """获取股票历史价格并画出走势图。ticker 是股票代码，period 是时间范围：1mo/3mo/6mo/1y/2y。"""
    os.makedirs("charts", exist_ok=True)
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return f"无法获取 {ticker} 的历史数据"
    plt.figure(figsize=(12, 5))
    plt.plot(hist.index, hist["Close"], linewidth=2, color="#1f77b4")
    plt.fill_between(hist.index, hist["Close"], alpha=0.1, color="#1f77b4")
    plt.title(f"{ticker} Price Chart ({period})", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"charts/{ticker}_{timestamp}_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 走势图已保存：{ticker}_chart.png")
    summary = {
        "ticker": ticker,
        "period": period,
        "start_price": round(hist["Close"].iloc[0], 2),
        "end_price": round(hist["Close"].iloc[-1], 2),
        "highest": round(hist["Close"].max(), 2),
        "lowest": round(hist["Close"].min(), 2),
        "change_pct": round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 2)
    }
    return json.dumps(summary)

@tool
def search_documents(query: str) -> str:
    """【优先调用】从已上传的财报PDF中检索相关内容。
    用户询问财报、营收、利润、EPS、毛利率、季报、年报等财务数据时，必须首先调用此工具，不得跳过。
    返回内容为空时才允许改用 search_web。
    如果没有上传任何文档，会返回提示信息。"""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "暂无上传文档，请先在侧边栏上传财报 PDF。"
    try:
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "文档库中未找到相关内容。"
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知文件")
            results.append(f"[片段 {i+1} · 来源: {source}]\n{doc.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"文档检索失败：{e}"

@tool
def send_email_report(to: str, subject: str, body: str) -> str:
    """发送股票分析报告到指定邮箱。to 是收件人邮箱，subject 是标题，body 是正文。"""
    import base64
    from email.mime.text import MIMEText
    try:
        service = get_gmail_service()
        message = MIMEText(body, 'plain', 'utf-8')
        message['to'] = to
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        return f"✅ 邮件已成功发送到 {to}"
    except Exception as e:
        return f"❌ 发送失败：{e}"
