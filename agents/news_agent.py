"""News Agent: 负责网络新闻搜索"""
from tools import search_web


def run(query: str) -> dict:
    """
    Args:
        query: 搜索关键词（建议英文）

    Returns:
        {"tool_calls": [...], "content": str}
    """
    result = search_web.invoke({"query": query})
    return {
        "tool_calls": [{"tool_name": "search_web", "tool_args": {"query": query}}],
        "content": f"[网络搜索结果]\n{result}",
    }
