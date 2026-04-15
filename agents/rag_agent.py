"""RAG Agent: 负责从已上传财报 PDF 中检索相关内容"""
from tools import search_documents


def run(query: str) -> dict:
    """
    Args:
        query: 检索关键词（建议英文）

    Returns:
        {"tool_calls": [...], "content": str}
    """
    result = search_documents.invoke({"query": query})
    return {
        "tool_calls": [{"tool_name": "search_documents", "tool_args": {"query": query}}],
        "content": f"[财报文档检索结果]\n{result}",
    }
