"""Report Agent: 汇总所有 sub-agent 的数据，生成最终分析报告"""
import os
import time

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

REPORT_SYSTEM = """你是一个专业的股票分析师，拥有10年股市投资经验。

你将收到由多个数据源汇总而来的上下文（实时股价、历史走势、新闻、财报等），
请基于这些数据为用户的问题生成一份详细的分析报告。

要求：
- 直接切题，基于提供的数据回答，不要编造数据
- 回答深度视问题而定：简单问题直接回答，综合分析需包含基本面、技术面、近期动态、投资建议和风险提示
- 回答用中文，长度不少于300字（综合分析不少于500字）
- 风险提示必须包含
"""


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def run(
    user_query: str,
    gathered_data: str,
    chat_history_text: str = "",
    dev_mode: bool = False,
    gemini_exhausted: bool = False,
    groq_api_key: str = "",
    gemini_api_key: str = "",
) -> dict:
    """
    Args:
        user_query:        用户原始问题
        gathered_data:     各 sub-agent 汇总的数据字符串
        chat_history_text: 近期对话摘要（可选，用于多轮上下文）
        dev_mode:          开发模式，强制使用 Groq
        gemini_exhausted:  Gemini 日配额已耗尽，降级 Groq
        groq_api_key / gemini_api_key: 从外部传入，避免重复读环境变量

    Returns:
        {"response": str, "model": str, "gemini_exhausted": bool}
    """
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    history_section = f"\n对话历史：\n{chat_history_text}\n" if chat_history_text else ""
    prompt = (
        f"用户问题：{user_query}\n"
        f"{history_section}"
        f"\n以下是收集到的数据：\n{gathered_data}\n\n"
        "请基于以上数据生成分析报告："
    )
    messages = [SystemMessage(content=REPORT_SYSTEM), HumanMessage(content=prompt)]

    def _call_groq() -> str:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        resp = llm.invoke(messages)
        return _extract_text(resp.content)

    # ── 开发模式 / Gemini 已耗尽 → 直接走 Groq ──
    if dev_mode or gemini_exhausted:
        return {"response": _call_groq(), "model": "Groq", "gemini_exhausted": False}

    # ── 正常模式：Gemini 优先，429 时重试一次，耗尽则降级 ──
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.1,
    )

    for attempt in range(2):
        try:
            resp = gemini_llm.invoke(messages)
            text = _extract_text(resp.content)
            if text.strip():
                return {"response": text, "model": "Gemini", "gemini_exhausted": False}
            # Gemini 返回空内容 → 兜底 Groq
            return {"response": _call_groq(), "model": "Groq", "gemini_exhausted": False}
        except ChatGoogleGenerativeAIError as e:
            err = str(e)
            if "429" not in err and "RESOURCE_EXHAUSTED" not in err:
                raise
            if attempt == 0:
                # 第一次 429：等 65 秒后重试（应对 RPM 限速）
                time.sleep(65)
                continue
            # 重试后仍 429：日配额耗尽，降级 Groq
            return {"response": _call_groq(), "model": "Groq", "gemini_exhausted": True}

    # 兜底（理论上不会走到这里）
    return {"response": _call_groq(), "model": "Groq", "gemini_exhausted": False}
