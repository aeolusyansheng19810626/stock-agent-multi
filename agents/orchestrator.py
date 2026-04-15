"""Orchestrator: 解析用户意图，并行调用 sub-agents，汇总结果交给 report_agent"""
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agents import data_agent, news_agent, rag_agent, report_agent

# ──────────────────────────────────────────
#  Planning prompt
# ──────────────────────────────────────────
PLAN_SYSTEM = """你是一个股票分析任务调度器。分析用户问题，输出一个 JSON 调度计划。

严格输出纯 JSON，不要 markdown 代码块，不要任何解释文字。

格式：
{
  "agents": ["data"],
  "data_params": {"tickers": ["AAPL"], "need_history": false, "periods": ["6mo"]},
  "news_params": {"query": "Apple latest news 2025"},
  "rag_params": {"query": "Apple revenue earnings Q4"},
  "email_params": null
}

agents 字段从以下选择（可多选）：data / news / rag / email
  data  → 查询股价、涨跌、估值、走势图
  news  → 查询新闻、近期动态、催化剂、市场消息
  rag   → 查询财报、营收、利润、EPS、毛利率、季报、年报
  email → 用户明确要求发送邮件报告

路由规则（严格遵守）：
  - 只问股价/走势  → agents: ["data"]
  - 只问新闻      → agents: ["news"]
  - 只问财报      → agents: ["rag"]
  - 综合分析      → agents: ["data", "news", "rag"]
  - 需要走势图    → need_history: true
  - 不需要某 agent → 对应 params 置 null

字段要求：
  - tickers: 标准大写股票代码（AAPL/TSLA/NVDA/MSFT 等），从用户问题中提取
  - news/rag query: 英文关键词，包含公司名和具体主题
  - email_params: 用户明确要求发邮件时填写 {"to": "邮箱地址", "subject": "主题"}，否则 null
  - 用户未提供邮箱时 email_params 为 null
"""


def _parse_plan(text: str) -> dict:
    """从 LLM 输出中提取 JSON，兼容 markdown 代码块包裹的情况"""
    text = text.strip()
    # 去除 ```json ... ``` 包裹
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：在文本中查找第一个 JSON 对象
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def run(
    user_query: str,
    chat_history: list,
    dev_mode: bool = False,
    gemini_exhausted: bool = False,
    status_callback=None,
    groq_api_key: str = "",
    gemini_api_key: str = "",
) -> dict:
    """
    主入口：规划 → 并行执行 sub-agents → 生成报告 → （可选）发送邮件

    Returns:
        {
            "tool_calls":       [{"tool_name": str, "tool_args": dict}, ...],
            "final_response":   str,
            "final_model":      str,
            "gemini_exhausted": bool,
        }
    """
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    def _update(msg: str):
        if status_callback:
            status_callback(msg)

    # ── Step 1: 规划 ──────────────────────────────
    _update("正在分析问题，制定调度计划…")

    groq = ChatGroq(
        api_key=groq_api_key,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    # 取最近 2 轮对话作为规划上下文
    history_ctx = ""
    if chat_history:
        for msg in chat_history[-4:]:
            cls = msg.__class__.__name__
            if cls in ("HumanMessage", "AIMessage"):
                role = "用户" if cls == "HumanMessage" else "助手"
                snippet = str(msg.content)[:200]
                history_ctx += f"{role}: {snippet}\n"

    plan_messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(
            content=(
                f"{'对话历史：\n' + history_ctx if history_ctx else ''}"
                f"用户问题：{user_query}"
            )
        ),
    ]

    plan_resp = groq.invoke(plan_messages)
    plan_text = plan_resp.content if isinstance(plan_resp.content, str) else str(plan_resp.content)

    try:
        plan = _parse_plan(plan_text)
    except Exception:
        # 解析失败时的兜底计划
        plan = {
            "agents": ["news"],
            "data_params": None,
            "news_params": {"query": user_query},
            "rag_params": None,
            "email_params": None,
        }

    agents_to_run = plan.get("agents", [])

    # ── Step 2: 并行执行 sub-agents ───────────────
    parallel_agents = [a for a in agents_to_run if a in ("data", "news", "rag")]
    if parallel_agents:
        _update(f"并行调用：{' · '.join(parallel_agents)} agent…")

    all_tool_calls = []
    gathered_parts = []
    futures: dict = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        dp = plan.get("data_params")
        if "data" in agents_to_run and dp and dp.get("tickers"):
            futures["data"] = executor.submit(
                data_agent.run,
                tickers=dp["tickers"],
                need_history=dp.get("need_history", False),
                periods=dp.get("periods", ["6mo"] * len(dp["tickers"])),
            )

        np_ = plan.get("news_params")
        if "news" in agents_to_run and np_ and np_.get("query"):
            futures["news"] = executor.submit(news_agent.run, query=np_["query"])

        rp = plan.get("rag_params")
        if "rag" in agents_to_run and rp and rp.get("query"):
            futures["rag"] = executor.submit(rag_agent.run, query=rp["query"])

        # 等待所有并行任务完成，按完成顺序收集结果
        for future in as_completed(futures.values()):
            pass  # 确保所有 future 完成后再统一读取

        for agent_name, future in futures.items():
            try:
                result = future.result()
                all_tool_calls.extend(result.get("tool_calls", []))
                content = result.get("content", "")
                if content:
                    gathered_parts.append(content)
            except Exception as e:
                gathered_parts.append(f"[{agent_name} agent 执行失败: {e}]")

    # ── Step 3: 生成报告 ──────────────────────────
    gathered_data = "\n\n".join(gathered_parts) if gathered_parts else "（未获取到相关数据）"

    model_label = "Groq" if (dev_mode or gemini_exhausted) else "Gemini"
    _update(f"正在生成分析报告（{model_label}）…")

    # 构建对话历史摘要（供 report_agent 保持多轮上下文）
    history_text = ""
    if len(chat_history) > 2:
        for msg in chat_history[-4:]:
            cls = msg.__class__.__name__
            if cls in ("HumanMessage", "AIMessage"):
                role = "用户" if cls == "HumanMessage" else "助手"
                history_text += f"{role}: {str(msg.content)[:300]}\n"

    report_result = report_agent.run(
        user_query=user_query,
        gathered_data=gathered_data,
        chat_history_text=history_text,
        dev_mode=dev_mode,
        gemini_exhausted=gemini_exhausted,
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
    )

    # ── Step 4: 发送邮件（可选） ───────────────────
    ep = plan.get("email_params")
    if "email" in agents_to_run and ep and ep.get("to"):
        _update("正在发送邮件报告…")
        from tools import send_email_report
        email_to = ep["to"]
        email_subject = ep.get("subject", "AI 股票分析报告")
        email_res = send_email_report.invoke({
            "to": email_to,
            "subject": email_subject,
            "body": report_result["response"],
        })
        all_tool_calls.append({
            "tool_name": "send_email_report",
            "tool_args": {"to": email_to, "subject": email_subject},
        })

    return {
        "tool_calls": all_tool_calls,
        "final_response": report_result["response"],
        "final_model": report_result["model"],
        "gemini_exhausted": report_result.get("gemini_exhausted", False),
    }
