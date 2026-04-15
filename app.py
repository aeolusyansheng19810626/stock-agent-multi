import os
import glob
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools import get_stock_data, search_web, get_stock_history, send_email_report, search_documents
from components.stock_ticker import render_stock_ticker

# ====== API Keys ======
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ====== 加载 Skill 文档 ======
def load_skills(skill_dir: str) -> str:
    skill_docs = []
    for filename in sorted(os.listdir(skill_dir)):
        if filename.endswith(".md"):
            with open(os.path.join(skill_dir, filename), "r", encoding="utf-8") as f:
                skill_docs.append(f.read())
    return "\n\n---\n\n".join(skill_docs)

skill_content = load_skills("skills")

from datetime import datetime as _dt
_today = _dt.now().strftime("%Y年%m月%d日")

system_prompt = f"""你是一个专业的股票分析师，拥有10年股市投资经验。今天是 {_today}。

搜索时必须使用当前年份（{_dt.now().year}年），禁止在查询中写死历史年份如2024。

重要规则（必须严格遵守）：
- 用户询问财报、财务数据、营收、利润、季报、年报、EPS、毛利率等内容时，【必须】首先调用 search_documents，禁止直接调用 search_web 或使用训练数据回答
- search_documents 返回内容为空时，才允许改用 search_web 搜索
- 涉及新闻、近期动态、最新消息、近期走势时，必须先调用 search_web 工具
- 禁止用训练数据回答任何时效性问题，训练数据已过时
- 搜索关键词用英文，回答用中文

工具调用硬性限制（必须遵守）：
- search_web 最多调用 2 次，调用后立即基于结果回答，禁止继续搜索
- 每种工具最多调用 1 次（search_web 例外，最多 2 次）
- 问新闻 → 只调用 1 次 search_web，得到结果后直接回答
- 问股价 → 只调用 get_stock_data
- 问走势图 → 只调用 get_stock_history
- 问财报 → 先调用 search_documents，为空时调用 1 次 search_web
- 只有用户明确要求"完整分析"时，才可以调用多个不同工具

回答直接切题，不强制涵盖所有维度。

以下是你可以使用的工具的详细使用说明，请严格按照说明决定何时调用哪个工具：

{skill_content}
"""

# ====== 提取响应文本（兼容 Groq 字符串 和 Gemini 内容块列表） ======
def extract_text(content) -> str:
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

# ====== 处理上传 PDF，写入向量库 ======
_PROCESSED_JSON = "./vectorstore/processed_files.json"

def _load_processed_registry() -> dict:
    """从磁盘读取已处理文件记录，跨会话持久化"""
    if os.path.exists(_PROCESSED_JSON):
        import json as _json
        with open(_PROCESSED_JSON, "r", encoding="utf-8") as f:
            return _json.load(f)
    return {}

def _save_processed_registry(registry: dict):
    os.makedirs(os.path.dirname(_PROCESSED_JSON), exist_ok=True)
    import json as _json
    with open(_PROCESSED_JSON, "w", encoding="utf-8") as f:
        _json.dump(registry, f, ensure_ascii=False, indent=2)

def process_uploaded_pdfs(files):
    import tempfile
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from tools import get_embeddings, VECTORSTORE_DIR, invalidate_vectorstore

    # 磁盘持久化去重（重启后仍有效）
    registry = _load_processed_registry()
    new_files = [f for f in files if f.name not in registry]
    # 同步到 session_state 展示
    st.session_state.processed_docs = registry
    if not new_files:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    with st.spinner("正在处理文档，首次加载模型可能需要 1-2 分钟…"):
        embeddings = get_embeddings()
        vectorstore = None
        if os.path.exists(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR)):
            vectorstore = Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=embeddings,
                collection_name="stockai_docs",
            )

        for file in new_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                for chunk in chunks:
                    chunk.metadata["source"] = file.name
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        chunks, embeddings,
                        persist_directory=VECTORSTORE_DIR,
                        collection_name="stockai_docs",
                    )
                else:
                    vectorstore.add_documents(chunks)
                registry[file.name] = len(chunks)
                st.session_state.processed_docs = dict(registry)
            finally:
                os.unlink(tmp_path)

        _save_processed_registry(registry)
        invalidate_vectorstore()  # 让 search_documents 重新加载最新向量库

# ====== 初始化 LLM 和工具（按模型缓存） ======
@st.cache_resource
def init_agents():
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI

    tools = [get_stock_data, search_web, get_stock_history, send_email_report, search_documents]
    tools_map = {t.name: t for t in tools}

    groq_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY),
        # model="openai/gpt-oss-120b",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY),
        temperature=0.1,
    )

    groq_with_tools   = groq_llm.bind_tools(tools)
    gemini_with_tools = gemini_llm.bind_tools(tools)
    return groq_llm, groq_with_tools, gemini_with_tools, tools_map

# ====== 页面配置 ======
st.set_page_config(
    page_title="AI股票分析 · moomoo风格",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== 全局 CSS（moomoo 风格） ======
st.markdown("""
<style>
/* ─────────────────────────────────────────
   RESET & 全局变量
───────────────────────────────────────── */
:root {
    --orange:    #FF6B35;
    --orange-lt: #FFF3EE;
    --orange-md: #FFD4BF;
    --green:     #00C087;
    --green-lt:  #E6F9F4;
    --red:       #F5475B;
    --red-lt:    #FEF0F2;
    --bg:        #F5F6FA;
    --card:      #FFFFFF;
    --border:    #E5E7EB;
    --text-1:    #1D2129;
    --text-2:    #6B7280;
    --text-3:    #9CA3AF;
    --shadow-sm: 0 1px 4px rgba(0,0,0,0.06);
    --shadow-md: 0 2px 12px rgba(0,0,0,0.08);
    --radius:    10px;
}

/* 隐藏 Streamlit 默认 header/footer */
header[data-testid="stHeader"],
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* 强制侧边栏始终展开（覆盖 localStorage 的折叠状态） */
section[data-testid="stSidebar"] {
    transform: translateX(0) !important;
    min-width: 244px !important;
    width: 244px !important;
    visibility: visible !important;
}

/* 隐藏侧边栏折叠/展开按钮（固定导航，不允许收起） */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }

/* ─────────────────────────────────────────
   APP 背景
───────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text-1) !important;
    font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', sans-serif !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    max-width: 960px !important;
}

/* ─────────────────────────────────────────
   侧边栏 — 导航面板
───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* 侧边栏所有文字 */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: var(--text-2) !important;
}

/* 模型选择下拉框 */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text-1) !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.12) !important;
}

/* 清空按钮 */
[data-testid="stSidebar"] .stButton > button {
    background: var(--orange) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 8px rgba(255,107,53,0.25) !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.88 !important;
}

/* ─────────────────────────────────────────
   聊天消息 — 基础重置
───────────────────────────────────────── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
    gap: 10px !important;
}

/* 头像圆圈 */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: transparent !important;
    border: none !important;
    width: 36px !important;
    height: 36px !important;
    font-size: 1.2rem !important;
    flex-shrink: 0 !important;
}

/* ─── 用户气泡 ─── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
    background: var(--orange-lt) !important;
    border: 1px solid var(--orange-md) !important;
    border-right: 3px solid var(--orange) !important;
    border-radius: 12px 4px 12px 12px !important;
    color: var(--text-1) !important;
    padding: 11px 16px !important;
    box-shadow: 0 1px 6px rgba(255,107,53,0.09) !important;
    font-size: 0.92rem !important;
}

/* ─── AI气泡 ─── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--green) !important;
    border-radius: 4px 12px 12px 12px !important;
    color: var(--text-1) !important;
    padding: 14px 18px !important;
    box-shadow: var(--shadow-md) !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

/* 消息内文字 */
.stChatMessage p { color: var(--text-1) !important; margin: 0 0 6px 0 !important; }
.stChatMessage li { color: var(--text-1) !important; }
.stChatMessage strong { color: var(--text-1) !important; font-weight: 700 !important; }
.stChatMessage h1,.stChatMessage h2,.stChatMessage h3 {
    color: var(--text-1) !important;
    font-weight: 700 !important;
    margin-top: 12px !important;
}
.stChatMessage code {
    background: #F3F4F6 !important;
    color: var(--orange) !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.82em !important;
    border: 1px solid var(--border) !important;
}
.stChatMessage pre {
    background: #F8F9FA !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* caption 免责声明 */
.stChatMessage .stCaption p {
    color: var(--text-3) !important;
    font-size: 0.72rem !important;
    border-top: 1px solid var(--border) !important;
    padding-top: 8px !important;
    margin-top: 10px !important;
}

/* ─────────────────────────────────────────
   聊天输入框
───────────────────────────────────────── */
[data-testid="stBottom"] {
    background: linear-gradient(0deg, var(--bg) 75%, transparent) !important;
    padding-top: 12px !important;
}

[data-testid="stChatInput"] {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-md) !important;
    transition: border-color 0.15s !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.12) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-1) !important;
    font-size: 0.92rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-3) !important;
}

[data-testid="stChatInput"] button {
    color: var(--orange) !important;
}

/* ─────────────────────────────────────────
   滚动条
───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ─────────────────────────────────────────
   自定义组件
───────────────────────────────────────── */

/* === 顶部 Header === */
.moo-header {
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 0 4px;
    height: 54px;
    margin: 0 -4rem 0 -4rem;
    box-shadow: var(--shadow-sm);
    position: sticky;
    top: 0;
    z-index: 99;
}

.moo-logo {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 20px;
    border-right: 1px solid var(--border);
    height: 100%;
}

.moo-logo .logo-icon {
    font-size: 1.4rem;
    line-height: 1;
}

.moo-logo .logo-text {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--orange) !important;
    letter-spacing: -0.3px;
}

.moo-logo .logo-sub {
    font-size: 0.65rem;
    color: var(--text-3) !important;
    font-weight: 500;
    letter-spacing: 0.5px;
    display: block;
    margin-top: -2px;
}

.moo-nav {
    display: flex;
    align-items: center;
    height: 100%;
    padding: 0 8px;
    gap: 2px;
}

.moo-nav-item {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    height: 100%;
    font-size: 0.83rem;
    color: var(--text-2) !important;
    cursor: default;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
    white-space: nowrap;
}

.moo-nav-item.active {
    color: var(--orange) !important;
    border-bottom-color: var(--orange);
    font-weight: 600;
}

.moo-nav-item:hover { color: var(--text-1) !important; }

.moo-header-right {
    margin-left: auto;
    padding-right: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.moo-badge-live {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--green-lt);
    color: var(--green) !important;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

.moo-badge-live::before {
    content: '';
    width: 6px;
    height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 1.6s infinite;
}

.moo-badge-model {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #EEF2FF;
    color: #4F46E5 !important;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    border: 1px solid #C7D2FE;
    letter-spacing: 0.3px;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* === 侧边栏内部 === */
.sidebar-logo {
    padding: 18px 16px 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

.sidebar-logo .brand {
    font-size: 1rem;
    font-weight: 800;
    color: var(--orange) !important;
}

.sidebar-logo .brand-sub {
    font-size: 0.7rem;
    color: var(--text-3) !important;
    margin-top: 1px;
}

.sidebar-section {
    padding: 4px 12px 10px;
}

.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3) !important;
    padding: 10px 4px 6px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 10px;
    border-radius: 8px;
    margin: 2px 0;
    cursor: default;
    transition: background 0.12s;
}

.nav-item:hover { background: var(--bg); }

.nav-item.active {
    background: var(--orange-lt);
}

.nav-item.active .nav-label {
    color: var(--orange) !important;
    font-weight: 600;
}

.nav-icon { font-size: 1.05rem; width: 20px; text-align: center; flex-shrink: 0; }
.nav-label { font-size: 0.85rem; color: var(--text-2) !important; }

.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 10px 12px;
}

.sidebar-disclaimer {
    margin: 10px 12px 4px;
    padding: 9px 11px;
    background: #FFF8F0;
    border: 1px solid #FFE0CC;
    border-radius: 7px;
    font-size: 0.72rem;
    color: #B45309 !important;
    line-height: 1.5;
}

/* === 工具调用 — Badge 风格 === */
.tool-call-block {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid #6366F1;
    border-radius: 4px 8px 8px 4px;
    margin: 5px 0;
    box-shadow: var(--shadow-sm);
}

.tc-step {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #fff !important;
    background: #6366F1;
    padding: 2px 7px;
    border-radius: 20px;
    flex-shrink: 0;
}

.tc-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: #4F46E5 !important;
    font-family: 'Courier New', monospace;
    background: #EEF2FF;
    padding: 2px 9px;
    border-radius: 5px;
    border: 1px solid #C7D2FE;
    flex-shrink: 0;
}

.tc-args {
    font-size: 0.75rem;
    color: var(--text-2) !important;
    font-family: 'Courier New', monospace;
    background: var(--bg);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    word-break: break-all;
    flex: 1;
    min-width: 0;
}

.tc-status {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--green) !important;
    background: var(--green-lt);
    padding: 2px 8px;
    border-radius: 20px;
    flex-shrink: 0;
}

/* === 走势图卡片 === */
.chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--orange);
    border-radius: 4px 10px 10px 10px;
    overflow: hidden;
    margin: 12px 0 4px;
    box-shadow: var(--shadow-md);
}

.chart-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background: #FAFAFA;
}

.chart-card-icon { font-size: 1rem; }

.chart-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-1) !important;
}

.chart-card-tag {
    margin-left: auto;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--orange) !important;
    background: var(--orange-lt);
    border: 1px solid var(--orange-md);
    padding: 2px 8px;
    border-radius: 20px;
}

.chart-card-body { padding: 12px 16px 16px; }

/* === 欢迎屏（空状态） === */
.welcome-screen {
    padding: 48px 0 24px;
}

.welcome-top {
    text-align: center;
    margin-bottom: 32px;
}

.welcome-emoji {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 12px;
}

.welcome-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-1) !important;
    margin-bottom: 6px;
}

.welcome-desc {
    font-size: 0.88rem;
    color: var(--text-2) !important;
}

.example-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    max-width: 600px;
    margin: 0 auto;
}

/* 分隔线 */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 12px 0 !important; }

/* === 快捷卡片按钮 === */
.example-card-btn > div[data-testid="stButton"] > button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    text-align: left !important;
    height: auto !important;
    min-height: 88px !important;
    white-space: pre-line !important;
    color: var(--text-1) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.15s, border-color 0.15s, transform 0.1s !important;
    font-size: 0.85rem !important;
    line-height: 1.55 !important;
    width: 100% !important;
}

.example-card-btn > div[data-testid="stButton"] > button:hover {
    box-shadow: var(--shadow-md) !important;
    border-color: var(--orange-md) !important;
    border-left: 3px solid var(--orange) !important;
    transform: translateY(-1px) !important;
    color: var(--text-1) !important;
    background: var(--orange-lt) !important;
}

.example-card-btn > div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    box-shadow: var(--shadow-sm) !important;
}
</style>
""", unsafe_allow_html=True)

# ====== Session State 初始化 ======
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "gemini_exhausted" not in st.session_state:
    st.session_state.gemini_exhausted = False
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = _load_processed_registry()

# ====== 侧边栏 ======
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="brand">📈 StockAI</div>
        <div class="brand-sub">智能股票分析助手</div>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">导航</div>
        <div class="nav-item active">
            <span class="nav-icon">💬</span>
            <span class="nav-label">AI 对话分析</span>
        </div>
        <div class="nav-item">
            <span class="nav-icon">📊</span>
            <span class="nav-label">股价查询</span>
        </div>
        <div class="nav-item">
            <span class="nav-icon">📉</span>
            <span class="nav-label">历史走势</span>
        </div>
        <div class="nav-item">
            <span class="nav-icon">📰</span>
            <span class="nav-label">市场资讯</span>
        </div>
        <div class="nav-item">
            <span class="nav-icon">📧</span>
            <span class="nav-label">报告邮件</span>
        </div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # 操作区
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">操作</div></div>',
                unsafe_allow_html=True)

    dev_mode = st.toggle("🛠️ 开发模式（仅 Groq）", value=st.session_state.dev_mode)
    if dev_mode != st.session_state.dev_mode:
        st.session_state.dev_mode = dev_mode
        st.rerun()

    if st.button("🗑️ 清空对话记录", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if not st.session_state.dev_mode:
        gemini_label = "🔴 Gemini 已耗尽（点击恢复）" if st.session_state.gemini_exhausted else "✅ Gemini 正常"
        if st.button(gemini_label, use_container_width=True, disabled=not st.session_state.gemini_exhausted):
            st.session_state.gemini_exhausted = False
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">📡 实时股价</div></div>',
                unsafe_allow_html=True)
    render_stock_ticker()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">📄 财报文档</div></div>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "上传 PDF 财报",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )
    if uploaded_files:
        process_uploaded_pdfs(uploaded_files)

    if st.session_state.processed_docs:
        for fname, cnt in st.session_state.processed_docs.items():
            st.markdown(
                f'<div style="font-size:0.75rem;color:#6B7280;padding:2px 4px;">'
                f'✅ {fname} <span style="color:#9CA3AF">({cnt} 片段)</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="font-size:0.75rem;color:#9CA3AF;padding:2px 4px;">暂无文档，上传后可查询财报数据</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="sidebar-disclaimer">
        ⚠️ 内容仅供参考，不构成投资建议。投资有风险，入市需谨慎。
    </div>
    """, unsafe_allow_html=True)

# ====== 顶部 Header ======
st.markdown("""
<div class="moo-header">
    <div class="moo-logo">
        <span class="logo-icon">📈</span>
        <div>
            <span class="logo-text">StockAI</span>
            <span class="logo-sub">AI FINANCIAL ANALYSIS</span>
        </div>
    </div>
    <nav class="moo-nav">
        <div class="moo-nav-item active">💬 AI分析</div>
        <div class="moo-nav-item">📊 行情</div>
        <div class="moo-nav-item">📰 资讯</div>
        <div class="moo-nav-item">📁 自选股</div>
    </nav>
    <div class="moo-header-right">
        <span class="moo-badge-model">⚡ Groq + Gemini</span>
        <span class="moo-badge-live">实时数据</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ====== 欢迎屏（仅空状态时显示标题） ======
if not st.session_state.messages:
    if st.session_state.pending_input:
        st.markdown("""
        <div class="welcome-screen">
            <div class="welcome-top">
                <span class="welcome-emoji">⏳</span>
                <div class="welcome-title">AI 正在分析中，请稍候…</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="welcome-screen">
            <div class="welcome-top">
                <span class="welcome-emoji">💹</span>
                <div class="welcome-title">你好，我是 AI 股票分析师</div>
                <div class="welcome-desc">点击下方卡片或直接输入问题开始分析</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ====== 渲染历史消息 ======
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            model_label = f"　　由 {msg['model']} 生成" if msg.get("model") else ""
            st.caption(f"⚠️ 以上内容仅供参考，不构成投资建议。{model_label}")
    elif role == "tool":
        st.markdown(f"""
        <div class="tool-call-block">
            <span class="tc-step">STEP {msg["step"]}</span>
            <span class="tc-name">{msg["tool_name"]}</span>
            <span class="tc-args">{msg["tool_args"]}</span>
            <span class="tc-status">✓ 完成</span>
        </div>
        """, unsafe_allow_html=True)
    elif role == "chart":
        ticker = msg.get("caption", "走势图").replace(" 历史走势图", "")
        st.markdown(f"""
        <div class="chart-card">
            <div class="chart-card-header">
                <span class="chart-card-icon">📊</span>
                <span class="chart-card-title">{ticker} · 历史走势图</span>
                <span class="chart-card-tag">Price Chart</span>
            </div>
            <div class="chart-card-body">
        """, unsafe_allow_html=True)
        st.image(msg["content"], use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ====== 快捷卡片（始终显示，处理中除外） ======
example_cards = [
    ("🔍", "分析一下英伟达 NVDA 的基本面和技术面", "个股分析"),
    ("⚖️", "帮我对比苹果 AAPL 和微软 MSFT 的走势",  "对比分析"),
    ("📰", "特斯拉最近有什么重要新闻和催化剂？",      "新闻资讯"),
    ("📧", "分析完英伟达后发报告到我的邮箱",          "邮件报告"),
]
if not st.session_state.pending_input:
    if st.session_state.messages:
        st.markdown("<div style='margin-top:12px;margin-bottom:4px;color:#9CA3AF;font-size:13px;'>快捷提问</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, (icon, question, tag) in enumerate(example_cards):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown('<div class="example-card-btn">', unsafe_allow_html=True)
            if st.button(f"{icon}  {question}\n\n🏷 {tag}", key=f"example_card_{i}", use_container_width=True):
                st.session_state.pending_input = question
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ====== 处理用户输入（卡片点击 或 手动输入） ======
user_input = None

# 优先检测卡片点击写入的预填问题
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

# 手动输入（始终渲染输入框）
typed = st.chat_input("请输入你的问题，例如：分析一下英伟达的股票...")
if typed:
    user_input = typed

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    charts_before = set(glob.glob("charts/*.png"))

    try:
        # ====== 多 Agent 执行 ======
        from agents.orchestrator import run as orchestrator_run

        TOOL_LABEL = {
            "get_stock_data":    "获取股票实时数据",
            "search_web":        "搜索网络新闻",
            "get_stock_history": "获取历史走势图",
            "search_documents":  "检索财报文档",
            "send_email_report": "发送邮件报告",
        }

        with st.status("AI 正在分析…", expanded=True) as status_box:
            result = orchestrator_run(
                user_query=user_input,
                chat_history=st.session_state.chat_history,
                dev_mode=st.session_state.dev_mode,
                gemini_exhausted=st.session_state.gemini_exhausted,
                status_callback=lambda msg: status_box.update(label=msg),
            )

            # Gemini 日配额耗尽时更新状态
            if result.get("gemini_exhausted"):
                st.session_state.gemini_exhausted = True
                st.warning("Gemini 日配额已用完，已切换至 Groq 模式（下午 4 点后点击侧边栏按钮恢复）")

            # 显示各工具调用步骤
            for i, tc in enumerate(result["tool_calls"], 1):
                tool_label = TOOL_LABEL.get(tc["tool_name"], tc["tool_name"])
                st.write(f"🔧 **Step {i}** · `{tc['tool_name']}` — {tc['tool_args']}")
                st.session_state.messages.append({
                    "role": "tool",
                    "step": i,
                    "tool_name": tc["tool_name"],
                    "tool_args": tc["tool_args"],
                    "content": "",
                })

            final_response = result["final_response"]
            final_model    = result["final_model"]

            st.session_state.chat_history.append(AIMessage(content=final_response))
            with st.chat_message("assistant"):
                st.markdown(final_response)
                st.caption(f"⚠️ 以上内容仅供参考，不构成投资建议。　　由 {final_model} 生成")
            st.session_state.messages.append({"role": "assistant", "content": final_response, "model": final_model})

            status_box.update(label="✅ 分析完成", state="complete", expanded=False)

        # ====== 显示本次新生成的走势图 ======
        charts_after = set(glob.glob("charts/*.png"))
        new_charts = sorted(charts_after - charts_before)
        for chart_path in new_charts:
            ticker = os.path.basename(chart_path).split("_")[0]
            caption = f"{ticker} 历史走势图"
            st.markdown(f"""
            <div class="chart-card">
                <div class="chart-card-header">
                    <span class="chart-card-icon">📊</span>
                    <span class="chart-card-title">{ticker} · 历史走势图</span>
                    <span class="chart-card-tag">Price Chart</span>
                </div>
                <div class="chart-card-body">
            """, unsafe_allow_html=True)
            st.image(chart_path, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "chart",
                "content": chart_path,
                "caption": caption,
            })

        st.rerun()  # 刷新页面：清除"AI正在分析中"，显示正常聊天界面

    except Exception as e:
        st.error(f"❌ 处理出错：{e}")
        import traceback
        st.code(traceback.format_exc(), language="text")
