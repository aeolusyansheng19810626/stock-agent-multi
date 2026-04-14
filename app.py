import os
import glob
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools import get_stock_data, search_web, get_stock_history, send_email_report, search_documents

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

system_prompt = f"""你是一个专业的股票分析师，拥有10年股市投资经验。

重要规则：
- 如果用户询问具体财务数据、营收、利润、季报、年报等，先调用 search_documents，有结果则优先使用，没有结果再用 search_web 搜索
- 涉及任何新闻、近期动态、最新消息、近期走势时，必须先调用 search_web 工具
- 禁止用训练数据回答新闻类问题，训练数据已过时
- 搜索关键词用英文，回答用中文

分析时必须包含：
1. 基本面分析（营收、利润、估值）
2. 技术面分析（趋势、支撑位、压力位）
3. 行业对比和竞争格局
4. 近期重要新闻和催化剂
5. 明确的投资建议和目标价位
6. 风险提示

回答要详细深入，不少于500字。

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
def process_uploaded_pdfs(files):
    import tempfile
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from tools import get_embeddings, VECTORSTORE_DIR

    new_files = [f for f in files if f.name not in st.session_state.processed_docs]
    if not new_files:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    with st.spinner("正在处理文档，首次加载模型可能需要 1-2 分钟…"):
        embeddings = get_embeddings()
        vectorstore = None
        if os.path.exists(VECTORSTORE_DIR) and any(os.scandir(VECTORSTORE_DIR)):
            from langchain_community.vectorstores import Chroma as _C
            vectorstore = _C(
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
                st.session_state.processed_docs[file.name] = len(chunks)
            finally:
                os.unlink(tmp_path)

# ====== 初始化 LLM 和工具（按模型缓存） ======
@st.cache_resource
def init_agents():
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI

    tools = [get_stock_data, search_web, get_stock_history, send_email_report, search_documents]
    tools_map = {t.name: t for t in tools}

    groq_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY),
        temperature=0.1,
    )

    groq_with_tools   = groq_llm.bind_tools(tools)
    gemini_with_tools = gemini_llm.bind_tools(tools)
    return groq_with_tools, gemini_with_tools, tools_map

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
    st.session_state.processed_docs = {}  # filename -> chunk_count

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

# ====== 欢迎屏（空状态） ======
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-top">
            <span class="welcome-emoji">💹</span>
            <div class="welcome-title">你好，我是 AI 股票分析师</div>
            <div class="welcome-desc">点击下方卡片或直接输入问题开始分析</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    example_cards = [
        ("🔍", "分析一下英伟达 NVDA 的基本面和技术面", "个股分析"),
        ("⚖️", "帮我对比苹果 AAPL 和微软 MSFT 的走势",  "对比分析"),
        ("📰", "特斯拉最近有什么重要新闻和催化剂？",      "新闻资讯"),
        ("📧", "分析完英伟达后发报告到我的邮箱",          "邮件报告"),
    ]
    col1, col2 = st.columns(2)
    for i, (icon, question, tag) in enumerate(example_cards):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown('<div class="example-card-btn">', unsafe_allow_html=True)
            if st.button(f"{icon}  {question}\n\n🏷 {tag}", key=f"example_card_{i}", use_container_width=True):
                st.session_state.pending_input = question
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

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
    groq_with_tools, gemini_with_tools, tools_map = init_agents()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    charts_before = set(glob.glob("charts/*.png"))

    # ====== Agent 执行循环 ======
    # Groq 负责决策（调用工具），Gemini 负责生成最终报告
    step = 1

    while True:
        messages = [SystemMessage(content=system_prompt)] + st.session_state.chat_history

        # 用 Groq 判断是否还需要调用工具
        groq_response = groq_with_tools.invoke(messages)

        if groq_response.tool_calls:
            st.session_state.chat_history.append(groq_response)

            for tool_call in groq_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                st.markdown(f"""
                <div class="tool-call-block">
                    <span class="tc-step">STEP {step}</span>
                    <span class="tc-name">{tool_name}</span>
                    <span class="tc-args">{tool_args}</span>
                    <span class="tc-status">⏳ 运行中</span>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "tool",
                    "step": step,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "content": "",
                })

                result = tools_map[tool_name].invoke(tool_args)
                step += 1

                st.session_state.chat_history.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
        else:
            # 工具调用完毕，交给 Gemini 生成最终报告；限速时重试，每日配额耗尽时降级到 Groq
            import time
            from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
            messages = [SystemMessage(content=system_prompt)] + st.session_state.chat_history
            final_response = None

            final_model = None
            if st.session_state.dev_mode or st.session_state.gemini_exhausted:
                # 开发模式或日配额耗尽，直接走 Groq
                fallback_response = groq_with_tools.invoke(messages)
                final_response = extract_text(fallback_response.content)
                final_model = "Groq"
            else:
                for attempt in range(2):  # 最多重试1次
                    try:
                        final_response_obj = gemini_with_tools.invoke(messages)
                        final_response = extract_text(final_response_obj.content)
                        final_model = "Gemini"
                        break
                    except ChatGoogleGenerativeAIError as e:
                        err = str(e)
                        if "429" not in err and "RESOURCE_EXHAUSTED" not in err:
                            raise
                        if attempt == 0:
                            # 第一次 429：等待 65 秒后重试（应对 RPM 限速）
                            with st.spinner("Gemini 触发速率限制，65 秒后自动重试…"):
                                time.sleep(65)
                            continue
                        else:
                            # 重试后仍然 429：标记日配额耗尽，降级到 Groq
                            st.session_state.gemini_exhausted = True
                            st.warning("Gemini 日配额已用完，已切换至 Groq 模式（下午 4 点后点击侧边栏按钮恢复）")
                            fallback_response = groq_with_tools.invoke(messages)
                            final_response = extract_text(fallback_response.content)
                            final_model = "Groq"
                            break
            # 兜底：Gemini 返回空内容时降级到 Groq
            if not final_response or not final_response.strip():
                fallback_response = groq_with_tools.invoke(messages)
                final_response = extract_text(fallback_response.content)
                final_model = "Groq"

            st.session_state.chat_history.append(AIMessage(content=final_response))
            with st.chat_message("assistant"):
                st.markdown(final_response)
                st.caption(f"⚠️ 以上内容仅供参考，不构成投资建议。　　由 {final_model} 生成")
            st.session_state.messages.append({"role": "assistant", "content": final_response, "model": final_model})
            break

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
