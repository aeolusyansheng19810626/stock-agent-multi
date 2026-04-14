# StockAI 股票分析 Agent

基于 LangChain 的智能股票分析助手，支持 Streamlit Web UI 和命令行两种使用方式。

---

## 环境搭建

```bash
# 创建虚拟环境
python -m venv venv

# 激活（Windows）
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
pip install langchain-groq langchain-google-genai streamlit
```

### 启动 Web UI
```bash
venv\Scripts\streamlit.exe run app.py
```

### 启动命令行版
```bash
python main.py
```

---

## 项目结构

```
stock-agent/
├── app.py              # Streamlit Web UI（主程序）
├── main.py             # 命令行版本
├── tools.py            # 工具定义（不要修改）
├── components/
│   └── stock_ticker.py # 实时股价侧边栏组件
├── skills/             # 工具使用说明（注入 system prompt）
│   ├── skill_get_stock_data.md
│   ├── skill_get_stock_history.md
│   ├── skill_search_web.md
│   └── skill_send_email.md
├── charts/             # 走势图输出目录（运行时自动创建）
├── vectorstore/        # ChromaDB 向量库（运行时自动创建，不上传 git）
├── .env                # API Keys（不上传 git，参考 .env.example）
├── .env.example        # API Keys 模板
├── token.pickle        # Gmail OAuth 凭证（不上传 git）
└── requirements.txt
```

---

## API Keys

所有 key 存放在项目根目录的 `.env` 文件中（不上传 git），通过 `python-dotenv` 读取。

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

换机器时，参考 `.env.example` 创建 `.env` 并填入对应的 key。代码中通过 `os.getenv()` 读取，不会硬编码在源文件里。

---

## 双模型架构

```
用户提问
   ↓
Groq（工具调用循环）
   ├─ 调用 get_stock_data
   ├─ 调用 search_web
   ├─ 调用 get_stock_history
   └─ 调用 send_email_report
   ↓
Gemini（生成最终分析报告）
   ↓（429 限速）
等待 65 秒重试
   ↓（重试仍 429）
Groq 降级兜底
```

- **Groq**：负责多步工具调用，无日配额限制
- **Gemini 2.5 Flash**：负责生成最终高质量报告
- **Gemini 返回空内容**：自动降级到 Groq

---

## Gemini 免费层限制

| 限制 | 额度 |
|------|------|
| RPM（每分钟请求数） | 5 |
| RPD（每日请求数） | 20 |
| 配额重置时间 | 北京时间 15:00 / 日本时间 16:00（夏令时） |

### 限速处理逻辑
1. 触发 429 → 等待 65 秒自动重试（应对 RPM 打满）
2. 重试仍然 429 → 标记 `gemini_exhausted = True`，后续请求直接走 Groq
3. 刷新页面 → 状态重置，恢复使用 Gemini

---

## 工具说明

### `get_stock_data(ticker)`
- 数据源：yfinance
- 返回：当前价格、涨跌幅、52周高低点、市盈率、成交量

### `search_web(query)`
- 数据源：Tavily API
- 搜索关键词必须用英文，每次返回 3 条结果
- 自动附加当天日期提升结果时效性

### `get_stock_history(ticker, period)`
- period 可选：`1mo` / `3mo` / `6mo` / `1y` / `2y`
- 走势图保存到 `charts/` 目录，Streamlit 页面自动显示

### `search_documents(query)`
- 从已上传的财报 PDF 中检索相关内容
- 向量库持久化到 `./vectorstore`，重启后不丢失
- 嵌入模型：`paraphrase-multilingual-MiniLM-L12-v2`（支持中文，本地运行）
- 首次上传 PDF 时会自动下载模型（约 120MB）

### `send_email_report(to, subject, body)`
- 通过 Gmail API 发送
- 首次使用需要 OAuth 授权，会生成 `token.pickle`
- `token.pickle` 不要删除，否则需要重新授权

---

## Gmail OAuth 初始化

首次使用邮件功能时需要授权：
1. 运行程序后触发邮件发送
2. 浏览器弹出 Google 授权页面
3. 授权完成后生成 `token.pickle`，后续自动使用

换机器时需要把 `token.pickle` 一起复制过去，或重新授权。

---

## 注意事项

- `tools.py` 不要随意修改，工具签名变更会影响 LangChain 工具绑定
- `skills/*.md` 是注入 system prompt 的工具说明，修改后立即生效
- `charts/` 目录需要提前创建，否则走势图保存会报错
- Pydantic 警告（Python 3.14 兼容性）不影响运行，忽略即可
