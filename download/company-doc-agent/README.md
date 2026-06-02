# 🤖 企业文档智能助手 Agent

> 基于 LangChain + LangGraph 的 ReAct Agent，支持文档问答、员工查询、文档搜索

## 📋 项目简介

本项目是一个面向企业内部的文档智能助手 Agent，能够帮助员工快速获取公司制度、流程规范、员工信息等内容。

### 核心功能

| 功能 | 说明 | 技术实现 |
|------|------|---------|
| 📄 文档问答 | 上传公司文档，AI 自动回答相关问题 | RAG（检索增强生成）+ ChromaDB |
| 👥 员工查询 | 查询员工信息、部门归属 | Tool Calling |
| 🔍 文档搜索 | 语义搜索文档内容 | 向量相似度检索 |
| 💬 多轮对话 | 支持上下文连续对话 | 会话记忆管理 |

### 技术架构

```
用户提问
    ↓
┌──────────────────────────────────────┐
│          LangGraph ReAct Agent       │
│                                      │
│   Think → Act → Observe → Think...  │
│     ↓        ↓                       │
│   LLM    Tool Calling                │
│            ↓                         │
│   ┌─────────────┐  ┌─────────────┐  │
│   │ 文档检索工具 │  │ 员工查询工具 │  │
│   └──────┬──────┘  └──────┬──────┘  │
│          ↓                ↓         │
│   ┌─────────────┐  ┌─────────────┐  │
│   │   ChromaDB  │  │ employees   │  │
│   │  向量数据库  │  │   .json     │  │
│   └─────────────┘  └─────────────┘  │
└──────────────────────────────────────┘
    ↓
用户获得回答
```

### 技术栈

- **Agent 框架**: LangChain 0.3 + LangGraph 0.2（ReAct 模式）
- **LLM**: DeepSeek / OpenAI（OpenAI 兼容接口）
- **向量数据库**: ChromaDB（嵌入式，无需额外服务）
- **后端**: FastAPI（异步高性能）
- **前端**: Gradio（快速搭建聊天界面）
- **部署**: Docker + Docker Compose

---

## 🚀 快速开始

### 前置条件

- Python 3.11+
- DeepSeek 或 OpenAI API Key

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd company-doc-agent
```

### 2. 安装依赖

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

`.env` 文件内容：
```env
# 推荐使用 DeepSeek（便宜好用）
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 或使用 OpenAI
# LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o
```

### 4. 初始化测试数据

```bash
python scripts/seed_data.py
```

这会创建：
- 10 位示例员工信息
- 4 个示例公司文档（休假制度、员工手册、技术部规范、报销流程）

### 5. 启动服务

```bash
python app/main.py
```

启动后访问：
- 🤖 **Gradio 界面**: http://localhost:8000
- 📖 **API 文档**: http://localhost:8000/docs
- ❤️ **健康检查**: http://localhost:8000/health

---

## 📡 API 接口

### 与 Agent 对话

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "公司年假制度是什么？", "session_id": "user1"}'
```

### 上传文档

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@你的文档.pdf"
```

### 搜索文档

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "报销流程", "top_k": 3}'
```

### 列出文档

```bash
curl http://localhost:8000/api/v1/documents
```

---

## 🐳 Docker 部署

### 本地 Docker 运行

```bash
# 构建镜像
docker build -t company-doc-agent .

# 运行容器
docker run -d \
  --name company-doc-agent \
  -p 8000:8000 \
  -e LLM_API_KEY=sk-xxxxxxxxxxxxxxxx \
  -e LLM_BASE_URL=https://api.deepseek.com/v1 \
  -e LLM_MODEL=deepseek-chat \
  company-doc-agent
```

### Docker Compose 运行

```bash
# 设置环境变量
export LLM_API_KEY=sk-xxxxxxxxxxxxxxxx

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

---

## ☁️ 云端部署

### 方案一：Render（推荐，最简单）

1. 将代码推送到 GitHub
2. 登录 [Render](https://render.com/)，创建新的 **Web Service**
3. 连接 GitHub 仓库
4. 配置：
   - **Build Command**: `pip install -r requirements.txt && python scripts/seed_data.py`
   - **Start Command**: `python app/main.py`
   - **Environment Variables**:
     - `LLM_API_KEY` = 你的 API Key
     - `LLM_BASE_URL` = https://api.deepseek.com/v1
     - `LLM_MODEL` = deepseek-chat
5. 点击 **Deploy**，等待部署完成
6. 获得 `https://your-agent.onrender.com` 地址

### 方案二：Railway

1. 登录 [Railway](https://railway.app/)
2. 新建项目 → 从 GitHub 部署
3. 添加环境变量（同上）
4. 自动部署，获得访问地址

### 方案三：阿里云函数计算（国内推荐）

1. 安装 Fun 工具：`npm install -g fun`
2. 创建 `template.yml` 配置文件
3. 运行 `fun deploy` 部署

---

## 📁 项目结构

```
company-doc-agent/
├── app/
│   ├── main.py              # FastAPI 启动入口
│   ├── config.py            # 配置管理
│   ├── gradio_app.py        # Gradio 前端界面
│   ├── agent/
│   │   ├── core.py          # ⭐ Agent 核心（LangGraph ReAct）
│   │   ├── tools.py         # ⭐ 工具定义（搜索/查询/上传）
│   │   └── prompts.py       # 系统提示词
│   ├── api/
│   │   └── routes.py        # REST API 路由
│   ├── rag/
│   │   └── document.py      # ⭐ RAG 文档处理（加载/分块/向量化/检索）
│   └── memory/
│       └── manager.py       # 会话记忆管理
├── data/
│   ├── documents/           # 文档存放目录
│   ├── chroma_db/           # 向量数据库
│   └── employees.json       # 员工信息
├── scripts/
│   └── seed_data.py         # 初始化测试数据
├── Dockerfile               # Docker 构建文件
├── docker-compose.yml       # Docker Compose 配置
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
└── README.md                # 本文档
```

---

## 🔑 核心代码解析

### 1. ReAct Agent 工作流（app/agent/core.py）

```
用户输入 → LLM 思考(Think) → 需要工具？
                                ├─ 是 → 执行工具(Act) → 观察结果(Observe) → 回到思考
                                └─ 否 → 输出回答 → 结束
```

这是 LangGraph 的核心：用状态图（StateGraph）定义 Agent 的决策流程，支持循环和条件分支。

### 2. RAG 检索流程（app/rag/document.py）

```
上传文档 → 加载(PDF/TXT/DOCX) → 分块(500字/块) → 向量化(Embedding) → 存入 ChromaDB
用户提问 → 问题向量化 → ChromaDB 相似度检索 → 取出相关片段 → 喂给 LLM → 生成回答
```

### 3. 工具定义（app/agent/tools.py）

每个工具用 `@tool` 装饰器定义，包含：
- **函数名**：Agent 调用时的名称
- **文档字符串**：Agent 根据描述判断何时使用
- **参数类型**：Agent 自动构造调用参数

---

## 🌟 面试亮点

1. **ReAct 模式**：Agent 不是简单的问答，而是"思考→行动→观察→再思考"的循环
2. **RAG 检索增强**：基于向量数据库的语义检索，而非关键词匹配
3. **LangGraph 状态图**：用图结构定义 Agent 工作流，可扩展性强
4. **工具调用**：Agent 自主决定调用哪个工具，体现了 Agent 的"智能"
5. **工程化实践**：Docker 容器化、环境变量管理、API 接口设计、会话隔离
6. **可观测性**：每个工具调用都有明确的输入输出，便于调试

---

## 📄 License

MIT
