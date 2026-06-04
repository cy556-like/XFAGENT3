"""
Agent 核心逻辑模块
使用 LangGraph 构建 ReAct 模式的 Agent
ReAct = Reasoning(推理) + Acting(行动) → 边思考边行动
支持流式输出（Streaming SSE）
支持多步骤任务编排、工具并行执行、自省纠错

性能优化:
- 流式首Token优化：使用 astream_events v2 减少首Token延迟
- 意图路由：简单问题跳过Agent工具调用，直接LLM回答
- 提示词精简：减少系统提示词Token数，加速推理
- 历史消息窗口：限制上下文长度，避免过长上下文拖慢推理
- Agent单例复用：避免每次请求重建Agent图
"""
import asyncio
import time
import logging
import hashlib
import contextvars
import threading
from datetime import datetime
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings, VISION_MODELS, DEFAULT_VISION_MODEL, FAST_MODELS
from app.agent.tools import ALL_TOOLS, get_tools, set_current_agent_id, set_current_session_id, reset_search_count
from app.agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_WEB_SEARCH, CHAT_SYSTEM_PROMPT
from app.memory.manager import get_session_history

logger = logging.getLogger(__name__)

# [BUG FIX] 会话取消信号：当用户终止对话时，通过 contextvars 传递取消标志给 think()
# 解决同步 invoke() 在独立线程中无法被异步取消、导致"幽灵"LLM调用继续消耗 rate limit 的问题
# contextvars 自动从 asyncio 任务传播到 ThreadPoolExecutor 子线程
_cancel_ctx: contextvars.ContextVar = contextvars.ContextVar('cancel_event', default=None)

def _is_session_cancelled() -> bool:
    """检查当前 session 是否已被取消（在 think() 中调用以避免无效 LLM 调用）"""
    evt = _cancel_ctx.get(None)
    if evt is not None and evt.is_set():
        return True
    return False

# 最大历史消息数量（加速推理，避免上下文过长）
MAX_HISTORY_MESSAGES = 10

# [#6] 多步骤任务编排：最大工具调用轮数
# 从8降到5：大多数场景2-3次搜索+1次导出即完成，8轮导致LLM过度搜索
# 5轮仍足够处理复杂任务（3次搜索 + 2次其他操作）
MAX_TOOL_ROUNDS = 5

# [#11] 工具重试配置
MAX_TOOL_RETRIES = 2
RETRYABLE_TOOL_ERRORS = ["搜索失败", "未找到", "连接", "超时", "timeout", "error"]

# 意图路由：简单问题关键词（匹配这些关键词的问题直接用Chat模式，跳过Agent工具调用）
SIMPLE_QUERY_PATTERNS = [
    "你好", "嗨", "hello", "hi", "你是谁", "你叫什么", "介绍一下你自己",
    "谢谢", "感谢", "再见", "拜拜", "好的", "知道了",
]

def _is_simple_query(query: str) -> bool:
    """判断用户输入是否为简单问题（不需要工具调用的闲聊/通用问题）
    
    简单问题走 Chat 模式直接回答，跳过 Agent 的 Think→Act→Observe 循环，
    可以将响应时间从 3-5秒 降低到 0.5-1秒（首Token延迟）。
    """
    query_stripped = query.strip()
    # 短文本匹配（<15字）
    if len(query_stripped) <= 15:
        for pattern in SIMPLE_QUERY_PATTERNS:
            if pattern in query_stripped.lower():
                return True
    return False

def _inject_current_date(system_prompt: str) -> str:
    """在系统提示词中注入当前日期，避免LLM回复错误的时间信息
    
    在提示词末尾追加当前日期，让LLM知道现在是什么时间，
    避免在引用变更记录、描述事件时间时使用错误的年份。
    同时添加禁止编造日期的规则。
    """
    now = datetime.now()
    date_info = f"\n\n## 当前时间（重要！必须遵守）\n当前日期：{now.strftime('%Y年%m月%d日')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}。请在回答中涉及时间信息时使用正确的当前日期。**严禁**编造、猜测或使用错误的年份和日期。当知识库文档中没有明确标注日期时，**不要**自行添加\"XX年变更记录\"之类的日期描述，只需说\"根据知识库文档记载\"。"
    return system_prompt + date_info

# ===== 1. 定义 Agent 状态 =====
class AgentState(TypedDict):
    """
    Agent 的状态定义
    messages 使用 add_messages 策略：新消息追加而非覆盖
    retry_count: [#11] 工具重试计数
    """
    messages: Annotated[list, add_messages]
    retry_count: int

# ===== 2. 创建 LLM =====
# 主Key是否已确认失效（运行时标记，避免每次都重试失败的Key）
_primary_key_failed = False

# [优化1] LLM Client 缓存：按 (model, api_key, base_url, temperature) 缓存 ChatOpenAI 实例
# 避免每次请求新建 HTTP 连接，减少 500ms-3s 的连接建立开销
_llm_cache = {}  # cache_key -> ChatOpenAI instance

def create_llm(deep_think: bool = False, fast_mode: bool = False):
    """创建 LLM 实例（启用 streaming 支持，支持备用Key自动切换）
    
    [优化1] 使用缓存：按 (model, api_key, base_url, temperature) 作为缓存 key，
    相同参数复用同一个 ChatOpenAI 实例，避免每次请求重建 HTTP 连接（TCP + TLS 握手）。
    每次新建连接耗时 500ms-3s，复用后降至 <50ms。
    
    Args:
        deep_think: 是否启用深度思考模式（使用更强的模型）
        fast_mode: 是否使用快速模型（用于简单问题的快速响应）
    """
    global _primary_key_failed
    model = settings.LLM_MODEL
    
    if fast_mode:
        for m in ["glm-4-flash", "glm-4-air", "glm-4.7-flash"]:
            if m != model:
                model = m
                break
        logger.info(f"快速模式：使用模型 {model}")
    elif deep_think:
        deep_think_models = ["glm-4-plus", "glm-4.7", "glm-4-long", "glm-4-air"]
        for m in deep_think_models:
            if m != model:
                model = m
                break
        logger.info(f"深度思考模式：尝试使用模型 {model}")

    # 决定使用主Key还是备用Key
    use_backup = _primary_key_failed and settings.LLM_API_KEY_BACKUP
    api_key = settings.LLM_API_KEY_BACKUP if use_backup else settings.LLM_API_KEY
    base_url = settings.LLM_BASE_URL_BACKUP if use_backup else settings.LLM_BASE_URL
    temperature = 0.3 if deep_think else 0.1

    # [优化1] 检查缓存，复用已有的 ChatOpenAI 实例
    cache_key = (model, api_key, base_url, temperature)
    if cache_key in _llm_cache:
        logger.debug(f"LLM Client 缓存命中: model={model}")
        return _llm_cache[cache_key]

    if use_backup:
        logger.info(f"使用备用API Key（主Key已失效）: {base_url}")
    else:
        logger.info(f"使用主API Key: {base_url}")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        streaming=True,
        max_tokens=8192,
        request_timeout=120,  # 超时保护：120秒，长提示词需要更多时间
    )
    _llm_cache[cache_key] = llm
    logger.info(f"LLM Client 已创建并缓存: model={model}, 缓存数量={len(_llm_cache)}")
    return llm

def _check_and_switch_to_backup(error_exception):
    """检测到401错误时，自动切换到备用Key"""
    global _primary_key_failed
    error_str = str(error_exception).lower()
    if ("401" in error_str or "authentication" in error_str or "令牌" in error_str) and settings.LLM_API_KEY_BACKUP:
        if not _primary_key_failed:
            _primary_key_failed = True
            logger.warning(f"⚠️ 主API Key认证失败(401)，已自动切换到备用Key: {settings.LLM_BASE_URL_BACKUP}")
        return True
    return False

def reset_primary_key():
    """重置主Key状态（更换Key后调用）"""
    global _primary_key_failed
    _primary_key_failed = False

# ===== 3. 构建 Agent 图 =====
def create_agent_graph(web_search: bool = False):
    """
    构建 LangGraph Agent 执行图

    Args:
        web_search: 是否启用联网搜索工具

    流程：用户输入 → LLM 思考 → 是否调用工具？
           ├─ 是 → 执行工具 → 回到 LLM 思考（循环，最多8轮）
           └─ 否 → 输出回答 → 结束
    """
    llm = create_llm()
    tools = get_tools(web_search=web_search)
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = _inject_current_date(SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT)

    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具
        
        [性能修复 v3] 使用同步 invoke()：
        - invoke() 在独立线程执行，不与 astream_events 的事件循环竞争
        - 多轮工具调用场景（think→tool→think→tool），async ainvoke() 每轮多 3-5s 事件循环竞争
        - 3轮工具调用累计多 9-15s，这就是比 XF4 慢 10+ 秒的根因
        - 之前认为 sync 导致 stream 事件丢失，实际是 prompt 优化导致工具循环（已修复）
        
        [BUG FIX] 取消检查：用户终止对话时，通过 contextvars 传递的取消信号
        会阻止新一轮 think() 发起 LLM 调用，避免"幽灵"调用消耗 rate limit
        """
        if _is_session_cancelled():
            logger.warning("检测到会话已取消，跳过 LLM 调用")
            raise RuntimeError("Session cancelled by user")
        messages = state["messages"]
        system_msg = SystemMessage(content=system_prompt)
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        """判断是否需要继续调用工具"""
        messages = state["messages"]
        retry_count = state.get("retry_count", 0)
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            logger.info(f"Agent 工具调用已达上限 {MAX_TOOL_ROUNDS} 轮，强制结束")
            return END

        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if tool_message_count > 0:
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if any(err in tool_result for err in RETRYABLE_TOOL_ERRORS):
                            if retry_count < MAX_TOOL_RETRIES:
                                logger.info(f"Agent 检测到工具错误，第 {retry_count + 1} 次重试")
                                return "act"
                            else:
                                logger.info(f"Agent 工具重试已达上限 {MAX_TOOL_RETRIES} 次，继续执行")
                        break
            return "act"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", tool_node)
    graph.set_entry_point("think")
    graph.add_conditional_edges("think", should_continue, {"act": "act", END: END})
    graph.add_edge("act", "think")
    return graph.compile()

# ===== 4. Agent 实例管理 =====
_agent_graph = None
_agent_web_search = False

def get_agent(web_search: bool = False):
    """获取 Agent 实例（懒加载，根据 web_search 参数决定是否包含联网搜索工具）"""
    global _agent_graph, _agent_web_search
    if _agent_graph is None or _agent_web_search != web_search:
        _agent_graph = create_agent_graph(web_search=web_search)
        _agent_web_search = web_search
    return _agent_graph

def reset_agent():
    """重置 Agent 实例（切换模型后调用，下次对话会自动重建）"""
    global _agent_graph, _llm_cache, _agent_prompt_graph_cache
    _agent_graph = None
    _llm_cache.clear()  # [优化1] 模型切换时清空 LLM 缓存
    _agent_prompt_graph_cache.clear()  # [优化2] 清空 Agent Graph 缓存
    _agent_prompt_graph_timestamps.clear()  # 清空缓存时间戳


def cleanup_stale_caches():
    """[性能修复] 定期清理过期的缓存，防止长时间运行后内存增长
    
    由 main.py 的定期清理任务每10分钟调用一次。
    清理内容：
    1. 超过30分钟未使用的 Agent Graph 缓存
    2. 超过1小时未使用的 LLM Client 缓存（TCP连接会被服务端关闭，缓存的连接已无效）
    """
    _cleanup_stale_graph_cache()
    
    # [性能修复] 清理长时间未使用的 LLM Client 缓存
    # ChatOpenAI 实例内部持有 httpx 连接池，长时间不用会占用文件描述符
    # 只保留当前活跃模型的客户端
    global _llm_cache
    if len(_llm_cache) > 2:
        # 保留当前模型的缓存，清理其他
        current_key = None
        for key in _llm_cache:
            model, api_key, base_url, temp = key
            if model == settings.LLM_MODEL:
                current_key = key
                break
        if current_key and current_key in _llm_cache:
            kept = {current_key: _llm_cache[current_key]}
            _llm_cache.clear()
            _llm_cache.update(kept)
            logger.info(f"[缓存清理] LLM Client 缓存清理完成，保留当前模型，清理前={len(_llm_cache)+1 if current_key else 0}，清理后=1")

# [性能修复] Agent Graph 缓存过期检查：超过30分钟未使用的缓存自动清理
_AGENT_GRAPH_CACHE_TTL = 1800  # 30分钟
_agent_prompt_graph_timestamps = {}  # cache_key -> last_access_time

def _cleanup_stale_graph_cache():
    """清理超过 TTL 未使用的 Agent Graph 缓存，防止长时间运行后内存增长"""
    now = time.time()
    stale_keys = [k for k, t in _agent_prompt_graph_timestamps.items() if now - t > _AGENT_GRAPH_CACHE_TTL]
    for k in stale_keys:
        if k in _agent_prompt_graph_cache:
            del _agent_prompt_graph_cache[k]
        del _agent_prompt_graph_timestamps[k]
    if stale_keys:
        logger.info(f"[缓存清理] 清理了 {len(stale_keys)} 个过期 Agent Graph 缓存（>{_AGENT_GRAPH_CACHE_TTL}s未使用）")

def _build_agent_prompt(agent_task: str, web_search: bool = False) -> str:
    """根据智能体的任务描述构建专属系统提示词
    
    智能体的任务描述将作为角色定义的优先内容，
    覆盖默认的「小智」角色，但保留工具使用指南和安全边界。
    
    关键改进：当智能体有自定义任务描述时，强制要求优先检索知识库，
    避免LLM将专业问题误判为"通用问题"而直接回答。
    """
    base_prompt = SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT
    
    custom_header = f"""# 角色

{agent_task}

## 身份
- 你的角色由上述定义决定，请严格按照任务描述中的角色定位和行为规则来行动
- 你的核心职责是完成上述任务描述中定义的工作
- 语气与风格应与角色定位保持一致

## 重要原则：不要拒绝合理请求
在符合角色定位的前提下，用户提出的合理请求你应当尽力帮助。
**绝对不要**说"这不属于我的服务范围"或"我无法帮你"这类话——只要你能做到，就给出回答。

## 知识库优先规则（最高优先级！必须严格遵守）
作为专属智能体，你**必须优先检索知识库**来回答用户的任何专业问题，而不是直接用自身知识回答。

### 强制检索规则
1. **凡是与你角色定义（上述任务描述）相关的专业问题，必须先调用 search_documents_tool 检索知识库**，基于检索结果回答
2. **即使用户没有明确说"根据知识库回答"，你也必须自动检索知识库**——用户默认期望你从知识库获取专业信息
3. 如果知识库检索无结果，可以补充自身知识，但**必须明确标注**：「以下内容非来自知识库，仅供参考」
4. **绝对禁止**在未检索知识库的情况下，直接用自己的知识回答专业问题

### 搜索效率规则
- 同一主题只搜1次，用组合关键词，不要拆成多次搜索
- 每轮最多搜索3次，信息足够就回答
- 生成文档时搜1次拿模板后直接生成

### 判断标准
- 必须检索知识库的问题：与你的角色定义、专业领域、公司制度/流程/规范/标准相关的问题
  - 示例：「FMEA成员有哪些」「质量方针是什么」「VDA6.4有什么要求」「乌龟图怎么画」
- 也必须检索：看起来简单但可能知识库有专门记载的问题
  - 示例：「团队有哪些人」「流程是什么」「有哪些文件」
- 可以直接回答的问题：纯编程、数学计算、翻译、闲聊等与你专业领域无关的通用问题
  - 示例：「Python怎么写」「1+1等于几」「帮我翻译一下」

### 标准回答流程
```
用户提问 → 判断是否与专业领域相关？
├─ 是 → 1. 先调用 search_documents_tool 检索知识库
│       2. 基于检索结果回答，标注来源
│       3. 如果无结果，补充自身知识并标注
└─ 否 → 直接回答（通用问题）
```

"""
    
    tools_section_marker = "## 工具选择指南"
    tools_idx = base_prompt.find(tools_section_marker)
    
    if tools_idx > 0:
        preserved_section = base_prompt[tools_idx:]
        result = custom_header + preserved_section
        
        old_general_rule = """### 通用问题处理
- 编程、知识问答、写作、翻译等通用问题，**直接用自己的知识回答**，不要拒绝
- 不要说"这不是我的服务范围"——回答时依然保持专业、清晰"""
        
        new_general_rule = """### 通用问题处理规则
- **与你的专业领域相关的问题**（如质量体系、公司制度、流程规范等）：必须先检索知识库，详见上方「知识库优先规则」
- **纯通用问题**（编程、数学计算、翻译、闲聊等与专业领域无关的问题）：可以直接用自己的知识回答
- 不要说"这不是我的服务范围"、"我只处理企业事务"之类的话
- 回答通用问题时，依然保持专业、清晰的风格"""
        
        result = result.replace(old_general_rule, new_general_rule)
        return result
    else:
        return custom_header + base_prompt

def _build_chat_prompt(agent_task: str) -> str:
    """根据智能体任务描述构建Chat模式的系统提示词"""
    return f"""{agent_task}

## 核心原则
- 严格按照上述角色定义来回答问题
- 专业、简洁、友好，使用规范中文回答
- 不拒绝合理的用户请求，尽力提供有价值的帮助
- 回答要有深度和细节，不要过于简略
- 适时使用结构化格式（编号、分段、表格）组织回答

## 知识库优先规则
- 与你的专业领域相关的问题，必须优先基于知识库内容回答
- 如果知识库中没有相关信息，可以补充自身知识，但必须标注：「以下内容非来自知识库，仅供参考」
- 只有纯通用问题（编程、数学、翻译、闲聊等与专业领域无关的问题）可以直接回答

## 回答规则
- 编程问题：给出完整代码，附上关键注释和运行说明
- 知识问答：准确、详细地回答，必要时补充背景信息
- 写作任务：根据需求撰写，保持风格一致
- 翻译任务：准确翻译，保留原文的语气和风格
- 闲聊：轻松自然地回应

## 格式要求
- 使用Markdown格式组织回答
- 代码使用代码块，标注语言类型
- 涉及流程时使用有序列表
- 涉及对比时使用表格
"""

# [优化2] Agent Graph 缓存：自定义 prompt 的图按 hash 缓存，避免重复编译
# 同一个 agent_task + web_search 组合只需编译一次 LangGraph
_agent_prompt_graph_cache = {}  # cache_key -> compiled graph
_AGENT_PROMPT_CACHE_MAX_SIZE = 8  # 最多缓存 8 个不同的自定义 Agent 图

def get_agent_with_prompt(custom_system_prompt: str, web_search: bool = False):
    """获取带有自定义系统提示词的 Agent 实例
    
    [优化2] 按 prompt hash + web_search 缓存编译后的 Agent Graph，
    避免每次 agent_task 请求都重新编译 LangGraph（减少 200ms-1s）。
    同一个智能体连续对话时，直接复用已编译的图。
    """
    # 生成缓存 key
    prompt_hash = hashlib.md5(custom_system_prompt.encode()).hexdigest()[:16]
    cache_key = f"{prompt_hash}:{web_search}"
    
    if cache_key in _agent_prompt_graph_cache:
        logger.debug(f"Agent Graph 缓存命中: prompt_hash={prompt_hash}, web_search={web_search}")
        _agent_prompt_graph_timestamps[cache_key] = time.time()  # [性能修复] 更新访问时间
        return _agent_prompt_graph_cache[cache_key]
    
    llm = create_llm()
    tools = get_tools(web_search=web_search)
    llm_with_tools = llm.bind_tools(tools)

    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具
        
        [性能修复 v3] 同上，使用同步 invoke() 避免事件循环竞争。
        
        [BUG FIX] 取消检查：用户终止对话时，通过 contextvars 传递的取消信号
        会阻止新一轮 think() 发起 LLM 调用，避免"幽灵"调用消耗 rate limit
        """
        if _is_session_cancelled():
            logger.warning("检测到会话已取消，跳过 LLM 调用（自定义智能体）")
            raise RuntimeError("Session cancelled by user")
        messages = state["messages"]
        system_msg = SystemMessage(content=custom_system_prompt)
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        messages = state["messages"]
        retry_count = state.get("retry_count", 0)
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            return END

        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if tool_message_count > 0:
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if any(err in tool_result for err in RETRYABLE_TOOL_ERRORS):
                            if retry_count < MAX_TOOL_RETRIES:
                                return "act"
                        break
            return "act"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", tool_node)
    graph.set_entry_point("think")
    graph.add_conditional_edges("think", should_continue, {"act": "act", END: END})
    graph.add_edge("act", "think")
    compiled = graph.compile()
    
    # [优化2] 缓存编译结果（LRU：超过上限时移除最早的）
    if len(_agent_prompt_graph_cache) >= _AGENT_PROMPT_CACHE_MAX_SIZE:
        oldest_key = next(iter(_agent_prompt_graph_cache))
        del _agent_prompt_graph_cache[oldest_key]
        logger.debug(f"Agent Graph 缓存已满，淘汰: {oldest_key}")
    _agent_prompt_graph_cache[cache_key] = compiled
    _agent_prompt_graph_timestamps[cache_key] = time.time()  # [性能修复] 记录缓存时间
    _cleanup_stale_graph_cache()  # [性能修复] 顺便清理过期缓存
    logger.info(f"Agent Graph 已编译并缓存: prompt_hash={prompt_hash}, web_search={web_search}, 缓存数量={len(_agent_prompt_graph_cache)}")
    
    return compiled

def chat(user_input: str, session_id: str = "default", web_search: bool = False, mode: str = "agent", deep_think: bool = False, agent_id: str = None, agent_task: str = None) -> str:
    """非流式对话（保留兼容）"""
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    reset_search_count()  # 每轮新对话重置搜索计数
    
    if agent_task:
        custom_prompt = _inject_current_date(_build_agent_prompt(agent_task, web_search=web_search))
    elif web_search:
        custom_prompt = _inject_current_date(SYSTEM_PROMPT_WITH_WEB_SEARCH)
    else:
        custom_prompt = _inject_current_date(SYSTEM_PROMPT)
    
    if mode == "chat":
        llm = create_llm(deep_think=deep_think)
        history = get_session_history(session_id)
        recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
        all_messages = recent_messages + [HumanMessage(content=user_input)]
        chat_prompt = _inject_current_date(_build_chat_prompt(agent_task) if agent_task else CHAT_SYSTEM_PROMPT)
        result = llm.invoke([SystemMessage(content=chat_prompt)] + all_messages)
        full_response = result.content
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=full_response))
        return full_response

    agent = get_agent(web_search=web_search)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    all_messages = recent_messages + [HumanMessage(content=user_input)]
    result = agent.invoke({"messages": all_messages, "retry_count": 0})
    ai_message = result["messages"][-1]
    history.add_message(HumanMessage(content=user_input))
    history.add_message(ai_message)
    return ai_message.content

# ===== 5. 流式对话 =====

TOOL_DISPLAY_NAMES = {
    "search_documents_tool": "搜索文档",
    "lookup_employee_tool": "查询员工",
    "list_departments_tool": "部门列表",
    "list_documents_tool": "文档列表",
    "get_document_content_tool": "获取文档全文",
    "upload_document_tool": "上传文档",
    "delete_document_tool": "删除文档",
    "modify_document_tool": "修改文档",
    "export_document_tool": "导出文档",
    "export_xlsx_tool": "导出Excel",
    "web_search_tool": "联网搜索",
    "github_api_tool": "GitHub操作",
    "send_email_tool": "发送邮件",
    "database_query_tool": "数据库查询",
}

def _extract_content(chunk) -> str:
    """从流式 chunk 中提取文本内容，处理字符串和列表两种格式
    
    某些LLM返回的content是列表格式（如包含tool_calls时），
    需要安全地提取文本部分，避免拼接错误。
    """
    content = getattr(chunk, 'content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif isinstance(item, str):
                text_parts.append(item)
        return ''.join(text_parts)
    return ''

# [BUG FIX] 整体超时保护：Agent 对话最大允许时长（秒）
# 超过此时间强制结束，避免 LLM API 挂起导致服务器无响应需 Ctrl+C
AGENT_STREAM_TIMEOUT = 180  # 3分钟

async def chat_stream_generator(user_input: str, session_id: str = "default", web_search: bool = False, mode: str = "agent", deep_think: bool = False, agent_id: str = None, agent_task: str = None) -> AsyncGenerator[dict, None]:
    """流式对话：逐token输出，同时显示工具调用进度
    
    性能优化：
    - 意图路由：简单问题自动走Chat模式，跳过Agent工具循环
    - 流式首Token：使用 astream_events v2 实现毫秒级首Token输出
    - 非流式回退：流式失败时自动回退到非流式，确保总能得到回复
    
    BUG FIX：
    - 添加整体超时保护，防止 LLM API 挂起导致服务器无响应
    - 确保 tool_done 和 done 事件总是发送，避免前端工具标签一直转圈
    - 跟踪已启动的工具，异常时自动发送未完成的 tool_done 事件
    - [v5] 会话级取消追踪：终止对话时真正取消 Agent 执行，不再只是停止消费事件
    """
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    reset_search_count()  # 每轮新对话重置搜索计数
    
    # [BUG FIX v5] 为每次对话创建独立的取消信号
    cancel_event = threading.Event()
    
    # [BUG FIX v5] 取消同一 session 上一个还在执行的 Agent 任务
    # 解决用户终止对话后立即开启新对话时，上一个"幽灵"任务仍在消耗 LLM rate limit 的问题
    previous_event = _cancel_ctx.get(None)
    if previous_event is not None:
        previous_event.set()
        logger.info(f"[取消追踪] 标记上一个 Agent 任务为已取消: session={session_id}")
    
    # [BUG FIX v5] 将当前对话的取消信号注入 context，自动传播到子线程
    token = _cancel_ctx.set(cancel_event)
    
    # 性能优化：意图路由 - 简单问题走Chat模式（跳过Agent循环，减少3-5秒延迟）
    if mode == "agent" and _is_simple_query(user_input) and not web_search:
        logger.info(f"意图路由：检测到简单问题，自动走Chat模式加速响应")
        mode = "chat"
    
    if mode == "chat":
        async for chunk in _chat_mode_stream(user_input, session_id, deep_think=deep_think, web_search=web_search, agent_id=agent_id, agent_task=agent_task):
            yield chunk
        return

    # Agent模式：走Agent工具调用
    if agent_task:
        custom_system_prompt = _inject_current_date(_build_agent_prompt(agent_task, web_search=web_search))
        agent = get_agent_with_prompt(custom_system_prompt, web_search=web_search)
    else:
        agent = get_agent(web_search=web_search)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    full_response = ""
    start_time = time.time()
    
    # [BUG FIX] 跟踪已启动但未完成的工具，异常时自动发送 tool_done
    pending_tools = {}  # tool_name -> display_name

    try:
        yield {"type": "thinking", "content": "正在思考..."}

        # [性能修复 v2] 直接使用 astream_events + 每轮超时检查
        # - 去掉了无效的 _stream_with_timeout 嵌套包装（该包装名为超时保护但实际未加 wait_for）
        # - 在每轮事件循环中检查总耗时，超过 AGENT_STREAM_TIMEOUT 则抛 TimeoutError
        # - 配合 async think() + ainvoke()，流式事件直接在事件循环中触发，不走跨线程转发
        async for event in agent.astream_events(
            {"messages": all_messages, "retry_count": 0},
            version="v2",
        ):
            # [BUG FIX] 整体超时保护：每个事件都检查总耗时
            if time.time() - start_time > AGENT_STREAM_TIMEOUT:
                raise asyncio.TimeoutError()

            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = _extract_content(chunk)
                if content:
                    full_response += content
                    yield {"type": "token", "content": content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                pending_tools[tool_name] = display_name  # [BUG FIX] 跟踪未完成工具
                yield {"type": "tool", "name": tool_name, "display": display_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                pending_tools.pop(tool_name, None)  # [BUG FIX] 标记工具已完成
                yield {"type": "tool_done", "name": tool_name, "display": display_name}
            
            # [BUG FIX] 处理工具执行出错的情况：on_tool_end 可能不会触发
            elif kind == "on_tool_error":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                pending_tools.pop(tool_name, None)  # 标记工具已完成（出错也算完成）
                yield {"type": "tool_done", "name": tool_name, "display": display_name}

    except asyncio.TimeoutError:
        # [BUG FIX] 超时时：发送未完成工具的 tool_done + error + done
        logger.warning(f"Agent 流式输出超时（{AGENT_STREAM_TIMEOUT}s），强制结束")
        for tool_name, display_name in pending_tools.items():
            yield {"type": "tool_done", "name": tool_name, "display": display_name}
        pending_tools.clear()
        yield {"type": "error", "content": f"请求超时（{AGENT_STREAM_TIMEOUT}秒），LLM服务响应过慢，请稍后重试"}
        # 确保保存已有的部分回复
        if full_response:
            try:
                history.add_message(HumanMessage(content=user_input))
                history.add_message(AIMessage(content=full_response))
            except Exception:
                pass
        yield {"type": "done"}
        return
    except asyncio.CancelledError:
        # [BUG FIX] SSE 客户端断开时：发送未完成工具的 tool_done + done
        logger.info(f"Agent 流式输出被取消（客户端断开）")
        for tool_name, display_name in pending_tools.items():
            yield {"type": "tool_done", "name": tool_name, "display": display_name}
        pending_tools.clear()
        yield {"type": "done"}
        return
    except Exception as e:
        logger.error(f"Agent 流式输出异常: {e}", exc_info=True)
        # [BUG FIX] 异常时：先发送未完成工具的 tool_done
        for tool_name, display_name in pending_tools.items():
            yield {"type": "tool_done", "name": tool_name, "display": display_name}
        pending_tools.clear()
        # 检测401认证错误，自动切换备用Key
        if _check_and_switch_to_backup(e):
            yield {"type": "error", "content": "主API Key已失效，已自动切换到备用Key，请重新提问"}
            yield {"type": "done"}
            return
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": all_messages, "retry_count": 0}),
                timeout=60.0  # [BUG FIX] 非流式回退也加超时
            )
            ai_message = result["messages"][-1]
            full_response = ai_message.content or ""
            if full_response:
                for i in range(0, len(full_response), 3):
                    yield {"type": "token", "content": full_response[i:i+3]}
                    await asyncio.sleep(0.02)
        except asyncio.TimeoutError:
            yield {"type": "error", "content": "非流式回退也超时，请稍后重试"}
            yield {"type": "done"}
            return
        except Exception as e2:
            yield {"type": "error", "content": f"处理失败: {str(e2)}"}
            yield {"type": "done"}
            return

    # 流式输出为空时回退到非流式
    if not full_response:
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": all_messages, "retry_count": 0}),
                timeout=60.0  # [BUG FIX] 非流式回退也加超时
            )
            ai_message = result["messages"][-1]
            full_response = ai_message.content or ""
            if full_response:
                for i in range(0, len(full_response), 3):
                    yield {"type": "token", "content": full_response[i:i+3]}
                    await asyncio.sleep(0.02)
            else:
                yield {"type": "error", "content": "未能获取到回复，请重试"}
        except asyncio.TimeoutError:
            yield {"type": "error", "content": "获取回复超时，请稍后重试"}
        except Exception as e3:
            yield {"type": "error", "content": f"处理失败: {str(e3)}"}

    # 保存到会话历史
    if full_response:
        try:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    elapsed = time.time() - start_time
    tool_rounds = sum(1 for m in all_messages if isinstance(m, ToolMessage))
    logger.info(f"Agent 对话完成 | 耗时={elapsed:.2f}s | 模型={settings.LLM_MODEL} | 工具轮数={tool_rounds}")

    yield {"type": "done"}

async def _chat_mode_stream(user_input: str, session_id: str = "default", deep_think: bool = False, web_search: bool = False, agent_id: str = None, agent_task: str = None) -> AsyncGenerator[dict, None]:
    """Chat模式：直接LLM流式对话，不经过Agent工具调用，可选联网搜索
    
    性能优化：Chat模式跳过了Agent的 Think→Act→Observe 循环，
    直接让LLM流式输出，首Token延迟从3-5秒降到0.5-1秒。
    """
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    chat_system_prompt = _inject_current_date(_build_chat_prompt(agent_task) if agent_task else CHAT_SYSTEM_PROMPT)
    llm = create_llm(deep_think=deep_think)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    
    # 联网搜索：先搜索再将结果注入消息
    search_context = ""
    if web_search:
        try:
            yield {"type": "thinking", "content": "正在联网搜索..."}
            yield {"type": "tool", "name": "web_search_tool", "display": "联网搜索"}
            from app.agent.tools import web_search_tool
            # [性能修复] 使用 asyncio.to_thread 在线程池中执行同步HTTP调用，避免阻塞事件循环
            # 原代码直接调用 web_search_tool.invoke() 最多阻塞15秒，期间整个服务器无法处理任何请求
            search_result = await asyncio.to_thread(web_search_tool.invoke, user_input)
            yield {"type": "tool_done", "name": "web_search_tool", "display": "联网搜索"}
            search_context = f"\n\n【联网搜索结果】\n{search_result}\n\n请根据以上联网搜索结果回答用户问题。如果搜索结果没有相关信息，请根据自身知识回答。"
        except Exception as e:
            yield {"type": "tool_done", "name": "web_search_tool", "display": "联网搜索"}
            search_context = f"\n\n【联网搜索失败：{str(e)}】请根据自身知识回答。"
    
    enhanced_input = user_input + search_context
    all_messages = recent_messages + [HumanMessage(content=enhanced_input)]

    full_response = ""

    try:
        yield {"type": "thinking", "content": "深度思考中..." if deep_think else "正在思考..."}

        async for chunk in llm.astream([SystemMessage(content=chat_system_prompt)] + all_messages):
            content = _extract_content(chunk)
            if content:
                full_response += content
                yield {"type": "token", "content": content}

    except asyncio.TimeoutError:
        yield {"type": "error", "content": "请求超时，LLM服务响应过慢，请稍后重试"}
        return
    except Exception as e:
        # 检测401认证错误，自动切换备用Key
        if _check_and_switch_to_backup(e):
            yield {"type": "error", "content": "主API Key已失效，已自动切换到备用Key，请重新提问"}
            return
        yield {"type": "error", "content": f"处理失败: {str(e)}"}
        return

    if full_response:
        try:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    yield {"type": "done"}

async def chat_stream_generator_multimodal(multimodal_content: list, session_id: str = "default", agent_id: str = None, agent_task: str = None) -> AsyncGenerator[dict, None]:
    """多模态流式对话：支持图片+文本的混合消息"""
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    current_model = settings.LLM_MODEL
    use_model = current_model
    if current_model not in VISION_MODELS:
        use_model = DEFAULT_VISION_MODEL

    llm = create_llm()  # 统一使用 create_llm()，支持备用Key自动切换
    # 如果当前模型不支持视觉，切换到视觉模型
    if settings.LLM_MODEL not in VISION_MODELS:
        llm = ChatOpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
            model=DEFAULT_VISION_MODEL,
            temperature=0.1,
            streaming=True,
            max_tokens=8192,
            request_timeout=60,
        )

    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]

    # 修复：[human_msg] → [human_msg]
    human_msg = HumanMessage(content=multimodal_content)
    all_messages = recent_messages + [human_msg]

    system_prompt = _inject_current_date(SYSTEM_PROMPT)
    if agent_task:
        system_prompt = _inject_current_date(f"{SYSTEM_PROMPT}\n\n## 你的专属任务\n{agent_task}\n\n请优先围绕上述任务回答用户问题，并在需要时调用相关工具搜索知识库。")

    full_response = ""

    try:
        yield {"type": "thinking", "content": f"正在分析图片（使用{use_model}）..."}

        async for chunk in llm.astream([SystemMessage(content=system_prompt)] + all_messages):
            content = _extract_content(chunk)
            if content:
                full_response += content
                yield {"type": "token", "content": content}

    except Exception as e:
        try:
            text_parts = [p["text"] for p in multimodal_content if p["type"] == "text"]
            fallback_text = "\n".join(text_parts) + "\n\n[注意：图片分析失败，请用文字描述你的问题]"
            async for event in chat_stream_generator(fallback_text, session_id):
                yield event
            return
        except Exception as e2:
            yield {"type": "error", "content": f"图片分析失败: {str(e2)}"}
            return

    if full_response:
        try:
            text_summary = " ".join([p["text"] for p in multimodal_content if p["type"] == "text"])
            history.add_message(HumanMessage(content=text_summary))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    yield {"type": "done"}
