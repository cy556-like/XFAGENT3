"""
Agent 核心逻辑模块
使用 LangGraph 构建 ReAct 模式的 Agent
ReAct = Reasoning(推理) + Acting(行动) → 边思考边行动

优化策略：
1. LLM 超时保护 — 防止 httpx.ReadTimeout（设置 timeout + streaming_timeout）
2. 流式输出 — 用户实时看到回复，感知更快
3. 智能历史裁剪 — 去掉过长的 ToolMessage，保留对话主线
4. 搜索次数限制 — 每轮最多搜索 N 次，防止冗余搜索
5. 超时重试 — think 节点自带重试，防止单次超时导致整体失败
6. Agent 实例定期重建 — 防止长时间运行导致内存泄漏
"""
import asyncio
import logging
import time
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agent.tools import ALL_TOOLS, set_current_agent_id, set_current_session_id, reset_search_count
from app.agent.prompts import SYSTEM_PROMPT
from app.memory.manager import get_session_history

logger = logging.getLogger(__name__)

# 最大历史消息数量（加速推理，避免上下文过长）
MAX_HISTORY_MESSAGES = 8

# 最大工具调用轮数（防止无限循环，同时保证复杂任务能完成）
MAX_TOOL_ROUNDS = 5

# 每轮最大搜索次数（防止冗余搜索）
MAX_SEARCH_PER_ROUND = 3

# think 节点超时重试次数
THINK_RETRY_COUNT = 2

# Agent 实例重建间隔（秒）：防止长时间运行导致内存泄漏
AGENT_REBUILD_INTERVAL = 3600  # 1小时重建一次


# ===== 1. 定义 Agent 状态 =====
class AgentState(TypedDict):
    """
    Agent 的状态定义
    messages 使用 add_messages 策略：新消息追加而非覆盖
    search_count: 当前轮次的搜索次数计数
    """
    messages: Annotated[list, add_messages]
    search_count: int


# ===== 2. 创建 LLM =====
def create_llm():
    """
    创建 LLM 实例（带完整超时保护）

    关键：ChatOpenAI 的 timeout 参数直接传给底层 httpx，
    必须同时设置 read timeout，否则流式响应时会超时。
    """
    timeout_seconds = getattr(settings, 'LLM_TIMEOUT', 120)

    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.1,
        timeout=timeout_seconds,           # 总超时（秒）
        max_retries=1,                     # 超时自动重试1次（不要太多，避免用户等太久）
        streaming=True,                    # 流式输出，减少首 token 等待感知
    )


# ===== 3. 智能裁剪历史消息 =====
def trim_history(messages: list, max_count: int = MAX_HISTORY_MESSAGES) -> list:
    """
    智能裁剪历史消息
    - 优先保留 HumanMessage 和 AIMessage（对话主线）
    - ToolMessage 如果太长则截断
    - 保留最近的消息
    """
    if len(messages) <= max_count:
        return messages

    # 只取最近的消息
    recent = messages[-max_count:]

    # 确保 ToolMessage 不会太长（限制单个 ToolMessage 内容长度）
    trimmed = []
    for msg in recent:
        if isinstance(msg, ToolMessage) and len(msg.content) > 1500:
            # 截断过长的工具返回内容，保留开头和结尾
            truncated = msg.content[:1000] + "\n...[内容过长已截断]...\n" + msg.content[-300:]
            new_msg = ToolMessage(content=truncated, tool_call_id=msg.tool_call_id)
            trimmed.append(new_msg)
        else:
            trimmed.append(msg)

    return trimmed


# ===== 4. 构建 Agent 图 =====
def create_agent_graph():
    """
    构建 LangGraph Agent 执行图

    流程：用户输入 → LLM 思考 → 是否调用工具？
           ├─ 是 → 执行工具 → 回到 LLM 思考（循环，最多 MAX_TOOL_ROUNDS 轮）
           └─ 否 → 输出回答 → 结束
    """
    llm = create_llm()

    # 将工具绑定到 LLM，让它知道有哪些工具可以用
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 搜索计数器（每轮对话重置）
    search_counter = {"count": 0}

    # 节点1: LLM 思考节点（带超时重试）
    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具（含超时重试）"""
        messages = state["messages"]
        system_msg = SystemMessage(content=SYSTEM_PROMPT)

        last_error = None
        for attempt in range(THINK_RETRY_COUNT + 1):
            try:
                response = llm_with_tools.invoke([system_msg] + messages)
                return {"messages": [response], "search_count": state.get("search_count", 0)}
            except Exception as e:
                last_error = e
                error_msg = str(e)
                if "ReadTimeout" in error_msg or "timed out" in error_msg:
                    if attempt < THINK_RETRY_COUNT:
                        logger.warning(f"think 节点超时，第 {attempt + 1} 次重试...")
                        continue
                    else:
                        logger.error(f"think 节点超时，已重试 {THINK_RETRY_COUNT} 次，强制返回超时提示")
                        # 返回一个超时提示的 AI 消息，而不是让整个 Agent 崩溃
                        timeout_msg = AIMessage(content="⚠️ 抱歉，LLM 响应超时，请稍后重试或简化您的问题。")
                        return {"messages": [timeout_msg], "search_count": state.get("search_count", 0)}
                else:
                    # 非超时错误，不重试，直接抛出
                    raise

        # 不应该走到这里，但以防万一
        timeout_msg = AIMessage(content="⚠️ 抱歉，LLM 响应超时，请稍后重试。")
        return {"messages": [timeout_msg], "search_count": state.get("search_count", 0)}

    # 节点2: 工具执行节点（使用 LangGraph 内置的 ToolNode）
    tool_node = ToolNode(ALL_TOOLS)

    # 条件边：判断是否需要继续调用工具（限制最大轮数 + 搜索次数）
    def should_continue(state: AgentState):
        """
        判断是否需要继续调用工具
        - 如果工具调用轮数超过 MAX_TOOL_ROUNDS，强制结束
        - 如果搜索次数超过 MAX_SEARCH_PER_ROUND，强制结束
        """
        messages = state["messages"]
        search_count = state.get("search_count", 0)

        # 统计 ToolMessage 的数量，每轮工具调用会产生一个 ToolMessage
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            return END

        # 搜索次数超限
        if search_count >= MAX_SEARCH_PER_ROUND:
            return END

        # 使用 LangGraph 内置判断：最后一条消息是否有工具调用
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # 检查是否是搜索工具调用
            for tool_call in last_message.tool_calls:
                if tool_call.get("name") == "search_documents_tool":
                    if search_count >= MAX_SEARCH_PER_ROUND:
                        return END
            return "act"

        return END

    # 工具执行后更新搜索计数
    def act_with_counter(state: AgentState):
        """执行工具并更新搜索计数"""
        result = tool_node.invoke(state)
        messages = state["messages"]
        last_message = messages[-1]
        search_count = state.get("search_count", 0)
        if hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                if tool_call.get("name") == "search_documents_tool":
                    search_count += 1
        result["search_count"] = search_count
        return result

    # ===== 构建状态图 =====
    graph = StateGraph(AgentState)

    graph.add_node("think", think)
    graph.add_node("act", act_with_counter)

    graph.set_entry_point("think")

    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "act": "act",
            END: END,
        },
    )

    graph.add_edge("act", "think")

    return graph.compile()


# ===== 5. 全局 Agent 实例（带定期重建） =====
_agent_graph = None
_agent_create_time = 0


def get_agent():
    """
    获取 Agent 单例（懒加载，定期重建防止内存泄漏）

    重建机制：每隔 AGENT_REBUILD_INTERVAL 秒重建一次 Agent 实例，
    防止长时间运行导致的 LLM 连接池泄漏和内存累积。
    """
    global _agent_graph, _agent_create_time
    now = time.time()
    if _agent_graph is None or (now - _agent_create_time) > AGENT_REBUILD_INTERVAL:
        if _agent_graph is not None:
            logger.info(f"Agent 实例已运行 {int(now - _agent_create_time)} 秒，正在重建...")
        _agent_graph = create_agent_graph()
        _agent_create_time = now
    return _agent_graph


def _parse_session_agent_id(session_id: str) -> str:
    """从 session_id 中解析 agent_id（如果 session_id 包含 agent 信息）"""
    # session_id 格式：username_uuid 或 username_agentid_uuid
    # 目前简单处理：返回 None，后续可以扩展
    return None


def chat(user_input: str, session_id: str = "default") -> str:
    """
    与 Agent 对话的核心方法（同步版本，供 REST API 使用）

    优化：
    - 每次对话前重置搜索计数
    - 使用 trim_history 防止上下文过长
    - 异常处理：超时返回友好提示，不崩溃
    """
    # 重置搜索计数
    reset_search_count()

    agent = get_agent()
    history = get_session_history(session_id)
    recent_messages = trim_history(history.messages, MAX_HISTORY_MESSAGES)
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    try:
        result = agent.invoke({"messages": all_messages, "search_count": 0})
        ai_message = result["messages"][-1]
    except Exception as e:
        error_msg = str(e)
        if "ReadTimeout" in error_msg or "timed out" in error_msg:
            ai_message = AIMessage(content="⚠️ LLM 响应超时，请稍后重试或简化您的问题。")
        else:
            ai_message = AIMessage(content=f"⚠️ 处理出错: {error_msg}")

    # 保存到会话历史
    history.add_message(HumanMessage(content=user_input))
    history.add_message(ai_message)

    return ai_message.content


# ===== 6. 流式输出 =====
async def chat_stream(user_input: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """
    与 Agent 对话的流式输出方法（供 SSE 使用）

    优化：
    - 每次对话前重置搜索计数
    - 使用 trim_history 防止上下文过长
    - 异常处理：超时返回友好提示，不崩溃
    - 完成后自动保存到历史
    """
    # 重置搜索计数
    reset_search_count()

    agent = get_agent()
    history = get_session_history(session_id)
    recent_messages = trim_history(history.messages, MAX_HISTORY_MESSAGES)
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    full_response = ""
    search_count = 0

    try:
        async for event in agent.astream_events(
            {"messages": all_messages, "search_count": 0},
            version="v1",
        ):
            kind = event.get("event", "")

            # 流式输出 LLM 的文本内容
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    if isinstance(content, str):
                        full_response += content
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                full_response += item["text"]
                                yield item["text"]

            # 工具调用事件（给用户展示进度）
            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                if tool_name == "search_documents_tool":
                    search_count += 1
                    yield f"\n🔍 搜索文档中...（第{search_count}次）\n"
                elif tool_name == "lookup_employee_tool":
                    yield "\n👤 查询员工信息中...\n"
                elif tool_name == "list_documents_tool":
                    yield "\n📋 获取文档列表中...\n"

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                if tool_name == "search_documents_tool":
                    yield "✅ 搜索完成\n"

    except asyncio.TimeoutError:
        if not full_response:
            full_response = "⚠️ 响应超时，请稍后重试。"
            yield full_response
    except Exception as e:
        error_msg = str(e)
        if "ReadTimeout" in error_msg or "timed out" in error_msg:
            if not full_response:
                full_response = "⚠️ LLM 响应超时，请稍后重试或简化您的问题。"
                yield full_response
        else:
            if not full_response:
                full_response = f"⚠️ 处理出错: {error_msg}"
                yield full_response

    # 保存到会话历史
    if full_response:
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=full_response))
