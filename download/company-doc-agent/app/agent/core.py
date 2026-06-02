"""
Agent 核心逻辑模块
使用 LangGraph 构建 ReAct 模式的 Agent
ReAct = Reasoning(推理) + Acting(行动) → 边思考边行动

优化策略：
1. LLM 超时保护 — 防止 httpx.ReadTimeout
2. 流式输出 — 用户实时看到回复，感知更快
3. 智能历史裁剪 — 去掉过长的 ToolMessage，保留对话主线
4. 搜索次数限制 — 每轮最多搜索 N 次，防止冗余搜索
"""
import asyncio
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agent.tools import ALL_TOOLS
from app.agent.prompts import SYSTEM_PROMPT
from app.memory.manager import get_session_history

# 最大历史消息数量（加速推理，避免上下文过长）
MAX_HISTORY_MESSAGES = 8

# 最大工具调用轮数（防止无限循环，同时保证复杂任务能完成）
MAX_TOOL_ROUNDS = 5

# 每轮最大搜索次数（防止冗余搜索）
MAX_SEARCH_PER_ROUND = 3


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
    """创建 LLM 实例（带超时保护）"""
    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.1,
        request_timeout=settings.LLM_TIMEOUT,  # 关键：防止 ReadTimeout
        max_retries=2,  # 超时自动重试2次
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

    # 节点1: LLM 思考节点
    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具"""
        messages = state["messages"]
        # 在开头插入系统提示词
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response], "search_count": state.get("search_count", 0)}

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
                        # 搜索次数已满，不再允许搜索
                        return END
            return "act"

        return END

    # 工具执行后更新搜索计数
    def act_with_counter(state: AgentState):
        """执行工具并更新搜索计数"""
        # 先执行工具
        result = tool_node.invoke(state)
        # 检查是否有搜索工具被调用
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

    # 添加节点
    graph.add_node("think", think)       # 思考节点
    graph.add_node("act", act_with_counter)  # 行动节点（带搜索计数）

    # 设置入口
    graph.set_entry_point("think")

    # 添加条件边：思考后判断是否需要调用工具
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "act": "act",   # 需要工具 → 去执行
            END: END,       # 不需要工具 → 结束
        },
    )

    # 执行完工具后，回到思考节点（形成 ReAct 循环）
    graph.add_edge("act", "think")

    return graph.compile()


# ===== 5. 全局 Agent 实例 =====
_agent_graph = None


def get_agent():
    """获取 Agent 单例（懒加载）"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph


def chat(user_input: str, session_id: str = "default") -> str:
    """
    与 Agent 对话的核心方法（同步版本，供 REST API 使用）

    Args:
        user_input: 用户输入
        session_id: 会话 ID（支持多用户）

    Returns:
        str: Agent 的回答
    """
    agent = get_agent()

    # 获取该会话的历史消息
    history = get_session_history(session_id)

    # 智能裁剪历史消息（加速推理）
    recent_messages = trim_history(history.messages, MAX_HISTORY_MESSAGES)

    # 构建完整的消息列表 = 历史消息 + 新消息
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    # 调用 Agent
    result = agent.invoke({"messages": all_messages, "search_count": 0})

    # 提取最后的 AI 回答
    ai_message = result["messages"][-1]

    # 保存到会话历史
    history.add_message(HumanMessage(content=user_input))
    history.add_message(ai_message)

    return ai_message.content


# ===== 6. 流式输出 =====
async def chat_stream(user_input: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """
    与 Agent 对话的流式输出方法（供 SSE 使用）

    实时输出 Agent 的思考过程和最终回答，
    用户不需要等到全部完成才能看到回复

    Args:
        user_input: 用户输入
        session_id: 会话 ID

    Yields:
        str: 流式输出的文本片段
    """
    agent = get_agent()

    # 获取该会话的历史消息
    history = get_session_history(session_id)

    # 智能裁剪历史消息
    recent_messages = trim_history(history.messages, MAX_HISTORY_MESSAGES)

    # 构建完整的消息列表
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    # 用于收集最终的 AI 回答
    full_response = ""
    tool_rounds = 0
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
                        # 处理 content 是列表的情况
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
        yield "\n\n⚠️ 响应超时，请稍后重试。"
    except Exception as e:
        error_msg = str(e)
        if "ReadTimeout" in error_msg or "timed out" in error_msg:
            yield "\n\n⚠️ LLM 响应超时，请稍后重试。"
        else:
            yield f"\n\n⚠️ 处理出错: {error_msg}"

    # 保存到会话历史
    if full_response:
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=full_response))
