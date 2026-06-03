"""
对话记忆管理模块
管理每个用户/会话的对话历史，支持多轮对话
支持文件持久化存储，重启后历史不丢失
支持多会话管理：创建、列出、删除、重命名

性能优化:
- 会话存储LRU淘汰：_session_store 限制最大数量，避免内存无限增长
- 自动清理：长时间未访问的会话从内存中移除（文件持久化不受影响）
"""
import os
import json
import uuid
import time
import logging
from collections import OrderedDict
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from app.config import settings

logger = logging.getLogger(__name__)

# ===== 会话存储上限（防止内存泄漏）=====
# 超过此数量的会话将按 LRU 策略淘汰最久未访问的
# 淘汰仅释放内存中的对象，不影响文件持久化（下次访问时自动重新加载）
MAX_SESSION_STORE_SIZE = 200

# 会话最大空闲时间（秒），超过此时间未访问的会话从内存移除
# 2小时 = 7200秒
SESSION_MAX_IDLE_SECONDS = 7200


class FileBasedHistory(BaseChatMessageHistory):
    """
    基于文件的对话历史存储
    每个会话保存为一个 JSON 文件
    重启后历史不会丢失
    """

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._messages: list[BaseMessage] = []
        self._file_path = os.path.join(
            settings.DATA_DIR, "conversations", f"{session_id}.json"
        )
        self._load_from_file()

    def _load_from_file(self):
        """从文件加载历史"""
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for msg_data in data:
                    if msg_data["role"] == "user":
                        self._messages.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data["role"] == "assistant":
                        self._messages.append(AIMessage(content=msg_data["content"]))
            except Exception:
                self._messages = []

    def _save_to_file(self):
        """保存历史到文件"""
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        data = []
        for msg in self._messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            data.append({"role": role, "content": msg.content})
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._save_to_file()

    def clear(self) -> None:
        self._messages = []
        if os.path.exists(self._file_path):
            os.remove(self._file_path)


class InMemoryHistory(BaseChatMessageHistory):
    """
    基于内存的对话历史存储（后备方案）
    """

    def __init__(self):
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages = []


# 全局会话存储：session_id -> (ChatMessageHistory, last_access_time)
# 使用 OrderedDict 实现 LRU 淘汰，避免长时间运行后内存无限增长
_session_store: OrderedDict[str, tuple] = OrderedDict()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取指定会话的对话历史（文件持久化 + LRU淘汰）
    
    LRU策略：每次访问将会话移到OrderedDict末尾，
    超过 MAX_SESSION_STORE_SIZE 时淘汰最久未访问的会话。
    淘汰仅释放内存，不影响磁盘文件（下次访问时自动重新加载）。
    """
    if session_id in _session_store:
        # LRU: 移到末尾（最近访问）
        _session_store.move_to_end(session_id)
        history, _ = _session_store[session_id]
        _session_store[session_id] = (history, time.time())
        return history
    
    # 新会话：加载文件
    try:
        history = FileBasedHistory(session_id)
    except Exception:
        history = InMemoryHistory()
    
    _session_store[session_id] = (history, time.time())
    
    # LRU淘汰：超过上限时移除最久未访问的
    while len(_session_store) > MAX_SESSION_STORE_SIZE:
        oldest_id, _ = _session_store.popitem(last=False)
        logger.debug(f"LRU淘汰会话: {oldest_id}（内存释放，文件保留）")
    
    return history


def clear_session_history(session_id: str) -> None:
    """清除指定会话的对话历史"""
    if session_id in _session_store:
        history, _ = _session_store[session_id]
        history.clear()
        del _session_store[session_id]


def get_history_messages(session_id: str) -> list[dict]:
    """获取会话历史的格式化版本（用于 API 返回）"""
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})
    return messages


def cleanup_idle_sessions() -> int:
    """清理长时间未访问的会话（释放内存，不影响文件持久化）
    
    定期调用此函数，将超过 SESSION_MAX_IDLE_SECONDS 未访问的会话从内存中移除。
    下次访问时自动从文件重新加载。
    
    Returns:
        int: 被清理的会话数量
    """
    now = time.time()
    to_remove = []
    for sid, (history, last_access) in _session_store.items():
        if now - last_access > SESSION_MAX_IDLE_SECONDS:
            to_remove.append(sid)
    
    for sid in to_remove:
        del _session_store[sid]
    
    if to_remove:
        logger.info(f"清理了 {len(to_remove)} 个空闲超过 {SESSION_MAX_IDLE_SECONDS}s 的会话（内存释放，文件保留）")
    
    return len(to_remove)


# ===== 多会话管理 =====

def _get_user_chats_file(username: str) -> str:
    """获取用户的会话索引文件路径"""
    return os.path.join(settings.DATA_DIR, "users", f"{username}_chats.json")


def _load_user_chats(username: str) -> list[dict]:
    """加载用户的会话列表"""
    chats_file = _get_user_chats_file(username)
    if os.path.exists(chats_file):
        try:
            with open(chats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_user_chats(username: str, chats: list[dict]) -> None:
    """保存用户的会话列表"""
    chats_file = _get_user_chats_file(username)
    os.makedirs(os.path.dirname(chats_file), exist_ok=True)
    with open(chats_file, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


def create_chat(username: str, title: str = "新对话", mode: str = "agent", agent_id: str = None) -> dict:
    """
    为用户创建一个新的会话

    Args:
        mode: 会话模式 "agent" 或 "chat"，用于隔离不同模式的对话列表

    Returns:
        dict: 包含 chat_id、title 和 mode
    """
    chat_id = f"{username}_{uuid.uuid4().hex[:8]}"
    chats = _load_user_chats(username)

    chat_info = {
        "chat_id": chat_id,
        "title": title,
        "mode": mode,
        "agent_id": agent_id or "",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    chats.insert(0, chat_info)  # 新会话放在最前面
    _save_user_chats(username, chats)

    return chat_info


def list_chats(username: str, mode: str = None, skip_auto_title: bool = False) -> list[dict]:
    """列出用户的会话，按更新时间倒序

    Args:
        mode: 可选，按模式过滤 "agent" 或 "chat"，为 None 则返回全部
        skip_auto_title: 跳过自动标题更新（GET请求时为True，避免读副作用）
    """
    chats = _load_user_chats(username)

    # 自动更新标题：仅在需要时执行（非GET请求或显式请求时）
    if not skip_auto_title:
        updated = False
        for chat in chats:
            chat_id = chat["chat_id"]
            if not chat.get("title_custom"):
                history = get_session_history(chat_id)
                if history.messages:
                    for msg in history.messages:
                        if isinstance(msg, HumanMessage):
                            title = msg.content[:30].replace("\n", " ")
                            if len(msg.content) > 30:
                                title += "..."
                            chat["title"] = title
                            updated = True
                            break
        # 只有标题确实变更了才写回
        if updated:
            chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            _save_user_chats(username, chats)
            if mode:
                chats = [c for c in chats if c.get("mode", "agent") == mode]
            return chats

    # 按更新时间倒序（不写回文件）
    chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    # 按 mode 过滤（如果指定了 mode）
    if mode:
        # 兼容旧数据：没有 mode 字段的会话默认归为 "agent"
        chats = [c for c in chats if c.get("mode", "agent") == mode]
    return chats


def delete_chat(username: str, chat_id: str) -> bool:
    """删除用户的某个会话"""
    chats = _load_user_chats(username)
    chats = [c for c in chats if c["chat_id"] != chat_id]
    _save_user_chats(username, chats)
    # 同时清除对话历史文件
    clear_session_history(chat_id)
    return True


def rename_chat(username: str, chat_id: str, new_title: str) -> bool:
    """重命名用户的某个会话"""
    chats = _load_user_chats(username)
    for chat in chats:
        if chat["chat_id"] == chat_id:
            chat["title"] = new_title
            chat["title_custom"] = True
            chat["updated_at"] = time.time()
            break
    _save_user_chats(username, chats)
    return True


def update_chat_time(username: str, chat_id: str) -> None:
    """更新会话的更新时间（发送消息时调用）"""
    chats = _load_user_chats(username)
    for chat in chats:
        if chat["chat_id"] == chat_id:
            chat["updated_at"] = time.time()
            # 自动更新标题（取第一条用户消息）
            if not chat.get("title_custom"):
                history = get_session_history(chat_id)
                for msg in history.messages:
                    if isinstance(msg, HumanMessage):
                        title = msg.content[:30].replace("\n", " ")
                        if len(msg.content) > 30:
                            title += "..."
                        chat["title"] = title
                        break
            break
    _save_user_chats(username, chats)
