"""
对话记忆管理模块
管理每个用户/会话的对话历史，支持多轮对话
支持文件持久化存储，重启后历史不丢失
支持多会话管理：创建、列出、删除、重命名

性能优化：
- 会话缓存自动清理：长时间不活跃的会话从内存中移除（文件持久化不受影响）
- 防止 _session_store 无限增长导致内存泄漏和后端变慢
"""
import os
import json
import uuid
import time
import logging
from collections import defaultdict
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from app.config import settings

logger = logging.getLogger(__name__)


class FileBasedHistory(BaseChatMessageHistory):
    """
    基于文件的对话历史存储
    每个会话保存为一个 JSON 文件
    重启后历史不会丢失

    优化：
    - 增加最后访问时间追踪，用于缓存清理
    - 延迟加载：只在需要时从文件读取，减少内存占用
    """

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._messages: list[BaseMessage] = []
        self._file_path = os.path.join(
            settings.DATA_DIR, "conversations", f"{session_id}.json"
        )
        self._last_access_time = time.time()  # 最后访问时间
        self._loaded = False  # 是否已加载
        self._load_from_file()

    def _load_from_file(self):
        """从文件加载历史（仅在首次访问时加载）"""
        if self._loaded:
            return
        self._loaded = True
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

    def _touch(self):
        """更新最后访问时间"""
        self._last_access_time = time.time()

    @property
    def messages(self) -> list[BaseMessage]:
        self._touch()
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._touch()
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
        self._last_access_time = time.time()

    @property
    def messages(self) -> list[BaseMessage]:
        self._last_access_time = time.time()
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._last_access_time = time.time()
        self._messages.append(message)

    def clear(self) -> None:
        self._messages = []


# 全局会话存储：session_id -> ChatMessageHistory
_session_store: dict[str, BaseChatMessageHistory] = {}

# 会话缓存清理配置
_SESSION_MAX_IDLE_SECONDS = 1800  # 30分钟不活跃的会话从内存中移除
_SESSION_MAX_COUNT = 200  # 内存中最多保留的会话数
_last_cleanup_time = 0  # 上次清理时间
_CLEANUP_INTERVAL = 300  # 每5分钟执行一次清理


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取指定会话的对话历史（文件持久化）"""
    if session_id not in _session_store:
        try:
            _session_store[session_id] = FileBasedHistory(session_id)
        except Exception:
            # 如果文件持久化失败，回退到内存存储
            _session_store[session_id] = InMemoryHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    """清除指定会话的对话历史"""
    if session_id in _session_store:
        _session_store[session_id].clear()
        del _session_store[session_id]


def get_history_messages(session_id: str) -> list[dict]:
    """获取会话历史的格式化版本（用于 API 返回）"""
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})
    return messages


def cleanup_stale_sessions(max_idle_seconds: int = None, max_count: int = None) -> int:
    """
    清理不活跃的会话缓存，防止内存泄漏

    策略：
    1. 移除超过 max_idle_seconds 未访问的会话（文件持久化不受影响，下次访问时重新加载）
    2. 如果会话数超过 max_count，移除最久未访问的会话
    3. 控制清理频率，避免频繁清理影响性能

    Args:
        max_idle_seconds: 最大空闲时间（秒），默认30分钟
        max_count: 最大缓存会话数，默认200

    Returns:
        int: 清理的会话数量
    """
    global _last_cleanup_time

    # 控制清理频率：最多每5分钟执行一次
    now = time.time()
    if now - _last_cleanup_time < _CLEANUP_INTERVAL:
        return 0

    _last_cleanup_time = now
    idle_limit = max_idle_seconds or _SESSION_MAX_IDLE_SECONDS
    count_limit = max_count or _SESSION_MAX_COUNT

    if not _session_store:
        return 0

    cleaned = 0

    # 策略1：移除空闲超时的会话
    stale_ids = []
    for sid, history in _session_store.items():
        last_access = getattr(history, '_last_access_time', now)
        if now - last_access > idle_limit:
            stale_ids.append(sid)

    for sid in stale_ids:
        del _session_store[sid]
        cleaned += 1

    if stale_ids:
        logger.info(f"会话缓存清理：移除 {len(stale_ids)} 个不活跃会话（空闲>{idle_limit}秒）")

    # 策略2：如果仍然超过数量限制，移除最久未访问的
    if len(_session_store) > count_limit:
        # 按最后访问时间排序，移除最旧的
        sorted_sessions = sorted(
            _session_store.items(),
            key=lambda x: getattr(x[1], '_last_access_time', 0)
        )
        excess = len(_session_store) - count_limit
        for sid, _ in sorted_sessions[:excess]:
            del _session_store[sid]
            cleaned += 1
        logger.info(f"会话缓存清理：移除 {excess} 个最旧会话（超过限制{count_limit}）")

    return cleaned


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


def create_chat(username: str, title: str = "新对话") -> dict:
    """
    为用户创建一个新的会话

    Returns:
        dict: 包含 chat_id 和 title
    """
    chat_id = f"{username}_{uuid.uuid4().hex[:8]}"
    chats = _load_user_chats(username)

    chat_info = {
        "chat_id": chat_id,
        "title": title,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    chats.insert(0, chat_info)  # 新会话放在最前面
    _save_user_chats(username, chats)

    return chat_info


def list_chats(username: str) -> list[dict]:
    """列出用户的所有会话，按更新时间倒序"""
    chats = _load_user_chats(username)
    # 更新每个会话的标题（取第一条用户消息的前20字）
    for chat in chats:
        chat_id = chat["chat_id"]
        history = get_session_history(chat_id)
        if history.messages and not chat.get("title_custom"):
            # 取第一条用户消息作为标题
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    title = msg.content[:30].replace("\n", " ")
                    if len(msg.content) > 30:
                        title += "..."
                    chat["title"] = title
                    break
    # 按更新时间倒序
    chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    _save_user_chats(username, chats)
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
