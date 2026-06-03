"""
使用统计模块
记录和查询系统使用统计数据

[性能修复] 统计写入防抖：避免每条消息都触发stats.json磁盘读写
改用内存缓存+延迟写入，5秒内的多次record_message只触发一次磁盘写入
"""
import os
import json
import time
import threading
from collections import defaultdict

from app.config import settings

# [性能修复] 统计写入防抖配置
_stats_dirty = False
_stats_save_timer = None
_stats_lock = threading.Lock()
_stats_save_interval = 5.0  # 至少间隔5秒才写一次磁盘


def _get_stats_file() -> str:
    """获取统计数据文件路径"""
    return os.path.join(settings.DATA_DIR, "stats.json")


# [性能修复] 内存缓存，避免每次都读磁盘
_stats_cache = None


def _load_stats() -> dict:
    """加载统计数据（优先使用内存缓存）"""
    global _stats_cache
    if _stats_cache is not None:
        return _stats_cache
    
    stats_file = _get_stats_file()
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                _stats_cache = json.load(f)
                return _stats_cache
        except Exception:
            pass

    # 默认结构
    _stats_cache = {
        "total_messages": 0,
        "total_sessions": 0,
        "daily_stats": {},  # "2024-01-01": {"messages": N, "sessions": N, "users": []}
        "tool_usage": {},   # "tool_name": count
        "model_usage": {},  # "model_id": count
    }
    return _stats_cache


def _save_stats(stats: dict) -> None:
    """保存统计数据（仅更新缓存，延迟写入磁盘）"""
    global _stats_dirty
    _stats_dirty = True
    _schedule_stats_save()


def _schedule_stats_save():
    """防抖：延迟5秒写入磁盘，5秒内的多次调用只触发一次写入"""
    global _stats_save_timer
    if _stats_save_timer is not None:
        _stats_save_timer.cancel()
    _stats_save_timer = threading.Timer(_stats_save_interval, _flush_stats_to_disk)
    _stats_save_timer.daemon = True
    _stats_save_timer.start()


def _flush_stats_to_disk():
    """将缓存中的统计数据写入磁盘"""
    global _stats_dirty, _stats_save_timer
    _stats_save_timer = None
    if not _stats_dirty or _stats_cache is None:
        return
    with _stats_lock:
        try:
            stats_file = _get_stats_file()
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(_stats_cache, f, ensure_ascii=False, indent=2)
            _stats_dirty = False
        except Exception:
            pass


def flush_stats():
    """强制将统计数据刷到磁盘（服务关闭或定期清理时调用）"""
    global _stats_save_timer
    if _stats_save_timer is not None:
        _stats_save_timer.cancel()
        _stats_save_timer = None
    _flush_stats_to_disk()


def record_message(username: str = None, model_id: str = None, tools_used: list = None) -> None:
    """记录一条消息（内存缓存+防抖写入）"""
    try:
        stats = _load_stats()
        stats["total_messages"] = stats.get("total_messages", 0) + 1

        # 每日统计
        today = time.strftime("%Y-%m-%d")
        daily = stats["daily_stats"]
        if today not in daily:
            daily[today] = {"messages": 0, "users": []}
        daily[today]["messages"] = daily[today].get("messages", 0) + 1
        if username and username not in daily[today].get("users", []):
            users_list = daily[today].get("users", [])
            users_list.append(username)
            daily[today]["users"] = users_list

        # 模型使用
        if model_id:
            model_usage = stats.get("model_usage", {})
            model_usage[model_id] = model_usage.get(model_id, 0) + 1
            stats["model_usage"] = model_usage

        # 工具使用
        if tools_used:
            tool_usage = stats.get("tool_usage", {})
            for tool in tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            stats["tool_usage"] = tool_usage

        # 只保留最近30天的日统计
        all_days = sorted(daily.keys())
        if len(all_days) > 30:
            for old_day in all_days[:-30]:
                del daily[old_day]

        _save_stats(stats)  # [性能修复] 只更新缓存+调度延迟写入
    except Exception:
        pass  # 统计不应影响正常功能


def record_session() -> None:
    """记录一个新会话"""
    try:
        stats = _load_stats()
        stats["total_sessions"] = stats.get("total_sessions", 0) + 1
        _save_stats(stats)  # [性能修复] 只更新缓存+调度延迟写入
    except Exception:
        pass


def get_stats() -> dict:
    """获取统计数据"""
    stats = _load_stats()

    # 计算今日统计
    today = time.strftime("%Y-%m-%d")
    daily = stats.get("daily_stats", {})
    today_data = daily.get(today, {"messages": 0, "users": []})

    # 活跃用户数（最近7天）
    active_users = set()
    all_days = sorted(daily.keys())
    for day in all_days[-7:]:
        if day in daily:
            active_users.update(daily[day].get("users", []))

    # 最近7天消息趋势
    recent_7d = []
    for day in all_days[-7:]:
        recent_7d.append({
            "date": day,
            "messages": daily.get(day, {}).get("messages", 0),
        })

    return {
        "total_messages": stats.get("total_messages", 0),
        "total_sessions": stats.get("total_sessions", 0),
        "today_messages": today_data.get("messages", 0),
        "today_users": len(today_data.get("users", [])),
        "active_users_7d": len(active_users),
        "tool_usage": stats.get("tool_usage", {}),
        "model_usage": stats.get("model_usage", {}),
        "recent_7d": recent_7d,
    }
