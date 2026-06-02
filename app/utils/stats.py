"""
使用统计模块
记录和查询系统使用统计数据
"""
import os
import json
import time
from collections import defaultdict

from app.config import settings


def _get_stats_file() -> str:
    """获取统计数据文件路径"""
    return os.path.join(settings.DATA_DIR, "stats.json")


def _load_stats() -> dict:
    """加载统计数据"""
    stats_file = _get_stats_file()
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 默认结构
    return {
        "total_messages": 0,
        "total_sessions": 0,
        "daily_stats": {},  # "2024-01-01": {"messages": N, "sessions": N, "users": []}
        "tool_usage": {},   # "tool_name": count
        "model_usage": {},  # "model_id": count
    }


def _save_stats(stats: dict) -> None:
    """保存统计数据"""
    stats_file = _get_stats_file()
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def record_message(username: str = None, model_id: str = None, tools_used: list = None) -> None:
    """记录一条消息"""
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

        # 模型使用（修复：原代码 model_usageodel_id] 缺少 [m，改为 model_usage[model_id]）
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

        _save_stats(stats)
    except Exception:
        pass  # 统计不应影响正常功能


def record_session() -> None:
    """记录一个新会话"""
    try:
        stats = _load_stats()
        stats["total_sessions"] = stats.get("total_sessions", 0) + 1
        _save_stats(stats)
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
