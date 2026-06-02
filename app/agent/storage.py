"""
智能体存储模块
- 按用户持久化智能体数据到 JSON 文件
- 支持按 agent_id 合并同步（避免覆盖）
- 数据目录: app/data/agents/{username}.json
"""
import os
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 智能体数据根目录
AGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agents")


def _ensure_dir():
    """确保智能体数据目录存在"""
    os.makedirs(AGENTS_DIR, exist_ok=True)


def _user_file(username: str) -> str:
    """获取用户的智能体数据文件路径"""
    _ensure_dir()
    return os.path.join(AGENTS_DIR, f"{username}.json")


# 允许的智能体ID白名单
ALLOWED_AGENT_IDS = {'xf-rd-agent', 'xf-quality-agent'}

def load_agents(username: str) -> list:
    """
    加载用户的智能体列表
    Returns: list of agent dicts
    """
    filepath = _user_file(username)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        agents = []
        if isinstance(data, list):
            agents = data
        elif isinstance(data, dict) and "agents" in data:
            agents = data["agents"]
        # 过滤：只保留允许的智能体ID
        return [a for a in agents if a.get("id") in ALLOWED_AGENT_IDS]
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"加载智能体数据失败 [{username}]: {e}")
        return []


def save_agents(username: str, agents: list) -> bool:
    """
    保存用户的智能体列表（全量覆盖）
    Returns: True if success
    """
    filepath = _user_file(username)
    try:
        _ensure_dir()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(agents, f, ensure_ascii=False, indent=2)
        logger.info(f"保存智能体数据 [{username}]: {len(agents)} 个智能体")
        return True
    except IOError as e:
        logger.error(f"保存智能体数据失败 [{username}]: {e}")
        return False


def sync_agents(username: str, client_agents: list) -> dict:
    """
    同步智能体：按 agent_id 合并，避免覆盖
    
    合并策略:
    1. 服务端和客户端都有的 agent → 以较新的 updated_at 为准
    2. 仅客户端有的 agent → 添加到服务端
    3. 仅服务端有的 agent → 保留在服务端
    
    Returns: {"agents": merged_list, "synced": count, "added": count, "updated": count}
    """
    server_agents = load_agents(username)
    
    # Build index by agent_id
    server_map = {a["id"]: a for a in server_agents if "id" in a}
    client_map = {a["id"]: a for a in client_agents if "id" in a}
    
    all_ids = set(server_map.keys()) | set(client_map.keys())
    merged = []
    synced = 0
    added = 0
    updated = 0
    
    for aid in all_ids:
        server_agent = server_map.get(aid)
        client_agent = client_map.get(aid)
        
        if server_agent and client_agent:
            # 两端都有：比较 updated_at，有 updated_at 的一方优先
            server_updated = server_agent.get("updated_at")
            client_updated = client_agent.get("updated_at")
            
            if server_updated and not client_updated:
                # Server was edited, use server version
                merged.append(server_agent)
            elif client_updated and not server_updated:
                # Client was edited, use client version
                merged.append(client_agent)
                updated += 1
            elif server_updated and client_updated:
                # Both edited, use the newer one
                if server_updated > client_updated:
                    merged.append(server_agent)
                else:
                    merged.append(client_agent)
                    updated += 1
            else:
                # Neither was edited after creation, use the newer created_at
                server_created = server_agent.get("created_at", 0)
                client_created = client_agent.get("created_at", 0)
                if server_created >= client_created:
                    merged.append(server_agent)
                else:
                    merged.append(client_agent)
                    updated += 1
            synced += 1
        elif client_agent:
            # 仅客户端有：添加
            merged.append(client_agent)
            added += 1
        else:
            # 仅服务端有：保留
            merged.append(server_agent)
    
    # 按 created_at 排序
    merged.sort(key=lambda a: a.get("created_at", 0), reverse=True)
    
    # 过滤：只保留允许的智能体ID
    merged = [a for a in merged if a.get("id") in ALLOWED_AGENT_IDS]
    
    # 保存合并结果
    save_agents(username, merged)
    
    logger.info(f"智能体同步 [{username}]: 合并={synced}, 新增={added}, 更新={updated}, 总计={len(merged)}")
    
    return {
        "agents": merged,
        "synced": synced,
        "added": added,
        "updated": updated,
        "total": len(merged),
    }


def delete_agent(username: str, agent_id: str) -> bool:
    """
    删除指定智能体
    Returns: True if deleted
    """
    agents = load_agents(username)
    new_agents = [a for a in agents if a.get("id") != agent_id]
    if len(new_agents) < len(agents):
        save_agents(username, new_agents)
        return True
    return False


def get_agent(username: str, agent_id: str) -> Optional[dict]:
    """
    获取指定智能体
    Returns: agent dict or None
    """
    agents = load_agents(username)
    for a in agents:
        if a.get("id") == agent_id:
            return a
    return None


def debug_info(username: str) -> dict:
    """
    获取智能体诊断信息
    """
    filepath = _user_file(username)
    agents = load_agents(username)
    file_exists = os.path.exists(filepath)
    file_size = os.path.getsize(filepath) if file_exists else 0
    file_mtime = os.path.getmtime(filepath) if file_exists else 0
    
    return {
        "username": username,
        "file_path": filepath,
        "file_exists": file_exists,
        "file_size_bytes": file_size,
        "file_modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_mtime)) if file_exists else None,
        "agent_count": len(agents),
        "agent_ids": [a.get("id", "?") for a in agents],
        "agent_names": [a.get("name", "?") for a in agents],
        "agents": agents,
    }
