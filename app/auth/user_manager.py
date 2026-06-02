"""
用户认证管理模块
支持用户注册、登录验证，密码使用 bcrypt 加盐加密存储
"""
import os
import json
import hashlib

from app.config import settings


def _get_users_file() -> str:
    """获取用户数据文件路径"""
    return os.path.join(settings.DATA_DIR, "users", "users.json")


def _load_users() -> dict:
    """加载用户数据"""
    users_file = _get_users_file()
    if os.path.exists(users_file):
        with open(users_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认管理员账号（bcrypt hash of "admin123"）
    default_users = {
        "admin": {
            "password_hash": _hash_password("admin123"),
        }
    }
    _save_users(default_users)
    return default_users


def _save_users(users: dict) -> None:
    """保存用户数据"""
    users_file = _get_users_file()
    os.makedirs(os.path.dirname(users_file), exist_ok=True)
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str) -> str:
    """密码哈希 - 使用 bcrypt 加盐加密"""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        # fallback: 如果 bcrypt 未安装，使用 SHA256（不推荐）
        return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(password: str, password_hash: str) -> bool:
    """验证密码"""
    try:
        import bcrypt
        # 尝试 bcrypt 验证
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ImportError, ValueError):
        # fallback: SHA256 验证（兼容旧密码）
        return password_hash == hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str) -> dict:
    """
    注册新用户

    Returns:
        dict: {"success": bool, "message": str}
    """
    if len(username) < 2:
        return {"success": False, "message": "用户名至少2个字符"}
    if len(password) < 4:
        return {"success": False, "message": "密码至少4个字符"}

    users = _load_users()
    if username in users:
        return {"success": False, "message": "用户名已存在"}

    users[username] = {
        "password_hash": _hash_password(password),
    }
    _save_users(users)
    return {"success": True, "message": "注册成功，请登录"}


def login_user(username: str, password: str) -> dict:
    """
    用户登录验证

    Returns:
        dict: {"success": bool, "message": str}
    """
    users = _load_users()
    if username not in users:
        return {"success": False, "message": "用户名或密码错误"}

    if not _verify_password(password, users[username]["password_hash"]):
        return {"success": False, "message": "用户名或密码错误"}

    # 如果密码是旧的 SHA256 格式，自动升级为 bcrypt
    if not users[username]["password_hash"].startswith("$2"):
        users[username]["password_hash"] = _hash_password(password)
        _save_users(users)

    return {"success": True, "message": "登录成功"}
