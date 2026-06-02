"""
JWT Token 认证处理模块
支持 Token 签发、验证、刷新
"""
import os
import time
import hashlib
from typing import Optional

from app.config import settings


# JWT Secret：优先从环境变量读取，否则自动生成
_JWT_SECRET = os.getenv("JWT_SECRET", "")
if not _JWT_SECRET:
    # 自动生成一个稳定的 secret（基于 DATA_DIR 路径的哈希）
    _JWT_SECRET = hashlib.sha256(f"docagent-{settings.DATA_DIR}".encode()).hexdigest()

_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_SECONDS = 86400 * 7  # 7 天过期


def _base64url_encode(data: bytes) -> str:
    """Base64url 编码（无填充）"""
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _base64url_decode(s: str) -> bytes:
    """Base64url 解码"""
    import base64
    padding = 4 - len(s) % 4
    if padding != 4:
        s += '=' * padding
    return base64.urlsafe_b64decode(s)


def _hmac_sign(payload: str) -> str:
    """HMAC-SHA256 签名"""
    import hmac
    sig = hmac.new(_JWT_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).digest()
    return _base64url_encode(sig)


def create_token(username: str, expire_seconds: int = None) -> str:
    """
    创建 JWT Token

    Args:
        username: 用户名
        expire_seconds: 过期时间（秒），默认7天

    Returns:
        JWT Token 字符串
    """
    if expire_seconds is None:
        expire_seconds = _JWT_EXPIRE_SECONDS

    header = _base64url_encode(
        __import__('json').dumps({"alg": "HS256", "typ": "JWT"}).encode('utf-8')
    )

    now = int(time.time())
    payload_data = {
        "sub": username,
        "iat": now,
        "exp": now + expire_seconds,
    }
    payload = _base64url_encode(
        __import__('json').dumps(payload_data, separators=(',', ':')).encode('utf-8')
    )

    signature = _hmac_sign(f"{header}.{payload}")

    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> Optional[dict]:
    """
    验证 JWT Token

    Args:
        token: JWT Token 字符串

    Returns:
        验证成功返回 payload dict，失败返回 None
    """
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None

        header, payload, signature = parts

        # 验证签名
        expected_sig = _hmac_sign(f"{header}.{payload}")
        if signature != expected_sig:
            return None

        # 解码 payload
        import json
        payload_data = json.loads(_base64url_decode(payload))

        # 检查过期
        if payload_data.get("exp", 0) < int(time.time()):
            return None

        return payload_data

    except Exception:
        return None


def get_username_from_token(token: str) -> Optional[str]:
    """从 Token 中获取用户名"""
    payload = verify_token(token)
    if payload:
        return payload.get("sub")
    return None
