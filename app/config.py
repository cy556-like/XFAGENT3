"""
应用配置管理
支持动态切换 LLM 模型

优化:
- [#22] 配置中心：支持运行时热更新，无需重启
"""
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# 可用的 LLM 模型列表
AVAILABLE_MODELS = [
    # DeepSeek 系列（火山引擎）
    {"id": "DeepSeek-V4-Flash", "name": "DeepSeek-V4-Flash", "desc": "DeepSeek快速版，性价比高"},
    # GLM 系列（智谱AI）
    {"id": "glm-5.1", "name": "GLM-5.1", "desc": "最新旗舰，Coding对齐Claude Opus 4.6"},
    # 豆包系列（火山引擎）
    {"id": "Doubao-Seed-2.0-pro", "name": "Doubao-Seed-2.0-Pro", "desc": "豆包旗舰，火山引擎"},
    # 千问系列（阿里云）
    {"id": "qwen3.7-plus", "name": "Qwen3.7-Plus", "desc": "千问旗舰，阿里云DashScope"},
    # MiMo系列（小米）
    {"id": "mimo-v2.5-pro", "name": "MiMo-V2.5-Pro", "desc": "小米旗舰，MiMo推理模型"},
]

# 支持图片分析的视觉模型列表
VISION_MODELS = {"glm-4v-plus", "glm-4v", "glm-4v-flash"}
# 默认视觉模型（当用户上传图片时自动切换）
DEFAULT_VISION_MODEL = "glm-4v-plus"

# 快速模型列表（用于意图路由，加速简单问题的响应）
FAST_MODELS = {"DeepSeek-V4-Flash"}

# 火山引擎模型列表（走火山引擎Ark API，包括DeepSeek和豆包）
VOLCENGINE_MODELS = {"DeepSeek-V4-Flash", "Doubao-Seed-2.0-pro"}

# DeepSeek 模型列表（兼容旧代码引用，走火山引擎API）
DEEPSEEK_MODELS = {"DeepSeek-V4-Flash"}

# 千问模型列表（走阿里云DashScope API）
QWEN_MODELS = {"qwen3.7-plus"}

# MiMo模型列表（走小米MiMo API）
MIMO_MODELS = {"mimo-v2.5-pro"}


class Settings:
    """应用配置（[#22] 支持运行时热更新）"""

    # LLM 配置（智谱AI默认）
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "glm-5.1")

    # LLM 备用配置（主Key失效时自动切换）
    LLM_API_KEY_BACKUP: str = os.getenv("LLM_API_KEY_BACKUP", "")
    LLM_BASE_URL_BACKUP: str = os.getenv("LLM_BASE_URL_BACKUP", "")

    # DeepSeek / 豆包 独立配置（火山引擎Ark）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", ""))
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")

    # 千问独立配置（阿里云DashScope）
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # MiMo独立配置（小米）
    MIMO_API_KEY: str = os.getenv("MIMO_API_KEY", "")
    MIMO_BASE_URL: str = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

    # Embedding 模型
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embedding-3")
    # [#12] Embedding 独立 API Key（如未设置则复用 LLM_API_KEY）
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", ""))
    # Embedding API Base URL（如未设置则复用 LLM_BASE_URL）
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"))

    # 应用配置
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # 数据目录
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", os.path.join(DATA_DIR, "documents"))
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", os.path.join(DATA_DIR, "chroma_db"))
    EMPLOYEES_FILE: str = os.getenv("EMPLOYEES_FILE", os.path.join(DATA_DIR, "employees.json"))

    # [#22] 配置变更回调列表
    _change_callbacks = []

    @classmethod
    def on_change(cls, callback):
        """注册配置变更回调"""
        cls._change_callbacks.append(callback)

    @classmethod
    def notify_change(cls, key: str, old_value, new_value):
        """通知配置变更"""
        for cb in cls._change_callbacks:
            try:
                cb(key, old_value, new_value)
            except Exception as e:
                logger.warning(f"配置变更回调异常: {e}")


settings = Settings()


def set_current_model(model_id: str) -> bool:
    """动态切换当前使用的模型"""
    valid_ids = [m["id"] for m in AVAILABLE_MODELS]
    if model_id in valid_ids:
        old = settings.LLM_MODEL
        settings.LLM_MODEL = model_id
        # 重置 Agent 单例，让下次对话使用新模型
        from app.agent.core import reset_agent
        reset_agent()
        # [#22] 通知配置变更
        Settings.notify_change("LLM_MODEL", old, model_id)
        logger.info(f"模型切换: {old} → {model_id}")
        return True
    return False


def get_current_model() -> str:
    """获取当前使用的模型ID"""
    return settings.LLM_MODEL
