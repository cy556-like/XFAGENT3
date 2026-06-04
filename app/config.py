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
    {"id": "ep-20260604175402-5665h", "name": "DeepSeek-V4-Pro", "desc": "DeepSeek旗舰，128K上下文"},
    # GLM-5 系列（最新）
    {"id": "glm-5.1", "name": "GLM-5.1", "desc": "最新旗舰，Coding对齐Claude Opus 4.6"},
    {"id": "glm-5-turbo", "name": "GLM-5-Turbo", "desc": "高智能基座，Agent能力SOTA"},
    {"id": "glm-5", "name": "GLM-5", "desc": "高智能基座，编程对齐Claude Opus 4.5"},
    # GLM-4.7 系列
    {"id": "glm-4.7", "name": "GLM-4.7", "desc": "高性能，综合能力提升"},
    {"id": "glm-4.7-flash", "name": "GLM-4.7-Flash", "desc": "快速版，性价比高"},
    # GLM-4V 视觉系列（支持图片分析）
    {"id": "glm-4v-plus", "name": "GLM-4V-Plus", "desc": "视觉旗舰，图片分析首选"},
    {"id": "glm-4v", "name": "GLM-4V", "desc": "视觉模型，支持图片理解"},
    # GLM-4 系列（经典）
    {"id": "glm-4-plus", "name": "GLM-4-Plus", "desc": "高性能，复杂任务首选"},
    {"id": "glm-4-long", "name": "GLM-4-Long", "desc": "超长上下文，支持128K"},
    {"id": "glm-4-flash", "name": "GLM-4-Flash", "desc": "最快，适合日常对话"},
    {"id": "glm-4-air", "name": "GLM-4-Air", "desc": "均衡，速度与质量兼顾"},
    {"id": "glm-4-air-0111", "name": "GLM-4-Air-0111", "desc": "Air升级版，效果更好"},
    {"id": "glm-4", "name": "GLM-4", "desc": "经典旗舰模型"},
]

# 支持图片分析的视觉模型列表
VISION_MODELS = {"glm-4v-plus", "glm-4v", "glm-4v-flash"}
# 默认视觉模型（当用户上传图片时自动切换）
DEFAULT_VISION_MODEL = "glm-4v-plus"

# 快速模型列表（用于意图路由，加速简单问题的响应）
FAST_MODELS = {"glm-4-flash", "glm-4-air", "glm-4-air-0111", "glm-4.7-flash"}

# DeepSeek 模型列表（走火山引擎API）
DEEPSEEK_MODELS = {"ep-20260604175402-5665h"}


class Settings:
    """应用配置（[#22] 支持运行时热更新）"""

    # LLM 配置
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.xixixixi.cloud/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "glm-5-turbo")

    # LLM 备用配置（主Key失效时自动切换）
    LLM_API_KEY_BACKUP: str = os.getenv("LLM_API_KEY_BACKUP", "")
    LLM_BASE_URL_BACKUP: str = os.getenv("LLM_BASE_URL_BACKUP", "")

    # DeepSeek 独立配置（火山引擎Ark）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", ""))
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

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
