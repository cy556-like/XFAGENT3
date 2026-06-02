"""
配置管理模块
从环境变量 / .env 文件中加载配置
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """全局配置"""

    # LLM 配置
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "glm-4-flash")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))  # LLM 请求超时(秒)，默认120秒

    # Embedding 模型配置（智谱: embedding-3, OpenAI: text-embedding-v3）
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embedding-3")

    # 应用配置
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # 数据目录
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    DOCUMENTS_DIR: str = os.path.join(DATA_DIR, "documents")
    CHROMA_DIR: str = os.path.join(DATA_DIR, "chroma_db")
    EMPLOYEES_FILE: str = os.path.join(DATA_DIR, "employees.json")


settings = Settings()
