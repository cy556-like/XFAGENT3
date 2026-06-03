"""
FastAPI 应用主入口
启动 API 服务 + 美化前端界面
"""
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse

from app.config import settings
from app.api.routes import router


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="企业文档智能助手 Agent",
        description="""
        基于 LangChain + LangGraph 的企业文档智能助手

        ## 功能
        - 文档问答：上传公司文档，AI 自动回答相关问题
        - 员工查询：查询员工信息、部门归属
        - 文档搜索：语义搜索文档内容
        - 文档修改：上传文档+修改要求，AI 修改后返回文件
        - 多轮对话：支持上下文连续对话

        ## 技术栈
        - LangChain + LangGraph (ReAct Agent)
        - ChromaDB (向量数据库)
        - FastAPI (后端服务)
        """,
        version="2.0.0",
    )

    # CORS 跨域支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册 API 路由
    app.include_router(router, prefix="/api/v1", tags=["Agent API"])

    # 确保静态文件目录存在
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    # 确保下载文件目录存在（修改后的文档）
    modified_dir = os.path.join(static_dir, "modified")
    os.makedirs(modified_dir, exist_ok=True)

    # 确保导出文件目录存在（Agent 导出的文档）
    exports_dir = os.path.join(static_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)

    # 挂载静态文件目录（包含 index.html 和修改后的文档）
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # 根路径重定向到美化的前端页面
    @app.get("/")
    async def root():
        return FileResponse(os.path.join(static_dir, "index.html"))

    # 健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "company-doc-agent"}

    return app


app = create_app()


if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)

    # 确保对话历史目录存在
    conversations_dir = os.path.join(settings.DATA_DIR, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)

    # 确保用户数据目录存在
    os.makedirs(os.path.join(settings.DATA_DIR, "users"), exist_ok=True)

    # 确保临时文件目录存在
    os.makedirs(os.path.join(settings.DATA_DIR, "temp"), exist_ok=True)

    print("=" * 50)
    print("企业文档智能助手 Agent 启动中...")
    print(f"  前端界面: http://localhost:{settings.APP_PORT}")
    print(f"  API 地址: http://localhost:{settings.APP_PORT}/api/v1")
    print(f"  API 文档: http://localhost:{settings.APP_PORT}/docs")
    print("=" * 50)

    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
    )
