"""
FastAPI 应用主入口
启动 API 服务 + 美化前端界面

优化:
- [#20] 可观测性：structured logging + 请求日志中间件
- [#24] 健康检查增强：检查 ChromaDB/LLM API/磁盘等依赖
- [#25] 优雅关闭：graceful shutdown 处理流式连接
"""
import os
import sys
import time
import signal
import logging

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse

from app.config import settings
from app.api.routes import router

# ===== [#25] 优雅关闭状态 =====
_shutdown_requested = False
_active_connections = 0


def is_shutting_down() -> bool:
    """检查是否正在关闭"""
    return _shutdown_requested


# ===== [#20] 可观测性：structured logging =====
class StructuredFormatter(logging.Formatter):
    """结构化日志格式，输出 JSON 格式的日志条目"""
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 附加额外字段
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "user"):
            log_entry["user"] = record.user
        
        # 异常信息
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 简单格式化（不用 json.dumps，保持可读性）
        base = f"{log_entry['timestamp']} [{log_entry['level']}] {log_entry['logger']}: {log_entry['message']}"
        if "duration_ms" in log_entry:
            base += f" ({log_entry['duration_ms']}ms)"
        if "user" in log_entry:
            base += f" [user={log_entry['user']}]"
        return base


def setup_logging():
    """配置全局日志（[#20] 结构化日志）"""
    log_dir = os.path.join(settings.DATA_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    formatter = StructuredFormatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件输出
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'app.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
    )
    
    # 设置第三方库日志级别
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    logger = logging.getLogger('app')
    logger.info("日志系统已初始化 (structured logging)")
    return logger


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="企业文档智能助手 Agent",
        description="""
        基于 LangChain + LangGraph 的企业文档智能助手

        ## 功能
        - 文档问答：上传公司文档，AI 自动回答相关问题
        - 员工查询：查询员工信息、部门归属
        - 文档搜索：混合检索（向量 + 关键词 + 重排序）
        - GitHub 操作：读取/更新 GitHub 仓库文件
        - 邮件发送：发送电子邮件通知
        - 数据库查询：执行 SQL 只读查询

        ## 技术栈
        - LangChain + LangGraph (ReAct Agent)
        - ChromaDB (向量数据库 + 混合检索)
        - FastAPI (后端服务)
        """,
        version="4.0.0",
    )

    # CORS 跨域支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # [#20] 请求日志中间件：记录每个请求的耗时、状态码、用户
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        global _active_connections
        
        # 跳过静态文件和健康检查的详细日志
        path = request.url.path
        skip_paths = ["/static/", "/favicon", "/health"]
        should_log = not any(path.startswith(p) for p in skip_paths)
        
        start_time = time.time()
        _active_connections += 1
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            if should_log:
                # 提取用户信息
                auth_header = request.headers.get("Authorization", "")
                user = "anonymous"
                if auth_header.startswith("Bearer "):
                    from app.auth.jwt_handler import get_username_from_token
                    uname = get_username_from_token(auth_header[7:])
                    if uname:
                        user = uname
                
                logger = logging.getLogger("app.request")
                logger.info(
                    f"{request.method} {path} → {response.status_code}",
                    extra={
                        "duration_ms": round(duration * 1000, 2),
                        "user": user,
                    }
                )
            
            return response
        finally:
            _active_connections -= 1

    # 注册 API 路由
    app.include_router(router, prefix="/api/v1", tags=["Agent API"])

    # 确保静态文件目录存在
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    # 确保子目录存在
    css_dir = os.path.join(static_dir, "css")
    js_dir = os.path.join(static_dir, "js")
    os.makedirs(css_dir, exist_ok=True)
    os.makedirs(js_dir, exist_ok=True)

    # 确保下载文件目录存在（修改后的文档）
    modified_dir = os.path.join(static_dir, "modified")
    os.makedirs(modified_dir, exist_ok=True)

    # 挂载静态文件目录（包含 index.html、CSS、JS 和修改后的文档）
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # 根路径重定向到美化的前端页面
    @app.get("/")
    async def root():
        return FileResponse(os.path.join(static_dir, "index.html"))

    # 健康检查（基础版）
    @app.get("/health")
    async def health():
        # 增加内存使用信息，方便监控
        mem_info = {}
        try:
            from app.memory.manager import _session_store
            mem_info["session_cache_size"] = len(_session_store)
            mem_info["session_cache_max"] = 200
        except Exception:
            pass
        return {
            "status": "ok" if not _shutdown_requested else "shutting_down",
            "service": "company-doc-agent",
            "version": "4.0.0",
            "active_connections": _active_connections,
            "memory": mem_info,
        }

    # [#25] 优雅关闭：处理 SIGTERM / SIGINT
    @app.on_event("shutdown")
    async def shutdown_event():
        global _shutdown_requested
        _shutdown_requested = True
        logger = logging.getLogger("app")
        logger.info(f"收到关闭信号，等待 {_active_connections} 个活跃连接完成...")

    # ===== 后台定期清理任务：防止内存泄漏 =====
    @app.on_event("startup")
    async def startup_cleanup_task():
        """启动后台定期清理任务，防止长时间运行后内存/磁盘无限增长"""
        import asyncio
        
        async def _periodic_cleanup():
            """每10分钟执行一次清理：
            1. 清理长时间未访问的会话（释放内存）
            2. 清理过期的导出文件（释放磁盘）
            3. 清理请求统计中的旧端点数据
            """
            while True:
                try:
                    await asyncio.sleep(600)  # 10分钟执行一次
                    if _shutdown_requested:
                        break
                    
                    logger = logging.getLogger("app.cleanup")
                    
                    # 1. 清理空闲会话（释放内存，文件持久化不受影响）
                    try:
                        from app.memory.manager import cleanup_idle_sessions
                        cleaned = cleanup_idle_sessions()
                        if cleaned > 0:
                            logger.info(f"[定期清理] 释放了 {cleaned} 个空闲会话的内存")
                    except Exception as e:
                        logger.warning(f"[定期清理] 会话清理失败: {e}")
                    
                    # 2. 清理过期的导出文件（超过24小时的）
                    try:
                        from app.rag.document import cleanup_export_files
                        export_dir = os.path.join(settings.DATA_DIR, "export")
                        if os.path.exists(export_dir):
                            cleaned_files = 0
                            now = time.time()
                            max_age = 86400  # 24小时
                            for item in os.listdir(export_dir):
                                sub_dir = os.path.join(export_dir, item)
                                if os.path.isdir(sub_dir):
                                    # 检查目录修改时间
                                    try:
                                        mtime = os.path.getmtime(sub_dir)
                                        if now - mtime > max_age:
                                            import shutil
                                            shutil.rmtree(sub_dir)
                                            cleaned_files += 1
                                    except Exception:
                                        pass
                            if cleaned_files > 0:
                                logger.info(f"[定期清理] 清理了 {cleaned_files} 个过期导出目录（>24h）")
                    except Exception as e:
                        logger.warning(f"[定期清理] 导出文件清理失败: {e}")
                    
                    # 3. 记录当前内存状态
                    try:
                        from app.memory.manager import _session_store
                        logger.info(f"[定期清理] 会话缓存: {len(_session_store)}/{200}, 活跃连接: {_active_connections}")
                    except Exception:
                        pass
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.getLogger("app.cleanup").warning(f"[定期清理] 异常: {e}")
        
        asyncio.create_task(_periodic_cleanup(), name="periodic_cleanup")

    return app


app = create_app()


if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)

    # 确保导出文件目录存在（AI生成的文档下载目录）
    export_dir = os.path.join(settings.DATA_DIR, "export")
    os.makedirs(export_dir, exist_ok=True)

    # 确保对话历史目录存在
    conversations_dir = os.path.join(settings.DATA_DIR, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)

    # 确保用户数据目录存在
    os.makedirs(os.path.join(settings.DATA_DIR, "users"), exist_ok=True)

    # 确保临时文件目录存在
    os.makedirs(os.path.join(settings.DATA_DIR, "temp"), exist_ok=True)

    # 确保统计数据目录存在
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    # 确保智能体数据目录存在
    os.makedirs(os.path.join(settings.DATA_DIR, "agents"), exist_ok=True)

    # 初始化日志
    logger = setup_logging()

    # [优化4] 启动预热：后台完全异步预初始化，零阻塞服务启动
    # 关键改进：不 join 等待预热线程，服务立即启动
    # 预热在后台默默进行，如果用户在预热完成前就发了请求，
    # 单例模式的 get_embeddings/get_vector_store/create_llm 会自动处理（等预热完或自行初始化）
    # 即使预热全部失败也不影响服务运行，首次请求时再按需初始化
    def _warmup_background():
        """后台预热线程：预初始化 LLM Client / Embedding / ChromaDB（零阻塞）"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        def _do_warmup():
            logger.info("[预热] 后台预热启动（零阻塞模式，服务立即可用）...")
            start = time.time()

            # 预热1: Embedding 模型（独立超时15秒，与 request_timeout 一致）
            try:
                from app.rag.document import get_embeddings, get_vector_store
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(get_embeddings)
                    try:
                        emb = future.result(timeout=15)
                    except FuturesTimeout:
                        logger.warning("[预热] Embedding 初始化超时(15s)，跳过，首次请求时按需初始化")
                        emb = None

                if emb is not None:
                    logger.info("[预热] Embedding 模型已初始化")
                    # 预热2: ChromaDB 向量数据库（独立超时15秒）
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(get_vector_store)
                            try:
                                vs = future.result(timeout=15)
                            except FuturesTimeout:
                                logger.warning("[预热] ChromaDB 初始化超时(15s)，跳过")
                                vs = None

                        if vs is not None:
                            logger.info("[预热] ChromaDB 向量数据库已初始化")
                        else:
                            logger.warning("[预热] ChromaDB 初始化失败（可能是空数据库），将在首次请求时重试")
                    except Exception as e:
                        logger.warning(f"[预热] ChromaDB 初始化异常（不影响运行）: {e}")
                else:
                    logger.warning("[预热] Embedding 不可用，系统将以关键词模式运行")
            except Exception as e:
                logger.warning(f"[预热] Embedding 初始化异常（不影响运行）: {e}")

            # 预热3: LLM Client（独立超时15秒）
            try:
                from app.agent.core import create_llm
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(create_llm)
                    try:
                        future.result(timeout=15)
                        logger.info("[预热] LLM Client 已初始化")
                    except FuturesTimeout:
                        logger.warning("[预热] LLM Client 初始化超时(15s)，跳过，首次请求时按需初始化")
            except Exception as e:
                logger.warning(f"[预热] LLM Client 初始化异常（不影响运行）: {e}")

            elapsed = time.time() - start
            logger.info(f"[预热] 后台预热完成，耗时 {elapsed:.1f}s")

        warmup_thread = threading.Thread(target=_do_warmup, name="warmup", daemon=True)
        warmup_thread.start()
        # 关键：不 join！服务立即启动，预热在后台默默进行
        # 这样无论预热耗时多久，都不影响服务启动速度和用户首次访问
        logger.info("[预热] 预热线程已启动，服务立即可用（预热在后台异步进行）")

    _warmup_background()

    print("=" * 50)
    print("企业文档智能助手 Agent v4.0.0 启动中...")
    print(f"  前端界面: http://localhost:{settings.APP_PORT}")
    print(f"  API 地址: http://localhost:{settings.APP_PORT}/api/v1")
    print(f"  API 文档: http://localhost:{settings.APP_PORT}/docs")
    print(f"  详细健康检查: http://localhost:{settings.APP_PORT}/api/v1/health/detailed")
    print(f"  日志文件: {os.path.join(settings.DATA_DIR, 'logs', 'app.log')}")
    print("=" * 50)

    # [#25] 优雅关闭：注册信号处理
    def handle_shutdown(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info(f"收到信号 {signum}，开始优雅关闭...")

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        # [#25] 优雅关闭：设置超时
        timeout_graceful_shutdown=30,
        # 性能优化：限制 keep-alive 连接超时，避免空闲连接堆积
        # 默认5秒，改为30秒（SSE流式响应需要较长的空闲间隔）
        timeout_keep_alive=30,
    )
