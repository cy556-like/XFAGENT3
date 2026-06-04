"""
FastAPI 路由定义
提供 REST API 接口供外部调用
包含：认证（JWT）、聊天（含流式）、文档管理、会话管理、模型管理、统计

优化:
- [#20] 可观测性：请求日志中间件 + 性能指标
- [#22] 配置中心：运行时热更新配置 API
- [#23] API 分页：对话列表/文档列表支持分页
- [#24] 健康检查增强：检查 ChromaDB/LLM API/磁盘等依赖
"""
import os
import asyncio
import time
import shutil
import json
import base64
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Query, Header
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from urllib.parse import unquote

from app.agent.core import chat, chat_stream_generator, chat_stream_generator_multimodal, reset_agent
from app.rag.document import index_document, search_documents, list_indexed_documents, delete_document, update_document, delete_agent_collection, list_all_collections, load_document, export_document_as_docx, reindex_all_documents, get_indexing_mode, _get_export_dir, cleanup_export_files
from app.auth.user_manager import login_user, register_user
from app.auth.jwt_handler import create_token, verify_token, get_username_from_token
from app.memory.manager import (
    get_history_messages, clear_session_history,
    create_chat, list_chats, delete_chat, rename_chat, update_chat_time,
)
from app.config import settings, AVAILABLE_MODELS, get_current_model, set_current_model
from app.utils.stats import record_message, record_session, get_stats
from app.agent.storage import sync_agents as storage_sync_agents, load_agents as storage_load_agents

logger = logging.getLogger(__name__)

# 文件大小限制：50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

router = APIRouter()


# ===== [#20] 可观测性：请求计时 + 性能日志 =====
import threading

_request_stats = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_response_time": 0.0,
    "endpoint_stats": {},  # path -> {count, avg_time, errors}
}
_request_stats_lock = threading.Lock()

# [性能修复] 端点统计上限，避免长时间运行后内存无限增长
_MAX_ENDPOINT_STATS = 50


def _record_request(path: str, duration: float, is_error: bool = False):
    """记录请求统计（线程安全）"""
    with _request_stats_lock:
        _request_stats["total_requests"] += 1
        if is_error:
            _request_stats["total_errors"] += 1
        
        # 更新平均响应时间
        total = _request_stats["total_requests"]
        prev_avg = _request_stats["avg_response_time"]
        _request_stats["avg_response_time"] = prev_avg + (duration - prev_avg) / total
        
        # 端点统计
        if path not in _request_stats["endpoint_stats"]:
            # [性能修复] 超过上限时淘汰请求量最少的端点
            if len(_request_stats["endpoint_stats"]) >= _MAX_ENDPOINT_STATS:
                min_path = min(_request_stats["endpoint_stats"], 
                              key=lambda k: _request_stats["endpoint_stats"][k]["count"])
                del _request_stats["endpoint_stats"][min_path]
            _request_stats["endpoint_stats"][path] = {"count": 0, "avg_time": 0.0, "errors": 0}
        ep = _request_stats["endpoint_stats"][path]
        ep["count"] += 1
        prev = ep["avg_time"]
        ep["avg_time"] = prev + (duration - prev) / ep["count"]
        if is_error:
            ep["errors"] += 1


# ===== JWT 认证依赖 =====
def get_current_user(request: Request) -> str:
    """
    从请求中提取当前用户名（JWT Token 验证）
    不强制认证，但如果有 Token 则验证
    注意：已移除查询参数回退，防止认证绕过
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = get_username_from_token(token)
        if username:
            return username
    return ""


def require_auth(request: Request) -> str:
    """
    强制要求 JWT 认证
    返回已认证的用户名
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = get_username_from_token(token)
        if username:
            return username
    raise HTTPException(status_code=401, detail="未认证，请重新登录")


# ===== 请求/响应模型 =====
class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: str = "default"
    web_search: bool = False
    mode: str = "agent"  # agent / chat
    deep_think: bool = False
    agent_id: str = None  # 智能体ID，用于知识库隔离
    agent_task: str = None  # 智能体任务描述，用于动态系统提示词


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str


class SearchRequest(BaseModel):
    """文档搜索请求"""
    query: str
    top_k: int = 3


class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """注册请求"""
    username: str
    password: str


class ModelSetRequest(BaseModel):
    """设置模型请求"""
    model_id: str


class RenameRequest(BaseModel):
    """重命名会话请求"""
    username: str
    chat_id: str
    new_title: str


# [#22] 配置中心请求模型
class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    key: str  # 配置项名称，如 LLM_MODEL, MAX_TOOL_ROUNDS 等
    value: str  # 新值（字符串形式，内部转换）


class ModifyDocumentRequest(BaseModel):
    """修改知识库文档请求"""
    content: str  # 新的文档内容（纯文本）
    append: bool = False  # 是否追加内容（True=在原文末尾追加，False=替换全部内容）
    return_docx: bool = False  # 是否同时返回修改后的docx文件下载链接
    agent_id: str = None  # 智能体ID，用于知识库隔离


class ExportDocumentRequest(BaseModel):
    """导出/生成文档请求"""
    content: str  # 文档内容（纯文本）
    filename: str = ""  # 输出文件名（含扩展名），为空则自动生成
    title: str = ""  # 文档标题，为空则使用filename


# ===== 认证接口 =====

@router.post("/auth/login", summary="用户登录")
async def auth_login(req: LoginRequest):
    """用户登录验证，返回 JWT Token"""
    start = time.time()
    try:
        result = login_user(req.username, req.password)
        if result.get("success"):
            # 签发 JWT Token
            token = create_token(req.username)
            result["token"] = token
        return result
    finally:
        _record_request("/auth/login", time.time() - start)


@router.post("/auth/register", summary="用户注册")
async def auth_register(req: RegisterRequest):
    """用户注册"""
    start = time.time()
    try:
        result = register_user(req.username, req.password)
        if result.get("success"):
            # 注册成功也签发 Token
            token = create_token(req.username)
            result["token"] = token
        return result
    finally:
        _record_request("/auth/register", time.time() - start)


@router.get("/auth/me", summary="验证 Token 有效性")
async def auth_me(request: Request):
    """验证当前 JWT Token 是否有效"""
    try:
        username = require_auth(request)
        return {"valid": True, "username": username}
    except HTTPException:
        return {"valid": False, "username": None}


# ===== 聊天接口 =====

@router.post("/chat", response_model=ChatResponse, summary="与 Agent 对话（非流式）")
async def chat_api(req: ChatRequest, username: str = Depends(get_current_user)):
    """
    核心接口：与文档助手 Agent 对话（非流式）

    - 支持 RAG 文档问答
    - 支持员工信息查询
    - 支持多轮对话
    """
    start = time.time()
    is_error = False
    try:
        response = chat(req.message, req.session_id, web_search=req.web_search, mode=req.mode, deep_think=req.deep_think, agent_id=req.agent_id, agent_task=req.agent_task)
        # 更新会话时间
        try:
            parts = req.session_id.split("_", 1)
            if len(parts) == 2:
                update_chat_time(parts[0], req.session_id)
        except Exception:
            pass
        # 记录统计
        record_message(username=username or "anonymous", model_id=get_current_model())
        return ChatResponse(response=response, session_id=req.session_id)
    except Exception as e:
        is_error = True
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")
    finally:
        _record_request("/chat", time.time() - start, is_error=is_error)


@router.post("/chat/stream", summary="与 Agent 对话（流式 SSE）")
async def chat_stream_api(req: ChatRequest, request: Request, username: str = Depends(get_current_user)):
    """
    流式对话接口：逐 token 输出，同时显示工具调用进度
    返回 Server-Sent Events (SSE) 流
    
    性能优化：检测客户端断开，避免服务端空转消耗资源
    
    BUG FIX v5：客户端断开时通过取消 asyncio.Task 真正终止 Agent 执行，
    不再只是 break 退出循环（旧方式会导致 LangGraph 后台继续调用 LLM，消耗 rate limit）
    """
    start = time.time()
    # 记录统计
    record_message(username=username or "anonymous", model_id=get_current_model())

    async def event_generator():
        # [BUG FIX v5] 用 Queue + Producer Task 模式替代直接迭代，
        # 使得客户端断开时可以真正 cancel Agent 执行（而非只 break 循环）
        queue = asyncio.Queue()
        stream_done = object()  # 流结束的哨兵值
        cancelled_by_client = False
        
        async def produce():
            """在独立 Task 中运行 chat_stream_generator，结果通过 queue 传递"""
            try:
                async for chunk in chat_stream_generator(req.message, req.session_id, web_search=req.web_search, mode=req.mode, deep_think=req.deep_think, agent_id=req.agent_id, agent_task=req.agent_task):
                    # 检查客户端是否断开
                    if await request.is_disconnected():
                        nonlocal cancelled_by_client
                        cancelled_by_client = True
                        logger.info(f"SSE客户端断开，正在终止Agent执行: session={req.session_id}")
                        # 中断 async for 循环，触发 chat_stream_generator 内部的 CancelledError/cleanup
                        break
                    await queue.put(chunk)
                await queue.put(stream_done)
            except asyncio.CancelledError:
                # 被外部 cancel 时正常退出
                logger.info(f"Agent执行任务被取消: session={req.session_id}")
                raise
            except Exception as e:
                logger.exception(f"SSE生产者异常: session={req.session_id}")
                await queue.put({'type': 'error', 'content': str(e)})
                await queue.put(stream_done)
        
        producer_task = asyncio.create_task(produce())
        
        try:
            while True:
                # 用独立 task 等待下一个 chunk，这样可以同时检测客户端断开
                get_task = asyncio.create_task(queue.get())
                while not get_task.done():
                    if await request.is_disconnected():
                        cancelled_by_client = True
                        logger.info(f"SSE客户端断开，正在取消Agent执行: session={req.session_id}")
                        get_task.cancel()
                        # 真正取消 producer task → 中断 chat_stream_generator → 触发 CancelledError handler
                        producer_task.cancel()
                        try:
                            await asyncio.shield(producer_task)
                        except asyncio.CancelledError:
                            pass
                        return
                    await asyncio.sleep(0.05)
                
                chunk = await get_task
                if chunk is stream_done:
                    break
                if isinstance(chunk, dict) and chunk.get('type') == 'error':
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                    break
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            logger.info(f"SSE流被取消（外部信号）: session={req.session_id}")
            producer_task.cancel()
            try:
                await asyncio.shield(producer_task)
            except asyncio.CancelledError:
                pass
            return
        finally:
            if not producer_task.done():
                producer_task.cancel()
        
        # 更新会话时间
        try:
            parts = req.session_id.split("_", 1)
            if len(parts) == 2:
                update_chat_time(parts[0], req.session_id)
        except Exception:
            pass
        _record_request("/chat/stream", time.time() - start)
        if cancelled_by_client:
            logger.info(f"SSE流完成（客户端主动断开）: session={req.session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat-with-file/stream", summary="带文件的流式对话")
async def chat_with_file_stream(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(""),
    session_id: str = Form("default"),
    web_search: bool = Form(False),
    mode: str = Form("agent"),
    deep_think: bool = Form(False),
    agent_id: str = Form(None),
    agent_task: str = Form(None),
    store_to_kb: str = Form("true"),
    username: str = Depends(get_current_user),
):
    """
    带文件的流式对话：支持图片和文档
    - 图片（png/jpg/jpeg/gif/bmp/webp）：转为base64传给LLM分析
    - 文档（pdf/txt/docx）：索引后基于内容回答
    - 其他文件：读取文本内容（如有）传给LLM
    返回 Server-Sent Events (SSE) 流
    """
    start = time.time()
    # 记录统计
    record_message(username=username or "anonymous", model_id=get_current_model())

    ext = os.path.splitext(file.filename)[1].lower()
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    doc_exts = {".pdf", ".txt", ".docx", ".xlsx", ".xls"}
    code_exts = {".py", ".js", ".html", ".css", ".json", ".md", ".csv", ".xlsx", ".xls", ".doc", ".ppt", ".pptx"}

    # 文件大小检查
    file_content_raw = await file.read()
    if len(file_content_raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 50MB），当前文件: {len(file_content_raw) // 1024 // 1024}MB")
    # 重置文件指针
    await file.seek(0)

    logger.info(f"收到文件上传: {file.filename}, 大小: {len(file_content_raw)} bytes")

    if ext in image_exts:
        # 图片文件：用多模态消息格式传给LLM做视觉分析（复用已读取的 file_content_raw）
        b64 = base64.b64encode(file_content_raw).decode("utf-8")
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")
        # 构建多模态消息内容
        image_url = f"data:{mime_type};base64,{b64}"
        multimodal_content = [
            {"type": "text", "text": f"[用户上传了图片: {file.filename}]\n\n{message or '请描述这张图片'}"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        # 直接调用多模态流式生成
        async def event_generator():
            try:
                async for chunk in chat_stream_generator_multimodal(multimodal_content, session_id, agent_id=agent_id, agent_task=agent_task):
                    if await request.is_disconnected():
                        logger.info(f"SSE客户端已断开(图片模式)，停止生成: session={session_id}")
                        break
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"SSE流异常(图片模式): session={session_id}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            try:
                parts = session_id.split("_", 1)
                if len(parts) == 2:
                    update_chat_time(parts[0], session_id)
            except Exception:
                pass
            _record_request("/chat-with-file/stream", time.time() - start)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    elif ext in doc_exts:
        # 普通聊天模式（无 agent_id）：不存入知识库，只临时读取文件内容回答
        # 文件保存到临时目录，删除会话时自动清理
        # URL解码文件名：浏览器上传的中文文件名可能是URL编码的，统一解码
        decoded_filename = unquote(file.filename)
        if agent_id:
            # 智能体模式：文件存到智能体专属目录
            agent_dir = os.path.join(settings.DOCUMENTS_DIR, f"agent_{agent_id}")
            os.makedirs(agent_dir, exist_ok=True)
            file_path = os.path.join(agent_dir, decoded_filename)
        else:
            # 普通模式：文件存到临时目录（不进知识库，删除会话时清理）
            temp_dir = os.path.join(settings.DATA_DIR, "temp", session_id)
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, decoded_filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if store_to_kb == "true" and agent_id:
            # 知识库模式 ON + 有 agent_id：索引到智能体知识库
            try:
                index_result = index_document(file_path, decoded_filename, agent_id=agent_id)
                indexing_mode = index_result.get('indexing_mode', 'unknown')
                logger.info(f"文件已索引到知识库: {file.filename}, agent_id={agent_id}, 分块数={index_result.get('chunks', 0)}, 索引模式={indexing_mode}")
            except Exception as e:
                os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"文档索引失败: {str(e)}")
            full_message = f"[用户上传了文档: {file.filename}]\n\n{message}"
        else:
            # 普通模式或 store_to_kb=false：只读取内容回答，不存入知识库
            try:
                docs = load_document(file_path)
                text = "\n".join([doc.page_content for doc in docs])
                full_message = f"[用户上传了文档: {file.filename}]\n\n文档内容：\n{text[:8000]}\n\n{message}"
                mode_label = "普通聊天（不存知识库）" if not agent_id else "知识库模式OFF"
                logger.info(f"文件仅读取内容（{mode_label}）: {file.filename}")
            except Exception as e:
                os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"文档读取失败: {str(e)}")

    elif ext in code_exts:
        # 代码/其他文本文件：读取内容传给LLM
        try:
            file_content = await file.read()
            text = file_content.decode("utf-8", errors="replace")
            full_message = f"[用户上传了文件: {file.filename}]\n\n文件内容：\n```\n{text[:8000]}\n```\n\n{message}"
        except Exception:
            full_message = f"[用户上传了文件: {file.filename}，但无法读取内容]\n\n{message}"
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    # 流式回答
    async def event_generator():
        aid = agent_id if agent_id else None
        atask = agent_task if agent_task else None
        try:
            async for chunk in chat_stream_generator(full_message, session_id, web_search=web_search, mode=mode, deep_think=deep_think, agent_id=aid, agent_task=atask):
                if await request.is_disconnected():
                    logger.info(f"SSE客户端已断开(文档模式)，停止生成: session={session_id}")
                    break
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"SSE流异常(文档模式): session={session_id}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        try:
            parts = session_id.split("_", 1)
            if len(parts) == 2:
                update_chat_time(parts[0], session_id)
        except Exception:
            pass
        _record_request("/chat-with-file/stream", time.time() - start)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===== 文档管理接口 =====

@router.post("/upload", summary="上传文档到知识库")
async def upload_document(file: UploadFile = File(...), agent_id: str = Form(None)):
    """
    上传文档并自动索引到向量数据库
    支持 PDF、TXT、MD、DOCX 格式
    必须指定 agent_id（普通聊天模式无知识库，不支持上传到知识库）
    """
    # 普通聊天模式无知识库，必须指定 agent_id
    if not agent_id:
        raise HTTPException(
            status_code=400,
            detail="请先选择一个智能体再上传文档到知识库。普通聊天模式不支持知识库功能。",
        )

    # 检查文件格式
    allowed_ext = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 {allowed_ext}",
        )

    # 文件大小检查
    file_content_raw = await file.read()
    if len(file_content_raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 50MB）")
    await file.seek(0)

    logger.info(f"知识库上传文档: {file.filename}, 大小: {len(file_content_raw)} bytes")

    # 保存文件 - 使用per-agent子目录实现文件隔离
    if agent_id:
        agent_dir = os.path.join(settings.DOCUMENTS_DIR, f"agent_{agent_id}")
        os.makedirs(agent_dir, exist_ok=True)
        # URL解码文件名：浏览器上传的中文文件名可能是URL编码的，统一解码
        decoded_filename = unquote(file.filename)
        file_path = os.path.join(agent_dir, decoded_filename)
    else:
        # URL解码文件名
        decoded_filename = unquote(file.filename)
        file_path = os.path.join(settings.DOCUMENTS_DIR, decoded_filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 索引文档（[#11] 自动降级：embedding不可用时切换为关键词索引）
    try:
        result = index_document(file_path, decoded_filename, agent_id=agent_id)
        indexing_mode_result = result.get('indexing_mode', 'unknown')
        logger.info(f"文档索引完成: {file.filename}, agent_id={agent_id}, 分块数={result.get('chunks', 0)}, 索引模式={indexing_mode_result}")
        return {"status": "success", "detail": result}
    except Exception as e:
        # 索引失败则删除文件
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档索引失败: {str(e)}")


@router.post("/search", summary="搜索文档内容")
async def search_api(req: SearchRequest, agent_id: str = Query(None, description="智能体ID，为空时搜全局知识库")):
    """在文档库中搜索相关内容（支持按智能体隔离）"""
    # 普通聊天模式没有知识库
    if not agent_id:
        return {"query": req.query, "results": [], "message": "普通聊天模式没有知识库，请先选择一个智能体"}
    results = search_documents(req.query, req.top_k, agent_id=agent_id)
    return {"query": req.query, "results": results}


@router.get("/documents", summary="列出所有已索引文档")
async def list_documents(
    page: int = Query(1, ge=1, description="页码"),          # [#23] 分页
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    agent_id: str = Query(None, description="智能体ID，为空时查全局知识库"),
):
    """获取知识库中所有文档列表（支持分页，按智能体隔离）

    [#11] 同时扫描关键词索引和磁盘文件，确保关键词模式下也能正确列出文档
    
    注意：普通聊天模式（agent_id=None）没有知识库，返回空列表
    """
    # 普通聊天模式没有知识库
    if not agent_id:
        return {
            "documents": [],
            "count": 0,
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
        }

    docs = list_indexed_documents(agent_id=agent_id)

    # 额外扫描：list_indexed_documents 已扫描 .pdf/.txt/.docx，
    # 这里补充扫描更多文件类型（代码文件、Office文档等）
    extra_extensions = {'.csv', '.xlsx', '.xls', '.doc', '.ppt', '.pptx', '.md', '.py', '.js', '.html', '.css', '.json'}
    indexed_filenames = set()
    for doc in docs:
        if isinstance(doc, dict) and doc.get('filename'):
            indexed_filenames.add(doc['filename'])
        elif isinstance(doc, str):
            indexed_filenames.add(doc)

    # 扫描对应的目录（仅补充额外格式的文件）
    if agent_id:
        scan_dir = os.path.join(settings.DOCUMENTS_DIR, f"agent_{agent_id}")
    else:
        scan_dir = settings.DOCUMENTS_DIR

    if os.path.exists(scan_dir):
        for fname in os.listdir(scan_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extra_extensions and fname not in indexed_filenames:
                file_path = os.path.join(scan_dir, fname)
                if os.path.isfile(file_path):
                    docs.append(fname)

    # 统一格式为纯文件名字符串（前端兼容）
    normalized_docs = []
    for doc in docs:
        if isinstance(doc, dict):
            normalized_docs.append(doc.get('filename', doc.get('name', str(doc))))
        else:
            normalized_docs.append(str(doc))
    docs = normalized_docs

    total = len(docs)
    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    paginated = docs[start:end]
    return {
        "documents": paginated,
        "count": total,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.put("/documents/{filename}", summary="修改知识库文档内容")
async def modify_document_api(filename: str, req: ModifyDocumentRequest):
    """
    修改知识库中指定文档的内容
    支持两种模式：
    - 替换模式（append=false）：用新内容完全替换原文档内容
    - 追加模式（append=true）：在原文档内容末尾追加新内容
    修改后会自动重新索引到向量数据库
    """
    # 检查文档是否存在（优先查找agent子目录）
    file_path = None
    if req.agent_id:
        agent_path = os.path.join(settings.DOCUMENTS_DIR, f"agent_{req.agent_id}", filename)
        if os.path.exists(agent_path):
            file_path = agent_path
    if not file_path:
        global_path = os.path.join(settings.DOCUMENTS_DIR, filename)
        if os.path.exists(global_path):
            file_path = global_path
    if not file_path:
        raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")

    # 追加模式：先读取原内容，拼接新内容
    final_content = req.content
    if req.append:
        try:
            from app.rag.document import load_document
            docs = load_document(file_path)
            original_text = "\n".join([doc.page_content for doc in docs])
            final_content = original_text + "\n" + req.content
        except Exception as e:
            logger.warning(f"读取原文档内容失败，改为替换模式: {e}")

    logger.info(f"知识库修改文档: {filename}, 追加模式={req.append}, 内容长度={len(final_content)}, agent_id={req.agent_id}")

    result = update_document(filename, final_content, agent_id=req.agent_id, async_reindex=True)  # 异步重索引，加速响应
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    response_data = {"status": "success", "detail": result}

    # 如果用户要求返回docx文件下载链接
    if req.return_docx:
        try:
            docx_filename = filename.rsplit('.', 1)[0] + '.docx'
            docx_result = export_document_as_docx(final_content, docx_filename)
            if docx_result["status"] == "success":
                actual_docx_filename = docx_result.get('filename', docx_filename)
                response_data["download_url"] = f"/api/v1/documents/export-download/{actual_docx_filename}"
                response_data["docx_filename"] = actual_docx_filename
        except Exception as e:
            logger.warning(f"生成docx下载文件失败: {e}")

    return response_data


@router.get("/documents/{filename}/download", summary="下载知识库文档")
async def download_document(filename: str, agent_id: str = Query(None, description="智能体ID，为空时查全局知识库")):
    """
    下载知识库中的文档文件
    支持 .docx / .txt / .pdf 格式
    """
    # 先查找agent子目录，再查全局目录
    if agent_id:
        file_path = os.path.join(settings.DOCUMENTS_DIR, f"agent_{agent_id}", filename)
    else:
        file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        # 回退：尝试全局目录
        fallback_path = os.path.join(settings.DOCUMENTS_DIR, filename)
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")

    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
    }
    media_type = mime_map.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        content = f.read()

    # RFC 5987: 中文文件名需要URL编码
    from urllib.parse import quote
    encoded_filename = quote(filename)

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@router.post("/documents/export", summary="导出/生成文档为docx")
async def export_document_api(req: ExportDocumentRequest):
    """
    将文本内容生成为docx文档并提供下载
    支持从知识库内容整合生成综合文档或简略文档
    """
    try:
        filename = req.filename or f"export_{int(time.time())}.docx"
        if not filename.endswith('.docx'):
            filename += '.docx'

        result = export_document_as_docx(req.content, filename, title=req.title)
        if result["status"] == "success":
            actual_filename = result.get('filename', filename)
            return {
                "status": "success",
                "filename": actual_filename,
                "download_url": f"/api/v1/documents/export-download/{actual_filename}",
                "message": result["message"],
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "导出失败"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档导出失败: {str(e)}")


class ExportXlsxRequest(BaseModel):
    """导出/生成XLSX文档请求"""
    content: str  # 文档内容（Markdown格式，支持表格）
    filename: str = ""  # 输出文件名（含扩展名），为空则自动生成
    title: str = ""  # 文档标题/工作表名称，为空则使用filename


@router.post("/documents/export-xlsx", summary="导出/生成文档为xlsx")
async def export_xlsx_api(req: ExportXlsxRequest):
    """
    将文本内容生成为xlsx（Excel）文档并提供下载
    支持Markdown表格自动转为Excel原生表格
    """
    try:
        from app.rag.document import export_document_as_xlsx
        
        filename = req.filename or f"export_{int(time.time())}.xlsx"
        if not filename.endswith('.xlsx'):
            filename = filename.rsplit('.', 1)[0] + '.xlsx'

        result = export_document_as_xlsx(req.content, filename, title=req.title)
        if result["status"] == "success":
            actual_filename = result.get('filename', filename)
            return {
                "status": "success",
                "filename": actual_filename,
                "download_url": f"/api/v1/documents/export-download/{actual_filename}",
                "message": result["message"],
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "导出失败"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档导出失败: {str(e)}")


@router.get("/documents/export-download/{filename}", summary="下载AI导出的文档")
async def download_export_document(filename: str):
    """
    下载AI生成的导出文档（docx/txt）
    文件保存在 data/export/{session_id}/ 目录中
    支持会话子目录查找 + 兼容旧版平铺目录
    """
    from urllib.parse import unquote
    import unicodedata

    # URL解码文件名（处理中文文件名）
    # FastAPI可能已经自动解码一次，再unquote确保双重编码也能处理
    decoded_filename = unquote(unquote(filename))
    # 安全检查：防止路径穿越
    safe_filename = decoded_filename.replace('/', '_').replace('\\', '_').replace('..', '_')

    export_root = _get_export_dir()  # export 根目录
    file_path = None

    # 1. 先在会话子目录中查找（新版本：data/export/{session_id}/xxx.docx）
    if os.path.exists(export_root):
        for item in os.listdir(export_root):
            sub_dir = os.path.join(export_root, item)
            if os.path.isdir(sub_dir):
                candidate = os.path.join(sub_dir, safe_filename)
                if os.path.exists(candidate):
                    file_path = candidate
                    logger.info(f"[导出下载] 在会话目录 {item}/ 中找到文件: {safe_filename}")
                    break

    # 2. 兼容旧版：直接在 export 根目录查找
    if file_path is None:
        file_path = os.path.join(export_root, safe_filename)

    # 3. 精确匹配
    if os.path.exists(file_path):
        logger.info(f"[导出下载] 文件匹配成功: {safe_filename}")
    else:
        # 4. 模糊匹配：尝试Unicode标准化 + 不带扩展名匹配
        found = False

        # 方法1：NFC/NFD Unicode标准化
        norm_filename = unicodedata.normalize('NFC', safe_filename)
        # 先搜子目录
        if os.path.exists(export_root):
            for item in os.listdir(export_root):
                search_dir = os.path.join(export_root, item) if os.path.isdir(os.path.join(export_root, item)) else export_root
                norm_path = os.path.join(search_dir, norm_filename)
                if os.path.exists(norm_path):
                    file_path = norm_path
                    safe_filename = norm_filename
                    found = True
                    logger.info(f"[导出下载] 通过Unicode标准化匹配成功: {safe_filename}")
                    break

        # 方法2：遍历所有目录做模糊匹配（忽略Unicode差异）
        if not found and os.path.exists(export_root):
            base_name = os.path.splitext(safe_filename)[0]
            # 遍历根目录和所有子目录
            search_dirs = [export_root]
            for item in os.listdir(export_root):
                sub = os.path.join(export_root, item)
                if os.path.isdir(sub):
                    search_dirs.append(sub)

            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                for existing_file in os.listdir(search_dir):
                    existing_path = os.path.join(search_dir, existing_file)
                    if not os.path.isfile(existing_path):
                        continue
                    existing_base = os.path.splitext(existing_file)[0]
                    # 比较Unicode标准化后的文件名
                    if (unicodedata.normalize('NFC', existing_base) == unicodedata.normalize('NFC', base_name)
                        and os.path.splitext(safe_filename)[1].lower() == os.path.splitext(existing_file)[1].lower()):
                        file_path = existing_path
                        safe_filename = existing_file
                        found = True
                        logger.info(f"[导出下载] 通过模糊匹配找到文件: {existing_file} (请求: {decoded_filename})")
                        break
                if found:
                    break

        if not found:
            # 记录目录中现有文件，帮助调试
            existing_files = []
            if os.path.exists(export_root):
                for item in os.listdir(export_root):
                    sub = os.path.join(export_root, item)
                    if os.path.isdir(sub):
                        existing_files.extend([f"{item}/{f}" for f in os.listdir(sub) if os.path.isfile(os.path.join(sub, f))])
                    elif os.path.isfile(sub):
                        existing_files.append(item)
            logger.warning(f"[导出下载] 文件未找到! 请求文件名: {safe_filename}, 目录中现有文件: {existing_files[:10]}")
            raise HTTPException(status_code=404, detail=f"导出文档 {safe_filename} 不存在")

    ext = os.path.splitext(safe_filename)[1].lower()
    mime_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
    }
    media_type = mime_map.get(ext, "application/octet-stream")

    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取导出文档失败: {str(e)}")

    # 使用 RFC 5987 编码处理中文文件名
    from urllib.parse import quote
    encoded_filename = quote(safe_filename)

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@router.delete("/documents/{filename}", summary="从知识库删除文档")
async def delete_document_api(filename: str, agent_id: str = Query(None, description="智能体ID，为空时删全局知识库文档")):
    """
    从知识库中删除指定文档
    同时删除 ChromaDB 中的向量分块和原始文件
    
    注意：普通聊天模式（agent_id=None）没有知识库，不支持删除
    """
    # 普通聊天模式没有知识库
    if not agent_id:
        raise HTTPException(status_code=400, detail="普通聊天模式没有知识库，不支持删除文档。请先选择一个智能体。")
    result = delete_document(filename, agent_id=agent_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"status": "success", "detail": result}


# ===== 会话历史接口 =====

@router.get("/history/{session_id}", summary="获取对话历史")
async def get_history(session_id: str):
    """获取指定会话的对话历史"""
    messages = get_history_messages(session_id)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}", summary="清除对话历史")
async def delete_history(session_id: str):
    """清除指定会话的对话历史，同时清理临时文件和导出文件"""
    clear_session_history(session_id)
    # 清理普通模式下的临时上传文件（data/temp/{session_id}/）
    temp_dir = os.path.join(settings.DATA_DIR, "temp", session_id)
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"清空历史时清理临时文件: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    # 清理该会话的导出文件（data/export/{session_id}/）
    try:
        deleted_count = cleanup_export_files(session_id=session_id)
        if deleted_count > 0:
            logger.info(f"清空历史时清理了 {deleted_count} 个导出文件")
    except Exception as e:
        logger.warning(f"清理导出文件失败: {e}")
    return {"status": "success", "message": f"会话 {session_id} 的历史已清除"}


# ===== 会话管理接口 =====

@router.get("/chats", summary="获取用户会话列表")
async def get_chats(
    username: str,
    mode: str = Query(None, description="模式过滤: agent/chat"),
    page: int = Query(1, ge=1, description="页码"),          # [#23] 分页
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    """获取用户的会话列表（支持分页，支持按模式过滤）"""
    chats = list_chats(username, mode=mode, skip_auto_title=True)  # GET请求跳过自动标题更新，避免写副作用
    total = len(chats)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = chats[start:end]
    return {
        "success": True,
        "chats": paginated,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/chats", summary="创建新会话")
async def create_chat_api(
    username: str,
    title: str = "新对话",
    mode: str = "agent",
    agent_id: str = Query(None, description="智能体ID，会话归属到指定智能体"),
):
    """为用户创建一个新的会话（支持指定模式和智能体归属）"""
    chat_info = create_chat(username, title, mode=mode, agent_id=agent_id)
    record_session()
    return {"success": True, "chat": chat_info}


@router.delete("/chats/{chat_id}", summary="删除会话")
async def delete_chat_api(chat_id: str, username: str):
    """删除用户的某个会话，同时清理普通模式下的临时文件和导出文件"""
    delete_chat(username, chat_id)
    # 清理普通模式下的临时文件（存放在 data/temp/{session_id}/ 目录）
    temp_dir = os.path.join(settings.DATA_DIR, "temp", chat_id)
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时文件: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    # 清理该会话的导出文件（data/export/{chat_id}/，只删当前会话的）
    try:
        deleted_count = cleanup_export_files(session_id=chat_id)
        if deleted_count > 0:
            logger.info(f"已清理 {deleted_count} 个导出文件")
    except Exception as e:
        logger.warning(f"清理导出文件失败: {e}")
    return {"success": True, "message": "会话已删除"}


@router.put("/chats/{chat_id}/rename", summary="重命名会话")
async def rename_chat_api(chat_id: str, req: RenameRequest):
    """重命名用户的某个会话"""
    rename_chat(req.username, req.chat_id, req.new_title)
    return {"success": True, "message": "会话已重命名"}


# ===== 模型管理接口 =====

@router.get("/models", summary="获取可用模型列表")
async def get_models():
    """获取所有可用的 LLM 模型列表"""
    current = get_current_model()
    return {"models": AVAILABLE_MODELS, "current": current}


@router.post("/models/set", summary="切换模型")
async def set_model(req: ModelSetRequest):
    """切换当前使用的 LLM 模型"""
    success = set_current_model(req.model_id)
    if success:
        return {"success": True, "message": f"已切换到模型: {req.model_id}"}
    return {"success": False, "message": f"不支持的模型: {req.model_id}"}


# ===== 使用统计接口 =====

@router.get("/stats", summary="获取使用统计")
async def get_usage_stats(username: str = Depends(get_current_user)):
    """获取系统使用统计数据"""
    stats = get_stats()
    # [#20] 附加 API 性能指标
    stats["api_performance"] = {
        "total_requests": _request_stats["total_requests"],
        "total_errors": _request_stats["total_errors"],
        "avg_response_time_ms": round(_request_stats["avg_response_time"] * 1000, 2),
        "error_rate": round(_request_stats["total_errors"] / max(_request_stats["total_requests"], 1) * 100, 2),
    }
    return {"success": True, "stats": stats}


# ===== [#22] 配置中心 API =====

@router.get("/config", summary="获取运行时配置")
async def get_config(username: str = Depends(require_auth)):
    """获取当前运行时配置（隐藏敏感信息）"""
    return {
        "success": True,
        "config": {
            "LLM_MODEL": settings.LLM_MODEL,
            "LLM_BASE_URL": settings.LLM_BASE_URL,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "APP_HOST": settings.APP_HOST,
            "APP_PORT": settings.APP_PORT,
            "GITHUB_TOKEN_CONFIGURED": bool(os.getenv("GITHUB_TOKEN", "")),
            "SMTP_CONFIGURED": bool(os.getenv("SMTP_HOST", "")),
            "DATABASE_CONFIGURED": bool(os.getenv("DATABASE_URL", "")),
        }
    }


@router.post("/config", summary="更新运行时配置（热更新）")
async def update_config(req: ConfigUpdateRequest, username: str = Depends(require_auth)):
    """
    [#22] 运行时热更新配置，无需重启服务
    支持更新的配置项：LLM_MODEL, APP_PORT 等
    """
    allowed_keys = {"LLM_MODEL", "APP_PORT", "EMBEDDING_MODEL"}
    
    if req.key not in allowed_keys:
        raise HTTPException(status_code=400, detail=f"不允许更新的配置项: {req.key}。支持: {allowed_keys}")
    
    old_value = getattr(settings, req.key, None)
    if old_value is None:
        raise HTTPException(status_code=400, detail=f"未知的配置项: {req.key}")
    
    # 类型转换
    try:
        if req.key == "APP_PORT":
            new_value = int(req.value)
        else:
            new_value = req.value
    except ValueError:
        raise HTTPException(status_code=400, detail=f"配置值类型错误: {req.key} 期望 {type(old_value).__name__}")
    
    # 应用更新
    setattr(settings, req.key, new_value)
    
    # 如果更新了模型，重置 Agent
    if req.key == "LLM_MODEL":
        reset_agent()
        logger.info(f"配置热更新: {req.key} = {new_value}, Agent 已重置")
    elif req.key == "EMBEDDING_MODEL":
        from app.rag.document import reset_vector_store
        reset_vector_store()
        logger.info(f"配置热更新: {req.key} = {new_value}, 向量数据库已重置")
    
    logger.info(f"配置热更新: {req.key} 由 {old_value} 变更为 {new_value}, 操作者: {username}")
    
    return {
        "success": True,
        "message": f"配置 {req.key} 已更新",
        "old_value": str(old_value),
        "new_value": str(new_value),
    }


# ===== 导出对话接口 =====

@router.get("/export/{session_id}", summary="导出对话")
async def export_chat(session_id: str, format: str = "md"):
    """
    导出对话为 Markdown 或 PDF 格式
    format: md | pdf
    """
    messages = get_history_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="没有可导出的对话内容")

    if format == "pdf":
        # PDF 导出
        try:
            from app.utils.pdf_generator import generate_chat_pdf
            pdf_bytes = generate_chat_pdf(messages, session_id)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=chat_{session_id[:12]}.pdf"
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF 生成失败: {str(e)}")
    else:
        # Markdown 导出
        content = ""
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            content += f"**{role}：**\n\n{msg['content']}\n\n---\n\n"

        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=chat_{session_id[:12]}.md"
            }
        )


# ===== [#24] 健康检查增强 =====

@router.get("/health/detailed", summary="详细健康检查")
async def health_detailed():
    """
    [#24] 详细健康检查：检查所有依赖组件状态
    - ChromaDB 可用性
    - LLM API 可达性
    - 磁盘空间
    - 内存使用
    """
    import platform
    
    checks = {}
    overall = "healthy"
    
    # 1. ChromaDB / 索引模式 检查
    indexing_mode = get_indexing_mode()
    if indexing_mode == "vector":
        try:
            from app.rag.document import get_vector_store
            vs = get_vector_store()
            if vs is not None:
                collection = vs._collection
                count = collection.count()
                checks["chromadb"] = {"status": "ok", "document_count": count, "indexing_mode": "vector"}
            else:
                checks["chromadb"] = {"status": "degraded", "indexing_mode": "keyword", "message": "Embedding 不可用，已自动降级为关键词搜索"}
        except Exception as e:
            checks["chromadb"] = {"status": "degraded", "indexing_mode": "keyword", "message": str(e)[:200]}
    elif indexing_mode == "keyword":
        checks["chromadb"] = {"status": "degraded", "indexing_mode": "keyword", "message": "Embedding API 不可用，已自动降级为关键词搜索模式"}
    else:
        checks["chromadb"] = {"status": "ok", "indexing_mode": "unknown", "message": "尚未检测 Embedding 可用性"}
    
    # 2. LLM API 检查
    try:
        import httpx
        api_url = settings.LLM_BASE_URL.rstrip("/") + "/models"
        resp = httpx.get(api_url, timeout=5)
        if resp.status_code == 200:
            checks["llm_api"] = {"status": "ok", "model": settings.LLM_MODEL}
        else:
            checks["llm_api"] = {"status": "error", "code": resp.status_code}
            overall = "degraded"
    except Exception as e:
        checks["llm_api"] = {"status": "unreachable", "message": str(e)[:100]}
        overall = "degraded"
    
    # 3. 磁盘空间检查
    try:
        disk_usage = shutil.disk_usage(settings.DATA_DIR)
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        usage_pct = (disk_usage.used / disk_usage.total) * 100
        checks["disk"] = {
            "status": "ok" if usage_pct < 90 else "warning",
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "usage_percent": round(usage_pct, 1),
        }
        if usage_pct >= 90:
            overall = "degraded"
    except Exception as e:
        checks["disk"] = {"status": "error", "message": str(e)[:100]}
    
    # 4. 内存检查
    try:
        import psutil
        mem = psutil.virtual_memory()
        checks["memory"] = {
            "status": "ok" if mem.percent < 90 else "warning",
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "used_percent": mem.percent,
        }
    except ImportError:
        checks["memory"] = {"status": "unknown", "message": "psutil not installed"}
    
    # 5. 系统信息
    checks["system"] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "version": "4.0.0",
        "indexing_mode": indexing_mode,
    }

    # 关键词模式下整体状态为 degraded（功能可用但非最佳）
    if indexing_mode == "keyword" and overall == "healthy":
        overall = "degraded"

    return {
        "status": overall,
        "checks": checks,
        "timestamp": time.time(),
    }


# ===== 智能体同步接口 =====

class AgentSyncItem(BaseModel):
    id: str
    name: str = ""
    task: str = ""
    mode: str = "agent"
    created_at: float = None
    updated_at: float = None

class AgentSyncRequest(BaseModel):
    agents: list = []

@router.post("/agents/sync", summary="同步智能体数据")
async def agents_sync(req: AgentSyncRequest, authorization: str = Header(None)):
    """
    同步智能体数据到服务端（按agent_id合并，updated_at较新的优先）
    用于跨浏览器/跨设备同步智能体prompt编辑
    """
    username = None
    if authorization and authorization.startswith("Bearer "):
        username = get_username_from_token(authorization[7:])
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    result = storage_sync_agents(username, req.agents)
    return {"success": True, "agents": result["agents"], "synced": result.get("synced", 0), "updated": result.get("updated", 0)}

@router.get("/agents", summary="获取用户智能体列表")
async def get_agents(authorization: str = Header(None)):
    """
    获取当前用户的智能体列表
    """
    username = None
    if authorization and authorization.startswith("Bearer "):
        username = get_username_from_token(authorization[7:])
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    agents = storage_load_agents(username)
    return {"success": True, "agents": agents}


# ===== 智能体知识库管理接口 =====

@router.delete("/agents/{agent_id}/knowledge", summary="删除智能体的知识库")
async def delete_agent_knowledge(agent_id: str):
    """
    删除智能体对应的整个 ChromaDB collection
    在删除智能体时调用，确保知识库数据同步清理
    """
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id 不能为空")
    result = delete_agent_collection(agent_id)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"status": "success", "detail": result}


# ===== 诊断接口 =====

@router.get("/debug/collections", summary="列出所有 ChromaDB collection")
async def debug_collections():
    """诊断接口：列出所有 ChromaDB collection 及其文档数"""
    collections = list_all_collections()
    return {"collections": collections}


@router.post("/reindex", summary="重建知识库索引（切换embedding模型后使用）")
async def reindex_knowledge(agent_id: str = Query(None, description="智能体ID，为空时重建全局知识库")):
    """
    重建指定知识库的所有文档索引。
    
    切换embedding模型后（如从智谱embedding-3切换到本地bge-large-zh-v1.5），
    旧的向量数据维度不同，必须重建索引才能正常使用向量搜索。
    
    此接口会：
    1. 记录旧collection中的文档列表
    2. 删除旧collection
    3. 用新的embedding模型重新索引所有文档
    """
    result = reindex_all_documents(agent_id=agent_id)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"status": "success", "detail": result}


@router.get("/migrate/cleanup-collections", summary="清理异常的 ChromaDB collection")
async def cleanup_collections():
    """
    清理空 collection 或有双重前缀的 collection
    例如：agent_agent_xxx → 应该是 agent_xxx
    """
    import chromadb
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    collections = client.list_collections()
    cleaned = []

    for c in collections:
        name = c.name
        # 修复双重前缀：agent_agent_xxx → agent_xxx
        if name.startswith("agent_agent_"):
            correct_name = name.replace("agent_agent_", "agent_", 1)
            try:
                # 获取旧 collection 的数据
                old_data = c.get(include=["documents", "metadatas", "embeddings"])
                if old_data.get("ids"):
                    # 创建正确名称的 collection 并迁移数据
                    from app.rag.document import get_vector_store
                    # 从 agent_agent_xxx 提取真正的 agent_id
                    real_agent_id = name.replace("agent_", "", 1)  # 去掉第一个 agent_ 前缀
                    new_vs = get_vector_store(agent_id=real_agent_id)
                    # 迁移文档
                    from langchain_core.documents import Document
                    docs = []
                    for i, doc_id in enumerate(old_data["ids"]):
                        doc = Document(
                            page_content=old_data["documents"][i] or "",
                            metadata=old_data["metadatas"][i] or {},
                        )
                        docs.append(doc)
                    if docs:
                        new_vs.add_documents(docs)
                    cleaned.append({"old": name, "new": correct_name, "migrated_docs": len(docs)})
                # 删除旧 collection
                client.delete_collection(name)
            except Exception as e:
                cleaned.append({"old": name, "error": str(e)})
        # 清理空 collection（除了 langchain）
        elif name != "langchain":
            try:
                count = c.count()
                if count == 0:
                    client.delete_collection(name)
                    cleaned.append({"deleted_empty": name})
            except:
                pass

    # 清理 vector_store 缓存
    from app.rag.document import reset_vector_store
    reset_vector_store()

    return {"status": "success", "cleaned": cleaned}
