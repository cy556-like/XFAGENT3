"""
Agent 工具定义模块
每个工具 = Agent 的一个「能力」
Agent 会根据用户问题自动选择调用哪个工具

优化:
- [#8] 工具结果缓存：LRU缓存 + TTL
- [#10] 引用溯源：返回结果标注文档名+段落位置
- [#12] 外部系统集成：GitHub API / 邮件 / 数据库查询
"""
import json
import os
import time
import hashlib
import logging
import re
from typing import Optional
from functools import wraps

from langchain_core.tools import tool

from app.config import settings
from app.rag.document import search_documents, index_document, list_indexed_documents, delete_document, update_document, export_document_as_docx, export_document_as_xlsx, get_document_content

logger = logging.getLogger(__name__)

# ===== 当前智能体上下文 =====
# 用于在 Agent 工具调用时传递 agent_id，实现知识库隔离
# 使用 contextvars.ContextVar 而非 threading.local，因为：
# 1. LangGraph 的 ToolNode 使用 ThreadPoolExecutor 执行工具
# 2. threading.local 的值不会传播到子线程，导致工具函数中 get_current_agent_id() 返回 None
# 3. contextvars.ContextVar 通过 asyncio.run_in_executor 自动复制上下文到子线程
# 4. 同时支持 asyncio 并发请求隔离（每个请求有独立的上下文）
import contextvars
_agent_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('agent_id', default=None)
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('session_id', default=None)


def set_current_agent_id(agent_id: str = None):
    """设置当前会话的智能体ID（contextvars 实现，支持 asyncio + ThreadPoolExecutor 上下文传播）"""
    _agent_id_var.set(agent_id)
    logger.debug(f"工具上下文 agent_id 设置为: {agent_id}")


def get_current_agent_id() -> str:
    """获取当前会话的智能体ID（contextvars 实现，在 ToolNode 工作线程中也能正确获取）"""
    return _agent_id_var.get()


def set_current_session_id(session_id: str = None):
    """设置当前会话的 session_id（用于导出文件按会话隔离存储）"""
    _session_id_var.set(session_id)


def get_current_session_id() -> str:
    """获取当前会话的 session_id"""
    return _session_id_var.get()


# ===== [#8] 工具结果缓存 =====
class ToolCache:
    """带 TTL 的 LRU 工具结果缓存（使用 OrderedDict 实现 O(1) 读写）
    
    性能优化：原实现使用 list + remove() 实现 LRU，每次访问 O(n)。
    改用 OrderedDict.move_to_end()，每次访问 O(1)，在高并发场景下性能显著提升。
    """
    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self._cache = {}  # key -> {"value": ..., "expire_at": float}
        self._order = {}  # OrderedDict for LRU: key -> True
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def _make_key(self, func_name: str, args: tuple, kwargs: dict, agent_id: str = "") -> str:
        """生成缓存 key（包含 agent_id 以隔离不同智能体的缓存）"""
        raw = f"{func_name}:{agent_id}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """获取缓存，过期返回 None（O(1) 操作）"""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry["expire_at"]:
            del self._cache[key]
            self._order.pop(key, None)
            self._misses += 1
            return None
        # LRU: O(1) 移到末尾
        self._order[key] = True
        self._hits += 1
        return entry["value"]

    def set(self, key: str, value: str, ttl: int = None):
        """设置缓存（O(1) 操作）"""
        if ttl is None:
            ttl = self._default_ttl
        # 容量超限时淘汰最久未访问的（FIFO from OrderedDict）
        while len(self._cache) >= self._max_size and self._order:
            oldest_key = next(iter(self._order))
            self._cache.pop(oldest_key, None)
            del self._order[oldest_key]
        self._cache[key] = {"value": value, "expire_at": time.time() + ttl}
        self._order[key] = True

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._order.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """缓存统计"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {"size": len(self._cache), "max_size": self._max_size, "hits": self._hits, "misses": self._misses, "hit_rate": f"{hit_rate:.1f}%"}


# 全局工具缓存实例
_tool_cache = ToolCache(max_size=100, default_ttl=300)

# ===== 搜索效率控制 =====
# 每轮对话的最大文档搜索次数（超过后返回提示，让LLM直接使用已有信息回答）
_MAX_SEARCH_PER_TURN = 3

# 使用 contextvars 而非全局变量，支持并发请求隔离
_search_count_var: contextvars.ContextVar[int] = contextvars.ContextVar('search_count', default=0)


def reset_search_count():
    """重置搜索计数（每次新对话轮次开始时调用）"""
    _search_count_var.set(0)


def increment_search_count() -> int:
    """递增搜索计数并返回当前值"""
    current = _search_count_var.get(0) + 1
    _search_count_var.set(current)
    return current


def get_search_count() -> int:
    """获取当前搜索计数"""
    return _search_count_var.get(0)


def cached_tool(ttl: int = 300, include_agent_id: bool = True):
    """工具缓存装饰器
    
    Args:
        ttl: 缓存有效期（秒），web_search 默认 5 分钟，文档搜索默认 2 分钟
        include_agent_id: 是否将 agent_id 纳入缓存 key（默认 True）
            - True: 缓存 key 包含 agent_id，确保不同智能体的知识库搜索结果互不干扰
            - False: 缓存 key 不包含 agent_id，适用于与智能体无关的工具（如 web_search）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 关键修复：将 agent_id 纳入缓存 key，避免不同智能体共享缓存结果
            # 例如：智能体A搜索"FMEA"和智能体B搜索"FMEA"应返回不同的结果
            agent_id_for_cache = get_current_agent_id() if include_agent_id else ""
            cache_key = _tool_cache._make_key(func.__name__, args, kwargs, agent_id=agent_id_for_cache)
            cached = _tool_cache.get(cache_key)
            if cached is not None:
                logger.info(f"工具缓存命中: {func.__name__} (agent_id={agent_id_for_cache})")
                return cached
            result = func(*args, **kwargs)
            # 【修复】错误结果不缓存：避免一次失败导致2分钟内所有搜索都返回同一错误
            if isinstance(result, str) and result.startswith("【检索失败】"):
                logger.warning(f"工具返回错误，不缓存: {func.__name__} -> {result[:100]}")
                return result
            _tool_cache.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator


# ===== 联网搜索工具 =====
@tool
@cached_tool(ttl=300, include_agent_id=False)  # [#8] web_search 缓存 5 分钟（与智能体无关）
def web_search_tool(query: str) -> str:
    """搜索互联网获取实时信息。当你需要最新资讯、实时数据、或知识库中没有的信息时使用此工具。

    【用途】搜索互联网上的最新信息、新闻、实时数据等。
    【典型问题】「最新新闻」「今天天气」「某产品最新价格」「最新技术动态」「实时汇率」
    【不适用】查公司制度文档（用search_documents_tool）、查员工信息（用lookup_employee_tool）。

    Args:
        query: 搜索查询关键词。示例：「2024年最新AI技术动态」「北京今天天气」
    """
    try:
        import httpx
        from urllib.parse import quote_plus

        # 使用百度搜索（国内最稳定）
        search_url = f"https://www.baidu.com/s?wd={quote_plus(query)}&rn=5"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        with httpx.Client(headers=headers, follow_redirects=True, timeout=15) as client:
            resp = client.get(search_url)
            html = resp.text

        # 解析百度搜索结果
        results = []

        # 方法1：从 h3 标签提取标题和链接
        h3_pattern = re.compile(r'<h3[^>]*class="[^"]*t[^"]*"[^>]*>(.*?)</h3>', re.DOTALL)
        for match in h3_pattern.finditer(html):
            block = match.group(1)
            title = re.sub(r'<[^>]+>', '', block).strip()
            href_match = re.search(r'href="(https?://[^"]+)"', block)
            href = href_match.group(1) if href_match else ""
            if title:
                results.append({"title": title, "href": href, "snippet": ""})

        # 方法2：如果方法1没有结果，尝试更宽松的匹配
        if not results:
            h3_all = re.compile(r'<h3[^>]*>(.*?)</h3>', re.DOTALL)
            for match in h3_all.finditer(html):
                block = match.group(1)
                title = re.sub(r'<[^>]+>', '', block).strip()
                href_match = re.search(r'href="(https?://[^"]+)"', block)
                href = href_match.group(1) if href_match else ""
                if title and len(title) > 3:
                    results.append({"title": title, "href": href, "snippet": ""})

        # 提取摘要
        abstract_pattern = re.compile(r'class="c-abstract[^"]*"[^>]*>(.*?)</(?:span|div|p)>', re.DOTALL)
        abstracts = [re.sub(r'<[^>]+>', '', m.group(1)).strip() for m in abstract_pattern.finditer(html)]

        for i, r in enumerate(results):
            if i < len(abstracts) and abstracts[i]:
                r["snippet"] = abstracts[i]

        if not any(r["snippet"] for r in results):
            snippet_pattern = re.compile(r'<span class="content-right_[^"]*">(.*?)</span>', re.DOTALL)
            snippets = [re.sub(r'<[^>]+>', '', m.group(1)).strip() for m in snippet_pattern.finditer(html)]
            for i, r in enumerate(results):
                if i < len(snippets) and snippets[i]:
                    r["snippet"] = snippets[i]

        if not results:
            return "【联网搜索】未找到相关结果。建议：1）尝试换用不同关键词搜索；2）检查网络连接是否正常。"

        output = f"【联网搜索】共找到 {len(results)} 条相关结果：\n\n"
        for i, r in enumerate(results[:5], 1):
            output += f"<web_result index=\"{i}\">\n"
            output += f"  标题：{r['title']}\n"
            if r['snippet']:
                output += f"  摘要：{r['snippet']}\n"
            if r['href']:
                output += f"  链接：{r['href']}\n"
            output += f"</web_result>\n\n"

        return output
    except Exception as e:
        return f"【联网搜索】搜索失败: {str(e)}\n建议：检查网络连接是否正常，或稍后重试。"


def _load_employees():
    """加载员工数据"""
    employees_file = settings.EMPLOYEES_FILE
    if not os.path.exists(employees_file):
        return None
    with open(employees_file, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
@cached_tool(ttl=300)  # [#8] 文档搜索缓存 5 分钟（与web_search一致，减少同会话重复搜索）
def search_documents_tool(query: str) -> str:
    """搜索公司文档知识库，检索与查询语义相关的文档片段。

    [#9] 采用混合检索策略：向量语义检索 + 关键词匹配，提升检索准确率
    [#10] 返回结果标注文档来源和段落位置，支持引用溯源

    【用途】查询公司制度、流程、规范、政策、规定等文档内容。
    【不适用】查员工信息（用lookup_employee_tool）、查看文档列表（用list_documents_tool）。

    Args:
        query: 搜索查询关键词。
               示例：「年假制度」「报销流程」「考勤规定」
    """
    # [#9] 混合检索：先向量搜索，再用关键词补充
    # 按 agent_id 隔离知识库：智能体只搜索自己的知识库
    # 普通聊天模式（agent_id=None）没有知识库
    current_aid = get_current_agent_id()
    logger.debug(f"搜索文档: query={query}, agent_id={current_aid}")

    # 普通聊天模式没有知识库
    if not current_aid:
        return "【检索结果】当前是普通聊天模式，没有关联的知识库。如需搜索文档，请先选择一个智能体。"
    
    # ===== 搜索效率控制：同一轮对话最多搜索 _MAX_SEARCH_PER_TURN 次 =====
    current_count = increment_search_count()
    if current_count > _MAX_SEARCH_PER_TURN:
        logger.info(f"搜索效率控制：本轮已搜索 {current_count-1} 次，超过上限 {_MAX_SEARCH_PER_TURN}，提示LLM直接回答")
        return f"【检索提示】已搜索{current_count-1}次，请直接基于已有结果回答，不要再搜索。"
    
    try:
        results = search_documents(query, top_k=8, agent_id=current_aid)  # 保持8，过多结果增加上下文长度拖慢LLM
    except Exception as e:
        error_str = str(e)
        logger.error(f"search_documents 异常: {error_str}", exc_info=True)
        if '429' in error_str or '余额' in error_str or '1113' in error_str:
            return f"【检索失败】Embedding API 余额不足（429错误），向量搜索不可用。建议用户充值智谱API余额。当前仅使用关键词检索，结果可能不完整。如需获取完整文档内容，请使用 get_document_content_tool 工具。"
        # 【修复】最后兜底：尝试磁盘文件搜索，而不是直接报错
        try:
            from app.rag.document import _search_disk_files
            fallback_results = _search_disk_files(query, top_k=5, agent_id=current_aid)
            if fallback_results:
                results = fallback_results
            else:
                return f"【检索失败】搜索出错: {error_str}。建议使用 get_document_content_tool 直接获取文档全文，或用 list_documents_tool 查看可用文档。"
        except Exception:
            return f"【检索失败】搜索出错: {error_str}。建议使用 get_document_content_tool 直接获取文档全文，或用 list_documents_tool 查看可用文档。"

    if not results:
        return f"【检索结果】未找到与查询相关的文档内容（当前搜索的知识库: agent_id={current_aid}）。建议：1）尝试换用不同关键词搜索；2）确认相关文档是否已上传至对应智能体的知识库。"

    # [#9] 简单重排序：关键词匹配度 + 向量相似度加权
    query_keywords = set(query.replace("？", "").replace("？", "").replace("的", "").replace("了", "").replace("是", "").replace("什么", ""))
    for r in results:
        keyword_score = sum(1 for kw in query_keywords if kw in r.get("content", "")) / max(len(query_keywords), 1)
        vector_score = r.get("relevance_score", 0.5)
        r["final_score"] = 0.4 * keyword_score + 0.6 * vector_score

    # 按综合分数排序
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    results = results[:3]  # 取 top 3（保持精简，过多结果增加上下文token拖慢LLM）

    output = f"【检索结果】共找到 {len(results)} 条相关内容：\n\n"
    for i, r in enumerate(results, 1):
        source = r.get('source', '未知来源')
        relevance = r.get('relevance_score', 0)
        content = r.get('content', '')
        # [#10] 引用溯源：标注文档名 + 段落位置
        # 提取内容的前30字作为段落定位
        content_preview = content[:50].replace('\n', ' ').strip()
        output += f"<document source=\"{source}\" relevance=\"{relevance:.2f}\" citation=\"{source} · {content_preview}...\">\n"
        output += f"{content}\n"
        output += f"</document>\n\n"

    # [#10] 添加引用说明
    sources = list(set(r.get('source', '') for r in results if r.get('source')))
    if sources:
        output += f"【引用来源】{', '.join(sources)}\n"

    return output


@tool
def lookup_employee_tool(name: str = "", department: str = "") -> str:
    """查询公司员工信息。不传参数则列出全部员工，传参数则按条件筛选。

    【用途】查询员工姓名、部门、职位、联系方式等人员信息。
    【典型问题】「所有员工」「张三的信息」「技术部有哪些人」「公司有哪些部门的人」
    【不适用】查公司制度文档（用search_documents_tool）、查文档列表（用list_documents_tool）。

    Args:
        name: 员工姓名（可选，支持模糊匹配）。示例：「张」可匹配「张三」「张伟」
        department: 部门名称（可选，支持模糊匹配）。示例：「技术」可匹配「技术部」
    """
    employees = _load_employees()

    if employees is None:
        return "【系统提示】员工数据库暂未初始化，请先运行 scripts/seed_data.py 初始化数据。"

    results = employees

    # 按姓名过滤
    if name:
        results = [e for e in results if name in e.get("name", "")]

    # 按部门过滤
    if department:
        results = [e for e in results if department in e.get("department", "")]

    if not results:
        return f"【查询结果】未找到匹配的员工信息。搜索条件：姓名=\"{name}\"，部门=\"{department}\"\n建议：检查姓名/部门名称是否正确，或尝试使用部分关键词搜索。你也可以不传参数查看全部员工列表。"

    # 生成部门统计摘要
    dept_count = {}
    for e in results:
        dept = e.get("department", "未知")
        dept_count[dept] = dept_count.get(dept, 0) + 1

    output = f"【查询结果】共找到 {len(results)} 位员工"
    if dept_count:
        dept_summary = "、".join([f"{d} {c}人" for d, c in dept_count.items()])
        output += f"（{dept_summary}）"
    output += "：\n\n"

    for e in results:
        output += f"<employee>\n"
        output += f"  姓名：{e['name']}\n"
        output += f"  部门：{e['department']}\n"
        output += f"  职位：{e['position']}\n"
        output += f"  邮箱：{e['email']}\n"
        if e.get("phone"):
            output += f"  电话：{e['phone']}\n"
        output += f"</employee>\n\n"

    return output


@tool
def list_departments_tool() -> str:
    """列出公司所有部门及各部门人数。

    【用途】当用户想知道公司有哪些部门、各部门有多少人时使用。
    【典型问题】「公司有哪些部门」「部门列表」「都有什么部门」。
    """
    employees = _load_employees()

    if employees is None:
        return "【系统提示】员工数据库暂未初始化，请先运行 scripts/seed_data.py 初始化数据。"

    # 统计部门
    dept_employees = {}
    for e in employees:
        dept = e.get("department", "未知")
        if dept not in dept_employees:
            dept_employees[dept] = []
        dept_employees[dept].append(e['name'])

    if not dept_employees:
        return "【查询结果】暂无部门信息。"

    output = f"【部门列表】公司共有 {len(dept_employees)} 个部门，{len(employees)} 位员工：\n\n"
    for i, (dept, names) in enumerate(dept_employees.items(), 1):
        output += f"  {i}. **{dept}**（{len(names)}人）：{'、'.join(names)}\n"

    output += f"\n如需查看某部门员工的详细信息，请告诉我部门名称。"

    return output


@tool
def list_documents_tool() -> str:
    """列出知识库中所有已索引的文档。

    【用途】查看知识库中有哪些可搜索的文档。
    【典型问题】「知识库有哪些文档」「文档列表」「你们有什么资料」。
    【不适用】查员工信息（用lookup_employee_tool）、查公司制度内容（用search_documents_tool）。
    """
    current_aid = get_current_agent_id()
    logger.debug(f"文档列表: agent_id={current_aid}")

    # 普通聊天模式没有知识库
    if not current_aid:
        return "【文档列表】当前是普通聊天模式，没有关联的知识库。如需管理文档，请先选择一个智能体。"

    docs = list_indexed_documents(agent_id=current_aid)

    if not docs:
        return f"【文档列表】知识库中暂无文档（当前搜索的知识库: agent_id={current_aid}）。请先通过上传功能添加文档。"

    output = f"【文档列表】知识库中共有 {len(docs)} 个文档：\n\n"
    for i, doc in enumerate(docs, 1):
        ext = doc.rsplit('.', 1)[-1].lower() if '.' in doc else ''
        type_label = {'pdf': 'PDF文档', 'docx': 'Word文档', 'txt': '文本文件'}.get(ext, '文档')
        output += f"  {i}. {doc}（{type_label}）\n"

    return output


@tool
def upload_document_tool(file_path: str) -> str:
    """将新文档上传并索引到知识库，使其可被搜索。

    【用途】当用户需要添加新文档到知识库时使用。
    支持格式：PDF、TXT、DOCX。

    Args:
        file_path: 要上传的文档文件路径，必须是已存在于服务器上的文件。
    """
    if not os.path.exists(file_path):
        return f"【上传失败】文件不存在：{file_path}\n请确认文件路径是否正确，或先通过界面功能上传文件。"

    current_aid = get_current_agent_id()
    if not current_aid:
        return "【上传失败】当前是普通聊天模式，没有关联的知识库。如需上传文档，请先选择一个智能体。"

    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    supported = ['pdf', 'txt', 'docx', 'md', 'xlsx', 'xls']
    if ext not in supported:
        return f"【上传失败】不支持的文件格式：.{ext}。目前支持：{', '.join(['.'+e for e in supported])}"

    try:
        result = index_document(file_path, agent_id=current_aid)
        return f"【上传成功】文档已索引到知识库。{result['message']}"
    except Exception as e:
        return f"【上传失败】{str(e)}\n可能原因：文件损坏、内容为空或格式异常。请检查文件后重试。"


@tool
def get_document_content_tool(filename: str) -> str:
    """获取知识库中指定文档的完整内容。直接从原始文件读取，不依赖向量搜索，不会消耗embedding额度。

    【用途】当需要查看或获取某个文档的完整内容时使用。修改文档前应先用此工具获取完整内容。
    【典型问题】「显示xxx文档的完整内容」「获取xxx文档全文」「查看xxx文档」
    【与search_documents_tool的区别】
    - search_documents_tool：搜索知识库，返回与查询相关的文档片段（500字/片），适合查找特定信息
    - get_document_content_tool：返回指定文档的完整全文，适合需要整体查看或修改文档的场景
    【重要】修改文档前，请先调用此工具获取完整内容，修改后再调用modify_document_tool保存。

    Args:
        filename: 文档文件名（含扩展名），需与知识库中的文件名完全一致。
                  示例：「员工手册.pdf」「FMEA新版手册.docx」
    """
    current_aid = get_current_agent_id()
    if not current_aid:
        return "【获取失败】当前是普通聊天模式，没有关联的知识库。如需查看文档，请先选择一个智能体。"
    result = get_document_content(filename, agent_id=current_aid)
    
    if result["status"] == "not_found":
        return f"【获取失败】文档 \"{filename}\" 在服务器上未找到。\n提示：请确认文件名是否正确（需包含扩展名），可通过 list_documents_tool 查看当前文档列表。"
    if result["status"] == "empty":
        return f"【获取失败】文档 \"{filename}\" 内容为空。"
    if result["status"] == "error":
        return f"【获取失败】{result['message']}"
    
    # 成功：返回完整内容
    output = f"【文档内容】{filename}（共 {result['char_count']} 字符）\n\n"
    output += result["content"]
    return output


@tool
def delete_document_tool(filename: str) -> str:
    """从知识库中删除指定文档，同时移除其所有向量分块和原始文件。此操作不可恢复。

    【用途】当用户确认要删除某个文档时使用。
    注意：删除操作不可逆，请在调用前确认用户已明确指定要删除的文档名称。

    Args:
        filename: 要删除的文档文件名（含扩展名），需与知识库中的文件名完全一致。
                  示例：「员工手册.pdf」而非「员工手册」
    """
    current_aid_del = get_current_agent_id()
    if not current_aid_del:
        return "【删除失败】当前是普通聊天模式，没有关联的知识库。如需管理文档，请先选择一个智能体。"
    try:
        result = delete_document(filename, agent_id=current_aid_del)
        if result["status"] == "not_found":
            return f"【删除失败】文档 \"{filename}\" 在知识库中未找到。\n提示：请确认文件名是否正确（需包含扩展名），可通过 list_documents_tool 查看当前文档列表。"
        return f"【删除成功】{result['message']}"
    except Exception as e:
        return f"【删除失败】{str(e)}"


@tool
def modify_document_tool(filename: str, content: str, append: bool = False) -> str:
    """修改知识库中已有文档的内容。支持替换全部内容或在原文末尾追加内容。

    【用途】当用户要求修改、编辑、更新知识库中某个文档的内容时使用。
    【典型问题】「帮我在xxx文件中添加yyy」「把xxx文档里的zzz改成www」「修改知识库的xxx文件」
    【重要】修改后会自动重新索引到向量数据库，无需手动操作。
    【操作流程】替换模式下，请先调用 get_document_content_tool 获取完整内容，在完整内容基础上进行修改，
    然后将修改后的完整内容作为 content 参数传入。不要凭记忆或片段拼凑内容！
    【注意】此工具仅用于修改知识库文档，不生成docx下载文件。如需导出文档，请使用 export_document_tool。

    Args:
        filename: 要修改的文档文件名（含扩展名），需与知识库中的文件名完全一致。
                  示例：「教务处归口管理的校外人员劳务费发放附页-zy.docx」
        content: 新的内容。如果是追加模式，这是要追加到文档末尾的内容；如果是替换模式，这是文档的完整新内容。
        append: 是否追加模式。True=在原文末尾追加内容，False=用新内容替换整个文档（默认False）。
                一般情况下，用户说"添加""追加""加上"用追加模式；用户说"修改""改为""替换"用替换模式。
    """
    current_aid_mod = get_current_agent_id()
    if not current_aid_mod:
        return "【修改失败】当前是普通聊天模式，没有关联的知识库。如需修改文档，请先选择一个智能体。"

    # 追加模式：先读取原文内容，拼接新内容
    final_content = content
    if append:
        try:
            doc_result = get_document_content(filename, agent_id=current_aid_mod)
            if doc_result["status"] == "success":
                original_text = doc_result["content"]
                final_content = original_text + "\n" + content
            else:
                return f"【修改失败】文档 \"{filename}\" 在服务器上未找到。可通过 list_documents_tool 查看当前文档列表。"
        except Exception as e:
            return f"【修改失败】读取原文档内容时出错: {str(e)}"
    else:
        # 替换模式安全检查：防止用少量内容覆盖大量原文
        _original_content_for_verify = ""
        try:
            doc_result = get_document_content(filename, agent_id=current_aid_mod)
            if doc_result["status"] == "success":
                _original_content_for_verify = doc_result["content"]
                original_len = len(doc_result["content"])
                new_len = len(content)
                # 如果新内容不到原文的30%，且原文超过500字，极可能是误操作
                if original_len > 500 and new_len < original_len * 0.3:
                    # 检查新内容是否包含原文的大部分结构（判断是真删减还是误覆盖）
                    # 如果新内容的前50字能在原文中找到，说明LLM保留了原文结构，可能是合理删减
                    content_head = content[:50].strip()
                    if content_head and content_head in doc_result["content"]:
                        logger.warning(f"替换模式大幅删减：原文 {original_len} 字 → 新内容 {new_len} 字，但新内容开头与原文匹配，可能是合理删减，允许执行")
                    else:
                        return f"【修改被拦截】安全检查：原文档共 {original_len} 字，新内容仅 {new_len} 字（不足原文30%）。\n替换模式会用新内容覆盖整个文档，这可能导致原文档大量内容丢失！\n\n如果您确实要大幅删减文档，请先调用 get_document_content_tool 获取完整原文，在原文基础上删减后提交完整内容。\n如果您只是想提取部分内容导出为docx，请使用 export_document_tool 而不是 modify_document_tool。"
        except Exception:
            pass  # 读取失败不阻塞修改流程

    try:
        # 使用同步重索引（async_reindex=False），确保修改后知识库立即可用
        # 虽然稍慢，但避免用户修改后搜索到旧内容
        result = update_document(filename, final_content, agent_id=current_aid_mod, async_reindex=False)
        if result["status"] == "not_found":
            return f"【修改失败】文档 \"{filename}\" 在知识库中未找到。\n提示：请确认文件名是否正确（需包含扩展名），可通过 list_documents_tool 查看当前文档列表。"
        if result["status"] == "error":
            return f"【修改失败】{result['message']}"

        # 写入后验证：确保原文核心内容仍然存在
        try:
            verify_result = get_document_content(filename, agent_id=current_aid_mod)
            if verify_result["status"] == "success":
                # 追加模式：原文前100字必须在
                if append and original_text and original_text[:100] not in verify_result["content"]:
                    logger.error(f"⚠️ 追加模式写入后验证失败：原文内容丢失！filename={filename}")
                    update_document(filename, original_text + "\n" + content, agent_id=current_aid_mod, async_reindex=False)
                    return f"【修改成功（已恢复）】{result['message']}（系统检测到原文可能丢失，已自动恢复）"
                # 替换模式：如果有原文参考，检查新内容开头是否一致
                elif not append and _original_content_for_verify:
                    content_head = content[:50].strip()
                    if content_head and content_head in _original_content_for_verify and content_head not in verify_result["content"]:
                        logger.error(f"⚠️ 替换模式写入后验证失败：提交内容与写入内容不一致！filename={filename}")
        except Exception:
            pass

        output = f"【修改成功】{result['message']}"

        return output
    except Exception as e:
        return f"【修改失败】{str(e)}"


@tool
def export_document_tool(content: str, filename: str = "", title: str = "") -> str:
    """将文本内容生成为docx文档并提供下载链接。用于生成综合文档、简略文档、汇总报告等。

    【用途】当用户要求生成一个可下载的文档时使用。
    【典型问题】
    - 「帮我整理一份综合文档」「生成一份汇总报告」
    - 「把知识库的内容整合成一个文档」
    - 「导出为docx文件」「给我一个Word文档」
    - 「生成一个简略版/精简版文档」
    【与modify_document_tool的区别】
    - modify_document_tool：修改知识库中已存在的文档（同时更新知识库索引）
    - export_document_tool：生成新的文档文件供下载（不影响知识库，适合整合/汇总/生成新文档）
    【DOCX内容要求】
    - content中不要包含emoji表情符号，只包含纯文字和章节格式
    - 用户说"不能出现表情包"等要求是对DOCX文档内容的要求，不是对对话回复的要求
    - 【表格必须使用Markdown表格语法】使用 | 列1 | 列2 | 格式，会自动转为Word原生表格
      正确示例：| 部门 | 职责 | 负责人 |
      错误示例：用空格或符号对齐的假表格（如 部门    职责    负责人）
    - 不要用多个空行分隔段落，系统会自动处理段落间距
    - 使用 **粗体** 标记重要文字，会转为Word粗体格式

    Args:
        content: 文档内容（Markdown格式，支持表格/标题/列表/粗体，不要包含emoji）。
        filename: 输出文件名（含扩展名），为空则自动生成。示例：「FMEA团队汇总.docx」
        title: 文档标题，为空则使用文件名。示例：「FMEA团队信息汇总」
    """
    try:
        if not filename:
            filename = f"export_{int(time.time())}.docx"
        if not filename.endswith('.docx'):
            filename += '.docx'

        result = export_document_as_docx(content, filename, title=title, session_id=get_current_session_id())
        if result["status"] == "success":
            actual_filename = result.get('filename', filename)
            download_url = f"/api/v1/documents/export-download/{actual_filename}"
            return f"【导出成功】文档已生成：{actual_filename}\n\n下载链接：{download_url}\n\n【重要】请直接将上面的「下载链接」原样展示给用户，不要修改、不要省略URL，不要添加额外说明。"
        else:
            return f"【导出失败】{result.get('message', '未知错误')}"
    except Exception as e:
        return f"【导出失败】{str(e)}"


@tool
def export_xlsx_tool(content: str, filename: str = "", title: str = "") -> str:
    """将文本内容生成为xlsx（Excel）文档并提供下载链接。用于生成表格数据、汇总报表等Excel文件。

    【用途】当用户要求生成一个可下载的Excel文件时使用。
    【典型问题】
    - 「帮我生成一个Excel表格」「导出为xlsx」
    - 「把数据整理成Excel文件」「给我一个表格文件」
    - 「生成一份报表」「导出数据到Excel」
    - 「我要xlsx格式的」「不要docx，要xlsx」
    【与export_document_tool的区别】
    - export_document_tool：生成docx（Word）文档，适合文字报告
    - export_xlsx_tool：生成xlsx（Excel）文档，适合表格数据和报表
    【XLSX内容要求】
    - content中使用Markdown表格语法：| 列1 | 列2 | 列3 |
    - 表格外的文字会保留在对应工作表中（放在表格上方）
    - 不要包含emoji表情符号
    - ⚠️ 避免多Sheet拆分：DFMEA/PFMEA/控制计划等分析类表格，所有内容放在同一个工作表中
    - 项目信息放在表格上方的单独行中（如：项目名称：XXX），不要另建Sheet
    - 严重度(S)/频度(O)/探测度(D)评级标准、AP矩阵等参考内容不需要单独建Sheet，直接省略
    - 不要使用 === Sheet: xxx === 标记拆分多个Sheet，除非用户明确要求多Sheet

    Args:
        content: 文档内容（Markdown格式，使用表格语法组织数据，不要包含emoji）。
        filename: 输出文件名（含扩展名），为空则自动生成。示例：「FMEA团队汇总.xlsx」
        title: 文档标题/工作表名称，为空则使用文件名。示例：「FMEA团队信息」
    """
    try:
        if not filename:
            filename = f"export_{int(time.time())}.xlsx"
        if not filename.endswith('.xlsx'):
            filename = filename.rsplit('.', 1)[0] + '.xlsx'

        result = export_document_as_xlsx(content, filename, title=title, session_id=get_current_session_id())
        if result["status"] == "success":
            actual_filename = result.get('filename', filename)
            download_url = f"/api/v1/documents/export-download/{actual_filename}"
            return f"【导出成功】Excel文档已生成：{actual_filename}\n\n下载链接：{download_url}\n\n【重要】请直接将上面的「下载链接」原样展示给用户，不要修改、不要省略URL，不要添加额外说明。"
        else:
            return f"【导出失败】{result.get('message', '未知错误')}"
    except Exception as e:
        return f"【导出失败】{str(e)}"


# ===== [#12] 外部系统集成工具 =====

@tool
def github_api_tool(action: str, repo: str = "", path: str = "", content: str = "", message: str = "", token: str = "") -> str:
    """与 GitHub 仓库进行交互，支持读取和更新文件。

    【用途】当代码仓库操作需求时使用，如查看仓库内容、更新文件、获取文件内容等。
    【典型问题】「帮我把这个改动推到GitHub」「查看仓库的文件列表」「更新某个文件」

    Args:
        action: 操作类型，支持 "read"（读取文件，大文件截断8000字）, "read_full"（读取完整文件，不截断）, "list"（列出目录内容）, "update"（更新文件）
        repo: 仓库名称，格式 "owner/repo"，示例 "cy556-like/company-doc-agent"
        path: 文件路径，示例 "app/config.py"
        content: 更新文件时的文件内容（仅 action=update 时需要）
        message: 更新文件时的 commit message（仅 action=update 时需要）
        token: GitHub Personal Access Token（可选）。用户在对话中提供时可传入，用于写操作鉴权。未提供时从环境变量 GITHUB_TOKEN 读取。
    """
    import httpx

    # Token 优先级：对话中传入 > 环境变量
    github_token = token or os.getenv("GITHUB_TOKEN", "")
    if not repo:
        return "【GitHub 操作】缺少仓库参数，请提供 repo 参数，格式：owner/repo"

    # 构建请求头：公开仓库的 read/list 不需要 Token，update 操作需要 Token
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    # 写操作（update）必须需要 Token
    if action == "update" and not github_token:
        return "【GitHub 操作】写入操作需要 Token 鉴权。请在对话中提供 Token，或在 .env 中设置 GITHUB_TOKEN。读取公开仓库不需要 Token。"
    base_url = f"https://api.github.com/repos/{repo}"

    try:
        if action == "list":
            url = f"{base_url}/contents/{path}" if path else f"{base_url}/contents"
            resp = httpx.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return f"【GitHub 操作】获取目录失败: {resp.status_code} {resp.text[:200]}"
            items = resp.json()
            if isinstance(items, dict) and items.get("message"):
                return f"【GitHub 操作】{items['message']}"
            output = f"【GitHub 目录】{repo}/{path}:\n\n"
            for item in items[:20]:
                icon = "📁" if item.get("type") == "dir" else "📄"
                output += f"  {icon} {item['name']} ({item.get('type', '')})\n"
            if len(items) > 20:
                output += f"  ... 共 {len(items)} 项\n"
            return output

        elif action in ("read", "read_full"):
            if not path:
                return "【GitHub 操作】读取文件需要提供 path 参数"

            # 对于大文件，使用 GitHub Blob API 避免内容截断
            # GitHub Contents API 对大文件会返回 403 且 base64 有大小限制
            # Blob API 可获取任意大小的文件完整内容
            import base64
            file_content = ""
            sha = ""

            # 先尝试 Contents API（小文件快速获取）
            url = f"{base_url}/contents/{path}"
            resp = httpx.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                sha = data.get("sha", "")
                file_size = data.get("size", 0)
                # 如果文件较大（>100KB），用 Blob API 获取完整内容
                if file_size > 100000:
                    blob_sha = data.get("sha", "")
                    blob_url = f"{base_url}/git/blobs/{blob_sha}"
                    blob_resp = httpx.get(blob_url, headers=headers, timeout=30)
                    if blob_resp.status_code == 200:
                        blob_data = blob_resp.json()
                        file_content = base64.b64decode(blob_data["content"]).decode("utf-8", errors="replace")
                    else:
                        # Blob API 也失败，用 Contents API 尽量获取
                        file_content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                else:
                    file_content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            elif resp.status_code == 403:
                # Contents API 对大文件返回 403，使用 Raw URL 直接获取文件内容
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
                raw_resp = httpx.get(raw_url, headers={"User-Agent": "DocAgent/1.0"}, timeout=30)
                if raw_resp.status_code == 200:
                    file_content = raw_resp.text
                else:
                    return f"【GitHub 操作】读取文件失败: Contents API 403, Raw URL {raw_resp.status_code}"
            else:
                return f"【GitHub 操作】读取文件失败: {resp.status_code} {resp.text[:200]}"

            output = f"【GitHub 文件】{repo}/{path} (sha: {sha[:8] if sha else 'unknown'}..., 共 {len(file_content)} 字符)\n\n"

            # action="read" 时限制返回长度（避免工具输出过长拖慢 Agent），
            # action="read_full" 时返回完整内容
            if action == "read" and len(file_content) > 8000:
                output += file_content[:8000]
                output += f"\n\n... (文件共 {len(file_content)} 字符，已显示前8000字。如需完整内容请使用 action=read_full)"
            else:
                output += file_content

            return output

        elif action == "update":
            if not path or not content:
                return "【GitHub 操作】更新文件需要提供 path 和 content 参数"
            import base64
            # 先获取当前文件的 sha
            url = f"{base_url}/contents/{path}"
            resp = httpx.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                # 文件不存在，创建新文件
                sha = None
            else:
                sha = resp.json().get("sha")

            commit_msg = message or f"Update {path} via DocAgent"
            body = {
                "message": commit_msg,
                "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            }
            if sha:
                body["sha"] = sha

            resp = httpx.put(url, headers=headers, json=body, timeout=15)
            if resp.status_code in (200, 201):
                return f"【GitHub 操作】文件更新成功: {repo}/{path}\nCommit: {commit_msg}"
            else:
                return f"【GitHub 操作】文件更新失败: {resp.status_code} {resp.text[:300]}"

        else:
            return f"【GitHub 操作】不支持的操作: {action}。支持: read, read_full, list, update"

    except Exception as e:
        return f"【GitHub 操作】操作失败: {str(e)}\n提示：读取公开仓库不需要 Token，写入操作才需要配置 GITHUB_TOKEN。"


@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """发送电子邮件通知。

    【用途】当需要发送邮件通知时使用，如发送报告、通知审批结果等。
    【典型问题】「发邮件通知技术部」「给张三发邮件」

    Args:
        to: 收件人邮箱地址，多人用逗号分隔。示例："zhangsan@company.com" 或 "a@co.com,b@co.com"
        subject: 邮件主题
        body: 邮件正文内容
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_host or not smtp_user:
        return "【邮件发送】未配置 SMTP 邮件服务。请在 .env 中设置 SMTP_HOST、SMTP_USER、SMTP_PASS。"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, to.split(","), msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, to.split(","), msg.as_string())

        return f"【邮件发送】邮件已成功发送给 {to}，主题：{subject}"

    except Exception as e:
        return f"【邮件发送】发送失败: {str(e)}\n建议：检查 SMTP 配置是否正确。"


@tool
def database_query_tool(query: str, database: str = "default") -> str:
    """执行 SQL 查询语句（只读），支持查询企业数据库。

    【用途】当需要从数据库中查询业务数据时使用，如订单、库存、销售数据等。
    【典型问题】「查询本月销售额」「库存还剩多少」「最近10笔订单」

    注意：此工具只支持 SELECT 查询，不支持 INSERT/UPDATE/DELETE 等写操作。

    Args:
        query: SQL 查询语句。示例："SELECT * FROM orders WHERE date > '2024-01-01' LIMIT 10"
        database: 数据库名称（可选，默认为 default）
    """
    # 安全检查：只允许 SELECT 语句
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return "【数据库查询】安全限制：仅支持 SELECT 查询，不允许执行 INSERT/UPDATE/DELETE 等写操作。"

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "EXEC"]
    for kw in forbidden:
        if kw in normalized.split():
            return f"【数据库查询】安全限制：检测到禁止的关键字 {kw}，仅支持只读查询。"

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        return "【数据库查询】未配置 DATABASE_URL 环境变量。请在 .env 中设置数据库连接字符串。"

    try:
        import sqlite3

        # 支持 SQLite 和 PostgreSQL
        if db_url.startswith("sqlite"):
            conn = sqlite3.connect(db_url.replace("sqlite:///", ""), timeout=10)
        elif db_url.startswith("postgresql"):
            try:
                import psycopg2
                conn = psycopg2.connect(db_url, connect_timeout=10)
            except ImportError:
                return "【数据库查询】PostgreSQL 驱动未安装，请运行: pip install psycopg2-binary"
        else:
            return f"【数据库查询】不支持的数据库类型: {db_url.split(':')[0]}"

        try:
            cursor = conn.cursor()
            cursor.execute(query)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(50)  # 限制最多返回 50 行

                output = f"【数据库查询】查询成功，返回 {len(rows)} 行：\n\n"
                # 表头
                output += "| " + " | ".join(columns) + " |\n"
                output += "|" + "|".join(["---" for _ in columns]) + "|\n"
                # 数据行
                for row in rows:
                    output += "| " + " | ".join(str(v) if v is not None else "NULL" for v in row) + " |\n"

                if len(rows) == 50:
                    output += "\n（最多显示 50 行，如需更多请添加 LIMIT 条件）"
                return output
            else:
                return "【数据库查询】查询执行成功，无返回结果。"
        finally:
            conn.close()

    except Exception as e:
        return f"【数据库查询】查询失败: {str(e)}\n建议：检查 SQL 语法和数据库连接配置。"


# ===== 导出工具列表 =====

# 基础工具（始终可用）
BASE_TOOLS = [
    search_documents_tool,
    lookup_employee_tool,
    list_departments_tool,
    list_documents_tool,
    get_document_content_tool,
    upload_document_tool,
    delete_document_tool,
    modify_document_tool,
    export_document_tool,
    export_xlsx_tool,
]

# 联网搜索工具（按需启用）
WEB_SEARCH_TOOLS = [
    web_search_tool,
]

# [#12] 外部系统集成工具（按需启用，需配置对应环境变量）
EXTERNAL_TOOLS = [
    github_api_tool,
    send_email_tool,
    database_query_tool,
]

# 全部工具
ALL_TOOLS = BASE_TOOLS + WEB_SEARCH_TOOLS + EXTERNAL_TOOLS


def get_tools(web_search: bool = False):
    """根据参数获取工具列表

    Args:
        web_search: 是否启用联网搜索工具

    Returns:
        工具列表
    """
    if web_search:
        return ALL_TOOLS
    return BASE_TOOLS + EXTERNAL_TOOLS


def get_cache_stats() -> dict:
    """获取工具缓存统计信息"""
    return _tool_cache.stats()
