"""
文档处理与向量化模块 (RAG)
负责：加载文档 → 分块 → 向量化 → 存入 ChromaDB → 检索
优化：单例缓存 Embeddings 和 VectorStore 实例，避免重复创建
"""
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.config import settings


# ===== 单例缓存 =====
_vector_store = None
_embeddings = None


def get_embeddings():
    """获取 Embedding 模型（单例缓存，避免重复创建）"""
    global _embeddings
    if _embeddings is None:
        embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'embedding-3')
        _embeddings = OpenAIEmbeddings(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=embedding_model,
        )
    return _embeddings


def get_vector_store():
    """获取 ChromaDB 向量数据库实例（单例缓存，避免重复创建）"""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        _vector_store = Chroma(
            persist_directory=settings.CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _vector_store


def load_document(file_path: str) -> list:
    """
    根据文件类型加载文档
    支持：PDF、TXT、DOCX
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF/TXT/DOCX")

    return loader.load()


def read_document_content(file_path: str) -> str:
    """
    读取文档的纯文本内容（用于文档修改功能）

    Args:
        file_path: 文档路径

    Returns:
        str: 文档纯文本内容
    """
    docs = load_document(file_path)
    content = "\n\n".join([doc.page_content for doc in docs])
    return content


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    文档分块
    - chunk_size: 每块最大字符数
    - chunk_overlap: 块间重叠字符数（保证上下文连续性）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def index_document(file_path: str, filename: str = None) -> dict:
    """
    完整的文档索引流程：加载 → 分块 → 向量化 → 存储

    Returns:
        dict: 包含分块数量和状态信息
    """
    if filename is None:
        filename = os.path.basename(file_path)

    # 1. 加载文档
    docs = load_document(file_path)

    # 2. 给文档添加元数据
    for doc in docs:
        doc.metadata["source_file"] = filename

    # 3. 分块
    chunks = split_documents(docs)

    # 4. 向量化并存储
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return {
        "filename": filename,
        "chunks": len(chunks),
        "status": "success",
        "message": f"文档 {filename} 已成功索引，共 {len(chunks)} 个分块",
    }


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    在向量数据库中检索与查询最相关的文档片段

    Args:
        query: 用户查询
        top_k: 返回最相关的 K 个结果

    Returns:
        list[dict]: 检索结果列表
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=top_k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "未知来源"),
            "relevance_score": round(1 - score, 4),  # 转换为相似度
        })

    return formatted


def list_indexed_documents() -> list[str]:
    """列出知识库中所有已索引的文档"""
    vector_store = get_vector_store()
    # 从 ChromaDB 的元数据中提取所有文档名
    try:
        collection = vector_store._collection
        all_docs = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_docs["metadatas"]:
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        return sorted(list(sources))
    except Exception:
        return []
