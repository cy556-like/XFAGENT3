"""
重建知识库索引脚本

用法: python rebuild_index.py [--agent AGENT_ID] [--no-clean] [--force]

默认行为:
  1. 自动清理孤立目录和 ChromaDB collection（不属于任何有效智能体的）
  2. 重建所有有效智能体的索引

选项:
  --agent AGENT_ID    只重建指定智能体的索引
  --no-clean          不清理孤立目录，只跳过
  --force             强制重建所有目录（包括无效智能体），不推荐
"""
import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.document import index_document, delete_agent_collection, _get_collection_name, get_embeddings
from app.agent.storage import ALLOWED_AGENT_IDS, AGENTS_DIR, load_agents


def get_valid_agent_ids():
    """获取所有有效智能体ID = ALLOWED_AGENT_IDS + 用户配置中的智能体"""
    valid_ids = set(ALLOWED_AGENT_IDS)
    
    # 扫描所有用户的智能体配置，收集 agent_id
    if os.path.exists(AGENTS_DIR):
        for fname in os.listdir(AGENTS_DIR):
            if fname.endswith('.json'):
                username = fname[:-5]
                try:
                    agents = load_agents(username)
                    for a in agents:
                        if a.get("id"):
                            valid_ids.add(a["id"])
                except Exception:
                    pass
    
    return valid_ids


def get_agent_dirs(base_dir):
    """获取所有 agent_* 目录"""
    dirs = {}
    for name in os.listdir(base_dir):
        if name.startswith("agent_"):
            agent_dir = os.path.join(base_dir, name)
            if os.path.isdir(agent_dir):
                aid = name.replace("agent_", "", 1)
                dirs[aid] = agent_dir
    return dirs


def clean_orphans(base_dir, chroma_dir):
    """清理已删除智能体的孤立目录和 ChromaDB collection（直接删除，不跳过）"""
    valid_ids = get_valid_agent_ids()
    agent_dirs = get_agent_dirs(base_dir)
    
    # 找出孤立目录
    orphans = []
    for aid, agent_dir in agent_dirs.items():
        if aid not in valid_ids:
            orphans.append((aid, agent_dir))
    
    # 找出孤立 ChromaDB collection
    orphan_collections = []
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        existing = [c.name for c in client.list_collections()]
        for c_name in existing:
            if c_name.startswith("agent_"):
                aid = c_name.replace("agent_", "", 1)
                if aid not in valid_ids:
                    orphan_collections.append((aid, c_name))
    except Exception as e:
        print(f"  检查 ChromaDB 失败: {e}")
    
    if not orphans and not orphan_collections:
        print("没有发现孤立目录或 collection，很干净！")
        return 0
    
    cleaned = 0
    
    if orphans:
        print(f"\n发现 {len(orphans)} 个孤立目录（不属于任何有效智能体），正在删除:")
        for aid, agent_dir in orphans:
            files = os.listdir(agent_dir) if os.path.exists(agent_dir) else []
            print(f"  - agent_{aid}/ ({len(files)} 个文件): {', '.join(files[:5])}")
            try:
                shutil.rmtree(agent_dir)
                print(f"    已删除目录: {agent_dir}")
                cleaned += 1
            except Exception as e:
                print(f"    删除目录失败: {e}")
    
    if orphan_collections:
        print(f"\n发现 {len(orphan_collections)} 个孤立 ChromaDB collection，正在删除:")
        for aid, c_name in orphan_collections:
            print(f"  - {c_name}")
            try:
                result = delete_agent_collection(aid)
                print(f"    已删除: {result.get('message', 'OK')}")
                cleaned += 1
            except Exception as e:
                print(f"    删除失败: {e}")
    
    print(f"\n已清理 {cleaned} 项孤立数据")
    return cleaned


def rebuild_index(base_dir, chroma_dir, target_agent=None, force=False, clean=True):
    """重建知识库索引"""
    valid_ids = get_valid_agent_ids() if not force else None
    agent_dirs = get_agent_dirs(base_dir)
    
    # 第一步：清理孤立数据
    if clean and not force:
        print("\n===== 第一步：清理孤立数据 =====")
        clean_orphans(base_dir, chroma_dir)
        # 清理后重新扫描目录
        agent_dirs = get_agent_dirs(base_dir)
    
    # 第二步：重建索引
    print(f"\n===== 第二步：重建索引 =====")
    count = 0
    failed = 0
    
    for aid, agent_dir in agent_dirs.items():
        # 如果指定了 target_agent，只处理那个
        if target_agent and aid != target_agent:
            continue
        
        # 检查是否为有效智能体
        if not force and valid_ids and aid not in valid_ids:
            print(f"跳过无效智能体: agent_{aid} (不在有效列表中)")
            continue
        
        for fn in os.listdir(agent_dir):
            if fn.endswith((".pdf", ".docx", ".txt", ".md")):
                fp = os.path.join(agent_dir, fn)
                print(f"indexing: {aid}/{fn}")
                try:
                    index_document(fp, agent_id=aid)
                    count += 1
                except Exception as e:
                    print(f"  failed: {e}")
                    failed += 1
    
    # 全局目录
    for fn in os.listdir(base_dir):
        if fn.endswith((".pdf", ".docx", ".txt", ".md")):
            fp = os.path.join(base_dir, fn)
            print(f"indexing(global): {fn}")
            try:
                index_document(fp)
                count += 1
            except Exception as e:
                print(f"  failed: {e}")
                failed += 1
    
    print(f"Done! {count} docs indexed, {failed} failed")
    return count


def main():
    parser = argparse.ArgumentParser(description="重建知识库索引（默认自动清理孤立数据）")
    parser.add_argument("--agent", help="只重建指定智能体的索引")
    parser.add_argument("--no-clean", action="store_true", help="不清理孤立目录，只跳过")
    parser.add_argument("--force", action="store_true", help="强制重建所有目录（包括无效智能体），不推荐")
    args = parser.parse_args()
    
    # 路径自动检测
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data", "documents")
    chroma_dir = os.path.join(script_dir, "data", "chroma_db")
    
    # 从 settings 获取路径（如果可用）
    try:
        from app.config import settings
        base_dir = settings.DOCUMENTS_DIR
        chroma_dir = settings.CHROMA_DIR
    except Exception:
        pass
    
    print(f"文档目录: {base_dir}")
    print(f"ChromaDB目录: {chroma_dir}")
    
    # 显示有效智能体
    valid_ids = get_valid_agent_ids()
    print(f"有效智能体: {valid_ids}")
    
    # 显示磁盘上的 agent 目录
    agent_dirs = get_agent_dirs(base_dir)
    print(f"磁盘上的 agent 目录: {len(agent_dirs)} 个")
    for aid, d in agent_dirs.items():
        marker = "有效" if aid in valid_ids else "孤立"
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        print(f"  - agent_{aid}/ [{marker}] 文件: {', '.join(files[:3])}")
    
    rebuild_index(base_dir, chroma_dir, target_agent=args.agent, force=args.force, clean=not args.no_clean)


if __name__ == "__main__":
    main()
