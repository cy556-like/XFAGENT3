"""
一键同步脚本：从 GitHub API 下载最新文件到本地 ECS
用法: python sync_from_github.py
"""
import urllib.request
import json
import base64
import os
import sys

# ===== 配置 =====
TOKEN = "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"
REPO = "cy556-like/XFAGENT3"
BRANCH = "main"
# 项目根目录（自动检测，优先用脚本所在目录）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要同步的文件列表（相对于项目根目录）
SYNC_FILES = [
    # ===== 核心应用文件 =====
    "app/agent/tools.py",       # 搜索效率优化：计数器+top5+缓存TTL
    "app/agent/core.py",        # MAX_TOOL_ROUNDS 8→5 + 搜索计数重置
    "app/agent/prompts.py",     # 搜索效率规则（提示词层）
    "app/rag/document.py",      # DOCX表格加载 + BM25检索
    "app/docx_export.py",       # DOCX导出列宽优化（如果存在）
    "app/api/routes.py",        # 路由修复（如果存在）
    "rebuild_index.py",         # 重建索引脚本
    "requirements.txt",         # 依赖文件
]


def download_file_github_api(filepath, token, repo, branch):
    """通过 GitHub API 下载单个文件（支持私有仓库）"""
    api_url = f"https://api.github.com/repos/{repo}/contents/{filepath}?ref={branch}"
    
    req = urllib.request.Request(api_url)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "Python-sync-script")
    
    print(f"  下载: {filepath} ...", end=" ", flush=True)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        if data.get("encoding") == "base64" and data.get("content"):
            content = base64.b64decode(data["content"])
            print(f"OK ({len(content)} bytes)")
            return content
        else:
            print("FAILED (unexpected format)")
            return None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"SKIP (文件不存在于仓库)")
        else:
            print(f"FAILED (HTTP {e.code})")
        return None
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def download_file_raw(token, repo, branch, filepath):
    """备选方式：通过 raw.githubusercontent.com 下载"""
    url = f"https://{token}@raw.githubusercontent.com/{repo}/{branch}/{filepath}"
    
    print(f"  下载(raw): {filepath} ...", end=" ", flush=True)
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Python-sync-script")
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        print(f"OK ({len(content)} bytes)")
        return content
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def main():
    print("=" * 60)
    print("GitHub 仓库同步工具")
    print(f"仓库: {REPO}")
    print(f"分支: {BRANCH}")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"同步文件: {len(SYNC_FILES)} 个")
    print("=" * 60)
    
    success = 0
    failed = 0
    skipped = 0
    
    for filepath in SYNC_FILES:
        # 先试 GitHub API，失败再试 raw
        content = download_file_github_api(filepath, TOKEN, REPO, BRANCH)
        
        if content is None:
            # 404 的不需要再试 raw
            print("  尝试 raw 方式...")
            content = download_file_raw(TOKEN, REPO, BRANCH, filepath)
        
        if content is None:
            # 检查文件是否存在本地（可能是不需要的文件）
            target_path = os.path.join(PROJECT_DIR, filepath)
            if not os.path.exists(target_path):
                skipped += 1
                print(f"  ⊘ {filepath} 本地和远程均不存在，跳过")
            else:
                failed += 1
                print(f"  ✗ {filepath} 下载失败，保留本地版本")
            continue
        
        # 写入文件
        target_path = os.path.join(PROJECT_DIR, filepath)
        target_dir = os.path.dirname(target_path)
        
        # 确保目录存在
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"  创建目录: {target_dir}")
        
        # 备份旧文件
        if os.path.exists(target_path):
            backup_path = target_path + ".bak"
            try:
                with open(target_path, "rb") as f:
                    old_content = f.read()
                # 只有内容不同才备份
                if old_content != content:
                    with open(backup_path, "wb") as f:
                        f.write(old_content)
                    print(f"  备份: {filepath} -> {filepath}.bak")
                else:
                    print(f"  (内容未变化，跳过写入)")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"  备份失败: {e}")
        
        # 写入新文件
        try:
            with open(target_path, "wb") as f:
                f.write(content)
            print(f"  ✓ {filepath} 已更新")
            success += 1
        except PermissionError:
            print(f"  ✗ 写入失败: 权限不足，请用 sudo 运行")
            failed += 1
            # 恢复备份
            if os.path.exists(target_path + ".bak"):
                try:
                    with open(target_path + ".bak", "rb") as bf:
                        old = bf.read()
                    with open(target_path, "wb") as f:
                        f.write(old)
                    print(f"  已恢复备份")
                except Exception:
                    pass
        except Exception as e:
            print(f"  ✗ 写入失败: {e}")
            failed += 1
            # 恢复备份
            if os.path.exists(target_path + ".bak"):
                try:
                    with open(target_path + ".bak", "rb") as bf:
                        old = bf.read()
                    with open(target_path, "wb") as f:
                        f.write(old)
                    print(f"  已恢复备份")
                except Exception:
                    pass
    
    print()
    print("=" * 60)
    print(f"同步完成: {success} 更新, {failed} 失败, {skipped} 跳过")
    
    if failed == 0 and success > 0:
        print("\n下一步操作:")
        print("  1. pip install rank_bm25 jieba python-docx  # 安装依赖（如未安装）")
        print("  2. python rebuild_index.py                  # 重建索引（可选）")
        print("  3. 重启服务:")
        print("     - Docker: docker restart <容器名>")
        print("     - K8s:    kubectl rollout restart deployment/<部署名>")
        print("     - 直接运行: 先 Ctrl+C 停止，再 python -m app.main")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
