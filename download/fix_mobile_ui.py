"""
手机端UI适配修复脚本
在ECS上运行: python fix_mobile_ui.py
会自动读取 index.html 并注入移动端适配代码
"""

import os
import re

# 目标文件路径
HTML_PATH = r"C:\Users\Administrator\Downloads\company-doc-agent\app\static\index.html"

def fix_mobile_ui():
    if not os.path.exists(HTML_PATH):
        print(f"错误: 找不到文件 {HTML_PATH}")
        return

    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    # ========== 1. 确保有 viewport meta 标签 ==========
    viewport_tag = '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">'
    if 'viewport' not in html:
        # 在 <head> 后面插入
        html = html.replace('<head>', '<head>\n    ' + viewport_tag, 1)
        print("[OK] 添加了 viewport meta 标签")
    else:
        print("[跳过] viewport meta 标签已存在")

    # ========== 2. 注入移动端 CSS ==========
    mobile_css = '''
    /* ========== 移动端适配 ========== */
    /* 汉堡菜单按钮 - 仅移动端显示 */
    .mobile-menu-btn {
        display: none;
        position: fixed;
        top: 12px;
        left: 12px;
        z-index: 1001;
        background: #40414f;
        border: 1px solid #565869;
        color: #fff;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 20px;
        align-items: center;
        justify-content: center;
        padding: 0;
        line-height: 1;
    }
    .mobile-menu-btn:hover {
        background: #565869;
    }
    /* 移动端遮罩层 */
    .sidebar-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 998;
    }
    .sidebar-overlay.active {
        display: block;
    }

    @media (max-width: 768px) {
        /* 汉堡菜单按钮显示 */
        .mobile-menu-btn {
            display: flex !important;
        }

        /* 侧边栏变为抽屉式 */
        .sidebar {
            position: fixed !important;
            left: -280px !important;
            top: 0 !important;
            height: 100vh !important;
            z-index: 999 !important;
            transition: left 0.3s ease !important;
            width: 260px !important;
            min-width: 260px !important;
        }
        .sidebar.open {
            left: 0 !important;
        }

        /* 主内容区域全宽 */
        .main-content, .chat-area, .chat-container {
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        /* 顶部留出汉堡菜单空间 */
        .chat-header, header, .header {
            padding-left: 56px !important;
        }

        /* 聊天消息区域 */
        .chat-messages, .messages-container {
            padding: 10px !important;
            font-size: 15px !important;
        }

        /* 消息气泡 */
        .message, .chat-message {
            max-width: 100% !important;
            padding: 12px !important;
            margin-bottom: 10px !important;
            font-size: 15px !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }

        /* 输入区域适配 */
        .chat-input-area, .input-area, .input-container {
            padding: 8px 10px !important;
            gap: 8px !important;
        }
        .chat-input-area textarea, .input-area textarea, 
        .chat-input-area input[type="text"], .input-area input[type="text"] {
            font-size: 16px !important;  /* 防止iOS自动缩放 */
            min-height: 44px !important;
            padding: 10px 14px !important;
        }

        /* 发送按钮 */
        .send-btn, button[type="submit"] {
            min-width: 44px !important;
            min-height: 44px !important;
        }

        /* 新建聊天按钮 */
        .new-chat-btn, .new-chat {
            padding: 12px !important;
            font-size: 14px !important;
        }

        /* 聊天列表项 */
        .chat-item, .chat-list-item {
            padding: 12px !important;
            font-size: 14px !important;
        }

        /* 模态框适配 */
        .modal, .modal-content {
            width: 95% !important;
            max-width: 95% !important;
            margin: 10px auto !important;
            max-height: 85vh !important;
            padding: 16px !important;
        }

        /* 知识库模态框 */
        .knowledge-modal, .kb-modal {
            width: 95% !important;
            max-height: 85vh !important;
        }

        /* 登录/注册页面 */
        .login-container, .auth-container, .login-form {
            width: 95% !important;
            max-width: 400px !important;
            padding: 24px 20px !important;
            margin: 20px auto !important;
        }
        .login-container input, .auth-container input {
            font-size: 16px !important;  /* 防止iOS自动缩放 */
            padding: 12px !important;
        }

        /* 欢迎页面 */
        .welcome-screen, .welcome {
            padding: 20px !important;
        }
        .welcome-screen h1, .welcome h1 {
            font-size: 24px !important;
        }
        .welcome-screen p, .welcome p {
            font-size: 15px !important;
        }

        /* 文件上传区域 */
        .upload-area, .file-drop-zone {
            padding: 20px !important;
            min-height: 120px !important;
        }

        /* 代码块适配 */
        pre, code {
            font-size: 13px !important;
            overflow-x: auto !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
        }

        /* 工具调用结果 */
        .tool-result, .tool-call {
            font-size: 13px !important;
            padding: 10px !important;
        }
    }

    /* 小屏手机 (<= 400px) */
    @media (max-width: 400px) {
        .sidebar {
            width: 220px !important;
            min-width: 220px !important;
        }
        .message, .chat-message {
            font-size: 14px !important;
            padding: 10px !important;
        }
        .modal, .modal-content {
            width: 98% !important;
            padding: 12px !important;
        }
    }
    '''

    # 检查是否已经注入过
    if '移动端适配' in html or 'mobile-menu-btn' in html:
        print("[跳过] 移动端CSS已存在，跳过注入")
    else:
        # 在 </style> 前注入CSS
        if '</style>' in html:
            html = html.replace('</style>', mobile_css + '\n    </style>', 1)
            print("[OK] 注入了移动端CSS")
        else:
            # 没有 style 标签，在 </head> 前添加
            html = html.replace('</head>', '<style>' + mobile_css + '</style>\n</head>', 1)
            print("[OK] 添加了移动端CSS样式块")

    # ========== 3. 注入汉堡菜单按钮 HTML ==========
    hamburger_html = '''<button class="mobile-menu-btn" onclick="toggleSidebar()" title="菜单">☰</button>
    <div class="sidebar-overlay" onclick="toggleSidebar()"></div>'''

    if 'mobile-menu-btn' not in html or 'toggleSidebar' not in html:
        # 在 <body> 后插入
        if '<body>' in html:
            html = html.replace('<body>', '<body>\n    ' + hamburger_html, 1)
            print("[OK] 注入了汉堡菜单按钮和遮罩层")
        else:
            print("[警告] 未找到 <body> 标签，请手动添加汉堡菜单")
    else:
        print("[跳过] 汉堡菜单按钮已存在")

    # ========== 4. 注入 sidebar toggle JavaScript ==========
    toggle_js = '''
    // 移动端侧边栏切换
    function toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        const overlay = document.querySelector('.sidebar-overlay');
        if (sidebar) {
            sidebar.classList.toggle('open');
        }
        if (overlay) {
            overlay.classList.toggle('active');
        }
    }
    
    // 点击聊天项后自动关闭侧边栏
    document.addEventListener('DOMContentLoaded', function() {
        const chatItems = document.querySelectorAll('.chat-item, .chat-list-item');
        chatItems.forEach(item => {
            item.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    setTimeout(() => toggleSidebar(), 200);
                }
            });
        });

        // 新建聊天后关闭侧边栏
        const origNewChat = window.newChat || window.createNewChat;
        if (origNewChat) {
            window.newChat = function() {
                origNewChat();
                if (window.innerWidth <= 768) {
                    setTimeout(() => toggleSidebar(), 200);
                }
            };
        }
    });
    '''

    if 'toggleSidebar' not in html:
        if '</script>' in html:
            # 在最后一个 </script> 前注入
            last_script = html.rfind('</script>')
            html = html[:last_script] + toggle_js + '\n' + html[last_script:]
            print("[OK] 注入了侧边栏切换JavaScript")
        else:
            # 在 </body> 前添加
            html = html.replace('</body>', '<script>' + toggle_js + '</script>\n</body>', 1)
            print("[OK] 添加了侧边栏切换JavaScript脚本")
    else:
        print("[跳过] 侧边栏切换JS已存在")

    # ========== 5. 保存文件 ==========
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 修复完成! 文件已保存到: {HTML_PATH}")
    print("请重启项目: python -m app.main")
    print("然后用手机访问 http://47.114.99.132:8000 测试效果")

if __name__ == '__main__':
    fix_mobile_ui()
