/**
 * ForgeAgent 前端应用
 * 主脚本 - 处理认证、聊天、会话管理、导出等功能
 */

let currentUser = null;
let authToken = null;
let selectedFile = null;
let selectedFileBase64 = null;
let isLoading = false;
let currentChatId = null;
let allChats = [];
let renamingChatId = null;
let currentAbortController = null;
let userScrolledUp = false;
let lastMessageText = '';
let webSearchEnabled = false;
let deepThinkEnabled = false;
let currentMode = 'agent';
const MAX_FILE_SIZE = 50 * 1024 * 1024;

// [#12] 同步防抖锁：避免短时间内重复调用 syncAgentsFromServer
let _syncAgentsLock = false;
let _syncAgentsLastTime = 0;
const _SYNC_AGENTS_COOLDOWN = 5000;  // 5秒内不重复同步
// [#12] 上次同步到服务器的智能体数据指纹（用于检测数据是否真变了）
let _lastSyncedAgentsHash = '';

// ===== Agent Management =====
// 强制只保留2个允许的智能体
const ALLOWED_AGENT_IDS = ['xf-rd-agent', 'xf-quality-agent'];

function forceCorrectAgents() {
    const existing = JSON.parse(localStorage.getItem('forgeAgents') || '[]');
    const existingMap = {};
    existing.forEach(a => { existingMap[a.id] = a; });

    const defaults = {
        'xf-rd-agent': { name: 'XF模具研发智能体', task: '专注于模具研发设计与工艺优化', summary: '研发设计与工艺优化' },
        'xf-quality-agent': { name: 'XF模具质量智能体', task: '专注于模具质量检测与控制', summary: '质量检测与缺陷控制' }
    };

    const correctAgents = Object.keys(defaults).map(id => {
        const def = defaults[id];
        const ex = existingMap[id];
        return {
            id: id,
            name: ex ? (ex.name || def.name) : def.name,
            task: ex ? (ex.task || def.task) : def.task,
            summary: ex ? (ex.summary || def.summary) : def.summary,
            mode: 'agent',
            icon: id === 'xf-rd-agent' ? '🔧' : '✅',
            created_at: ex ? (ex.created_at || 0) : 0,
            updated_at: ex ? (ex.updated_at || null) : null,
            chat_ids: ex ? (ex.chat_ids || []) : []
        };
    });

    localStorage.setItem('forgeAgents', JSON.stringify(correctAgents));
    return correctAgents;
}

function filterAgents(agents) {
    if (!agents || !Array.isArray(agents)) return forceCorrectAgents();
    const allowedIds = ['xf-rd-agent', 'xf-quality-agent'];
    const filtered = agents.filter(a => allowedIds.includes(a.id));
    // If we already have exactly the 2 correct agents with chat_ids, just return them (preserving chat_ids etc)
    if (filtered.length === 2 && filtered.every(a => a.chat_ids !== undefined)) {
        return filtered;
    }
    // Otherwise, force correct but preserve existing data
    return forceCorrectAgents();
}

let myAgents = filterAgents(JSON.parse(localStorage.getItem('forgeAgents') || 'null'));
let currentAgentId = null;
let agentKbUploadMode = false;

function _resolveMergeDirection(local, serverAgent) {
    // BUG FIX: Improved timestamp-based merge logic for prompt sync across browsers
    // If server has updated_at but local doesn't, prefer server data
    if (serverAgent.updated_at && !local.updated_at) return true;
    // If local has updated_at but server doesn't, prefer local data
    if (local.updated_at && !serverAgent.updated_at) return false;
    // Otherwise compare timestamps
    const localTime = local.updated_at || local.created_at || 0;
    const serverTime = serverAgent.updated_at || serverAgent.created_at || 0;
    return serverTime > localTime;
}

async function saveAgents() {
    // 过滤：只保留允许的智能体
    myAgents = filterAgents(myAgents);
    localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
    // [#12] 同步到服务器：检测数据是否真变了（chat_ids变化不算，服务端不存chat_ids）
    if (currentUser && authToken) {
        try {
            const agentsForServer = myAgents.map(a => ({
                id: a.id, name: a.name, task: a.task, mode: a.mode, created_at: a.created_at, updated_at: a.updated_at
            }));
            const newHash = JSON.stringify(agentsForServer);
            if (newHash === _lastSyncedAgentsHash) {
                console.log('[saveAgents] 数据未变化，跳过POST');
                return;
            }
            _lastSyncedAgentsHash = newHash;
            const resp = await fetch('/api/v1/agents/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authToken },
                body: JSON.stringify({ agents: agentsForServer })
            });
            const data = await resp.json();
            if (data.success && data.agents && data.agents.length > 0) {
                // Merge: preserve local chat_ids, use timestamp-based comparison for name/task/updated_at
                const localAgents = JSON.parse(localStorage.getItem('forgeAgents') || '[]');
                const localMap = {};
                localAgents.forEach(a => { localMap[a.id] = a; });
                const mergedAgents = data.agents.map(serverAgent => {
                    const local = localMap[serverAgent.id];
                    if (!local) return { ...serverAgent, chat_ids: [] };
                    const useServer = _resolveMergeDirection(local, serverAgent);
                    return {
                        ...serverAgent,
                        name: useServer ? serverAgent.name : (local.name || serverAgent.name),
                        task: useServer ? serverAgent.task : (local.task || serverAgent.task),
                        summary: local.summary || serverAgent.summary || '',
                        updated_at: useServer ? (serverAgent.updated_at || null) : (local.updated_at || null),
                        chat_ids: local.chat_ids || []
                    };
                });
                myAgents = filterAgents(mergedAgents);
                localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
            }
        } catch (e) {
            console.warn('[智能体同步失败]', e);
        }
    }
}



async function syncAgentsFromServer(force = false) {
    // [#12] 防抖锁：5秒内不重复同步（除非 force=true）
    if (!force && _syncAgentsLock) return;
    const now = Date.now();
    if (!force && (now - _syncAgentsLastTime) < _SYNC_AGENTS_COOLDOWN) return;
    _syncAgentsLock = true;
    _syncAgentsLastTime = now;

    // 从服务器拉取最新智能体数据并合并（保留本地 chat_ids）
    // 修复跨浏览器同步：先GET服务器数据，再与本地比较，只有本地更新时才POST
    if (!currentUser || !authToken) { _syncAgentsLock = false; return; }
    try {
        // Step 1: GET 服务器最新数据（不发送本地数据，避免旧数据覆盖服务器）
        const getResp = await fetch('/api/v1/agents', {
            method: 'GET',
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const getData = await getResp.json();
        
        if (getData.success && getData.agents && getData.agents.length > 0) {
            const serverAgents = getData.agents;
            const localAgents = JSON.parse(localStorage.getItem('forgeAgents') || '[]');
            const localMap = {};
            localAgents.forEach(a => { localMap[a.id] = a; });
            
            // Step 2: 比较时间戳，合并数据
            let localHasNewer = false;
            const mergedAgents = serverAgents.map(serverAgent => {
                const local = localMap[serverAgent.id];
                if (!local) return { ...serverAgent, chat_ids: [] };
                const useServer = _resolveMergeDirection(local, serverAgent);
                if (!useServer) localHasNewer = true; // 本地有更新的数据
                return {
                    ...serverAgent,
                    name: useServer ? serverAgent.name : (local.name || serverAgent.name),
                    task: useServer ? serverAgent.task : (local.task || serverAgent.task),
                    summary: local.summary || serverAgent.summary || '',
                    updated_at: useServer ? (serverAgent.updated_at || null) : (local.updated_at || null),
                    chat_ids: local.chat_ids || []
                };
            });
            
            myAgents = filterAgents(mergedAgents);
            localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
            
            // Step 3: 只有本地有更新数据时才POST到服务器
            if (localHasNewer) {
                const agentsForServer = myAgents.map(a => ({
                    id: a.id, name: a.name, task: a.task, mode: a.mode, 
                    created_at: a.created_at, updated_at: a.updated_at
                }));
                // [#12] 计算数据指纹，检测是否真变了（避免无变化的写操作）
                const newHash = JSON.stringify(agentsForServer);
                if (newHash !== _lastSyncedAgentsHash) {
                    _lastSyncedAgentsHash = newHash;
                    try {
                        await fetch('/api/v1/agents/sync', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authToken },
                            body: JSON.stringify({ agents: agentsForServer })
                        });
                    } catch (postErr) {
                        console.warn('[智能体POST同步失败]', postErr);
                    }
                } else {
                    console.log('[sync] 数据未变化，跳过POST');
                }
            }
        }

        // Rebuild chat_ids from server data
        await rebuildChatIdsFromServer();
        renderMyAgents();
    } catch (e) {
        console.warn('[智能体同步失败]', e);
    } finally {
        _syncAgentsLock = false;
    }
}
// BUG FIX: Rebuild agent.chat_ids from server chat data to restore agent-chat associations
// after refresh/cross-browser where local chat_ids are lost
async function rebuildChatIdsFromServer() {
    if (!currentUser || !authToken) return;
    try {
        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}`, { headers: apiHeaders() });
        const data = await resp.json();
        console.log('[rebuildChatIds] server chats:', data);
        if (data.success && data.chats) {
            const serverChats = data.chats;
            myAgents.forEach(agent => {
                // Find all chats where chat.agent_id matches this agent's id
                const matchingChatIds = serverChats
                    .filter(chat => chat.agent_id === agent.id)
                    .map(chat => chat.chat_id);
                console.log(`[rebuildChatIds] Agent ${agent.name} (${agent.id}): found ${matchingChatIds.length} chats`);
                // Merge: add any new server chat_ids
                const existingIds = new Set(agent.chat_ids || []);
                matchingChatIds.forEach(id => existingIds.add(id));
                agent.chat_ids = Array.from(existingIds);
            });
            localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
            console.log('[rebuildChatIds] Rebuilt chat_ids from server');
        }
    } catch (e) {
        console.warn('[rebuildChatIds失败]', e);
    }
}

function generateAgentId() {
    return 'agent_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 6);
}

function openAgentCreateModal() {
    document.getElementById('agentName').value = '';
    document.getElementById('agentTask').value = '';
    document.getElementById('agentCreateModal').classList.add('show');
    setTimeout(() => document.getElementById('agentName').focus(), 100);
}

function closeAgentCreateModal() {
    document.getElementById('agentCreateModal').classList.remove('show');
}

async function createAgent() {
    const name = document.getElementById('agentName').value.trim();
    const task = document.getElementById('agentTask').value.trim();
    if (!name) { showToast('请输入智能体名称'); return; }
    if (!task) { showToast('请输入任务描述'); return; }
    
    const agent = {
        id: generateAgentId(),
        name: name,
        task: task,
        mode: 'agent',
        created_at: Date.now() / 1000,
        chat_ids: []
    };
    myAgents.push(agent);
    saveAgents();
    closeAgentCreateModal();
    
    // Switch to the new agent
    await switchToAgent(agent.id);
    renderMyAgents();
    showToast(`智能体「${name}」锻造成功！`);
}

function deleteAgent(agentId) {
    const agent = myAgents.find(a => a.id === agentId);
    if (!agent) return;
    if (!confirm(`确定删除智能体「${agent.name}」？相关对话和知识库也将被删除。`)) return;
    
    // 先删除服务器端的知识库
    fetch(`/api/v1/agents/${encodeURIComponent(agentId)}/knowledge`, { method: 'DELETE', headers: apiHeaders() })
        .then(r => r.json())
        .then(data => console.log('[KB删除]', data))
        .catch(e => console.warn('[KB删除失败]', e));
    
    myAgents = myAgents.filter(a => a.id !== agentId);
    saveAgents();
    
    if (currentAgentId === agentId) {
        currentAgentId = null;
        agentKbUploadMode = false;
        document.getElementById('kbUploadToggle').classList.remove('active');
        document.getElementById('agentKbBar').style.display = 'none';
        modeChatId['agent'] = null;
        document.getElementById('chatTitle').textContent = 'XF模具智能体平台';
        updateKbUploadVisibility();
        updateHeaderKbVisibility();
    }
    renderMyAgents();
    loadChatList();
    showToast('智能体已删除');
}

async function switchToAgent(agentId) {
    const agent = myAgents.find(a => a.id === agentId);
    if (!agent) return;

    currentAgentId = agentId;

    // Force agent mode (智能体强制使用agent模式)
    if (currentMode !== 'agent') {
        switchMode('agent');
    }

    // 智能体模式默认开启联网搜索
    if (!webSearchEnabled) {
        webSearchEnabled = true;
        document.getElementById('webSearchToggle').classList.add('active');
        localStorage.setItem('webSearch', '1');
    }

    // Update header title
    document.getElementById('chatTitle').textContent = agent.name;

    // 更新知识库按钮可见性（选中智能体时显示📚）
    updateKbUploadVisibility();
    updateHeaderKbVisibility();

    // Render agents list
    renderMyAgents();
    
    // Load or create chat for this agent
    if (agent.chat_ids && agent.chat_ids.length > 0) {
        // Try to restore last active chat for this agent
        const lastChatId = agentActiveChatId[agentId];
        if (lastChatId && agent.chat_ids.includes(lastChatId)) {
            currentChatId = lastChatId;
        } else {
            currentChatId = agent.chat_ids[0];
            agentActiveChatId[agentId] = currentChatId;
            saveAgentActiveChatIds();
        }
        modeChatId['agent'] = currentChatId;
        renderChatList();
        await loadChatHistory(currentChatId);
    } else {
        await createNewChat();
    }
}

function renderMyAgents() {
    const list = document.getElementById('myAgentsList');
    if (!list) return;
    list.innerHTML = '';

    myAgents.forEach(agent => {
        const item = document.createElement('div');
        item.className = `agent-item${agent.id === currentAgentId ? ' active' : ''}`;
        item.onclick = (e) => {
            if (e.target.closest('.agent-action-btn')) return;
            switchToAgent(agent.id);
            closeSidebarOnMobile();
        };
        const initial = agent.name[0].toUpperCase();
        const summary = agent.summary || (agent.id === 'xf-rd-agent' ? '研发设计与工艺优化' : '质量检测与缺陷控制');
        item.innerHTML = `
            <div class="agent-item-icon">${initial}</div>
            <div class="agent-item-info">
                <div class="agent-item-name">${escapeHtml(agent.name)}</div>
                <div class="agent-item-task">${escapeHtml(summary)}</div>
            </div>
            <button class="agent-action-btn edit" onclick="event.stopPropagation(); openAgentEditModal('${agent.id}')" title="编辑提示词" aria-label="编辑智能体">✏</button>
        `;
        list.appendChild(item);
    });
}

// ===== Agent Edit =====
let editingAgentId = null;

function openAgentEditModal(agentId) {
    const agent = myAgents.find(a => a.id === agentId);
    if (!agent) return;
    editingAgentId = agentId;
    document.getElementById('editAgentName').value = agent.name;
    document.getElementById('editAgentTask').value = agent.task;
    document.getElementById('agentEditModal').classList.add('show');
    setTimeout(() => document.getElementById('editAgentTask').focus(), 100);
}

function closeAgentEditModal() {
    document.getElementById('agentEditModal').classList.remove('show');
    editingAgentId = null;
}

async function saveAgentEdit() {
    if (!editingAgentId) return;
    const task = document.getElementById('editAgentTask').value.trim();
    if (!task) { showToast('请输入任务描述'); return; }

    const agent = myAgents.find(a => a.id === editingAgentId);
    if (!agent) return;

    // 名称不可编辑，仅更新提示词
    agent.task = task;
    agent.updated_at = Date.now() / 1000;  // Update timestamp for cross-browser sync
    saveAgents();

    closeAgentEditModal();
    renderMyAgents();
    showToast(`智能体「${agent.name}」已更新`);
}

function toggleMyAgents() {
    // No longer a collapsible section - agents are always visible in sidebar
    // This function kept for compatibility but does nothing
}

// ===== Agent KB Upload Toggle & Header KB Button Visibility =====
function updateHeaderKbVisibility() {
    const wrapper = document.getElementById('headerKbWrapper');
    if (!wrapper) return;
    // 只在 agent 模式 且 选中了某个智能体 时才显示 header 知识库按钮
    if (currentMode === 'agent' && currentAgentId) {
        wrapper.style.display = '';
    } else {
        wrapper.style.display = 'none';
        // 同时关闭知识库面板
        const panel = document.getElementById('kbPanel');
        if (panel) panel.classList.remove('show');
    }
}

function updateKbUploadVisibility() {
    const kbBtn = document.getElementById('kbUploadToggle');
    // 只在 agent 模式 且 选中了某个智能体 时才显示知识库上传按钮
    if (currentMode === 'agent' && currentAgentId) {
        kbBtn.style.display = '';
    } else {
        kbBtn.style.display = 'none';
        // 同时关闭知识库上传模式
        if (agentKbUploadMode) {
            agentKbUploadMode = false;
            kbBtn.classList.remove('active');
            document.getElementById('agentKbBar').style.display = 'none';
        }
    }
}

function toggleAgentKbUpload() {
    if (!currentAgentId) {
        showToast('请先选择或创建一个智能体');
        return;
    }
    agentKbUploadMode = !agentKbUploadMode;
    document.getElementById('kbUploadToggle').classList.toggle('active', agentKbUploadMode);
    document.getElementById('kbUploadToggle').setAttribute('aria-pressed', agentKbUploadMode);
    document.getElementById('agentKbBar').style.display = agentKbUploadMode ? 'flex' : 'none';
}

// 每个模式独立记录当前会话ID，切换模式时恢复
let modeChatId = { agent: null, chat: null };
// Per-agent active chat tracking for conversation isolation
let agentActiveChatId = { 'xf-rd-agent': null, 'xf-quality-agent': null };

function saveAgentActiveChatIds() {
    localStorage.setItem('agentActiveChatIds', JSON.stringify(agentActiveChatId));
}

function loadAgentActiveChatIds() {
    try {
        const saved = localStorage.getItem('agentActiveChatIds');
        if (saved) agentActiveChatId = JSON.parse(saved);
    } catch(e) {}
}

// Load per-agent active chat IDs at startup
loadAgentActiveChatIds();

// ===== API Helper (with JWT Token) =====
function apiHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) {
        headers['Authorization'] = 'Bearer ' + authToken;
    }
    return headers;
}

// ===== Theme =====
function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? 'light' : 'dark');
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
    document.getElementById('themeBtn').textContent = isDark ? '🌙' : '☀️';
}

(function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
})();

// ===== Web Search Toggle =====
function toggleWebSearch() {
    webSearchEnabled = !webSearchEnabled;
    const btn = document.getElementById('webSearchToggle');
    btn.classList.toggle('active', webSearchEnabled);
    localStorage.setItem('webSearch', webSearchEnabled ? '1' : '0');
}

(function initWebSearch() {
    const saved = localStorage.getItem('webSearch');
    if (saved === '1') {
        webSearchEnabled = true;
        document.getElementById('webSearchToggle').classList.add('active');
    }
})();

// ===== Mode Switch =====
function switchMode(mode) {
    if (currentMode === mode) return;

    // Before switching away from agent mode, save the current agent's active chat
    if (currentMode === 'agent' && currentAgentId) {
        agentActiveChatId[currentAgentId] = currentChatId;
        saveAgentActiveChatIds();
    }

    // 保存当前模式的 chatId
    modeChatId[currentMode] = currentChatId;

    currentMode = mode;
    localStorage.setItem('chatMode', mode);

    document.getElementById('modeChat').classList.toggle('active', mode === 'chat');
    document.getElementById('modeAgent').classList.toggle('active', mode === 'agent');

    const webToggle = document.getElementById('webSearchToggle');
    const thinkToggle = document.getElementById('deepThinkToggle');

    if (mode === 'chat') {
        webToggle.style.display = '';
        thinkToggle.classList.add('visible');
    } else {
        webToggle.style.display = '';
        thinkToggle.classList.remove('visible');
        thinkToggle.classList.remove('active');
        deepThinkEnabled = false;
    }

    const titleEl = document.getElementById('chatTitle');
    if (titleEl) {
        if (mode === 'agent' && currentAgentId) {
            const agent = myAgents.find(a => a.id === currentAgentId);
            titleEl.textContent = agent ? agent.name : 'XF模具智能体平台';
        } else {
            titleEl.textContent = mode === 'agent' ? 'XF模具智能体平台' : 'Chat';
        }
    }
    // Reset agent when switching to chat mode
    if (mode === 'chat') {
        currentAgentId = null;
        renderMyAgents();
    }

    // After switching to agent mode, restore from agentActiveChatId
    if (mode === 'agent' && currentAgentId) {
        const lastChat = agentActiveChatId[currentAgentId];
        if (lastChat) {
            modeChatId['agent'] = lastChat;
        }
    }

    // 更新知识库上传按钮可见性
    updateKbUploadVisibility();
    updateHeaderKbVisibility();

    const welcomeH2 = document.querySelector('.welcome-center h2');
    const welcomeP = document.querySelector('.welcome-center p');
    if (welcomeH2) welcomeH2.textContent = mode === 'agent' ? 'XF模具智能体平台' : 'Chat';
    if (welcomeP) {
        welcomeP.textContent = mode === 'agent'
            ? '智能体锻造引擎，让每个想法都能锻造出专属Agent。'
            : '通用对话模式，深度思考更精准，随时为您解答各类问题。';
    }

    // 切换模式时：筛选该模式的历史对话，恢复该模式上次的会话
    renderChatList();
    restoreModeChat();
}

// 恢复当前模式上次的活跃会话，如果没有则新建
async function restoreModeChat() {
    const modeChats = getModeChats();
    const savedId = modeChatId[currentMode];
    if (modeChats.length === 0) {
        // 该模式没有会话，新建一个
        await createNewChat();
    } else if (savedId && modeChats.some(c => c.chat_id === savedId)) {
        // 恢复上次该模式的会话
        currentChatId = savedId;
        renderChatList();
        await loadChatHistory(savedId);
    } else {
        // 选择该模式的第一个会话
        currentChatId = modeChats[0].chat_id;
        modeChatId[currentMode] = currentChatId;
        renderChatList();
        await loadChatHistory(currentChatId);
    }
}

// 判断对话是否属于某个智能体（同时参考本地 chat_ids 和服务端 agent_id）
function chatBelongsToAgent(chat, agentId) {
    // 1. 检查本地 localStorage 的 chat_ids
    const agent = myAgents.find(a => a.id === agentId);
    if (agent && agent.chat_ids && agent.chat_ids.includes(chat.chat_id)) {
        return true;
    }
    // 2. 检查服务端返回的 agent_id 字段（跨浏览器同步的关键）
    if (chat.agent_id && chat.agent_id === agentId) {
        return true;
    }
    return false;
}

// 判断对话是否属于任意智能体
function chatBelongsToAnyAgent(chat) {
    return myAgents.some(agent => chatBelongsToAgent(chat, agent.id));
}

// 获取当前模式的会话列表
function getModeChats() {
    // Chat mode: show chats with mode='chat'
    if (currentMode === 'chat') {
        return allChats.filter(chat => chat.mode === 'chat');
    }
    // Agent mode with specific agent: show that agent's chats
    if (currentMode === 'agent' && currentAgentId) {
        return allChats.filter(chat => chatBelongsToAgent(chat, currentAgentId));
    }
    // Agent mode but no specific agent: show agent-mode chats not belonging to any agent
    if (currentMode === 'agent' && !currentAgentId) {
        return allChats.filter(chat => {
            const modeMatch = chat.mode === 'agent' || (!chat.mode && currentMode === 'agent');
            if (!modeMatch) return false;
            return !chatBelongsToAnyAgent(chat);
        });
    }
    return [];
}

(function initMode() {
    const saved = localStorage.getItem('chatMode');
    if (saved === 'chat') {
        currentMode = 'chat';
        localStorage.setItem('chatMode', 'chat');
        document.getElementById('modeChat').classList.add('active');
        document.getElementById('modeAgent').classList.remove('active');
    }
    // 初始化时根据状态决定知识库按钮可见性
    updateKbUploadVisibility();
    updateHeaderKbVisibility();
})();

// ===== Deep Think Toggle =====
function toggleDeepThink() {
    deepThinkEnabled = !deepThinkEnabled;
    const btn = document.getElementById('deepThinkToggle');
    btn.classList.toggle('active', deepThinkEnabled);
    localStorage.setItem('deepThink', deepThinkEnabled ? '1' : '0');
}

(function initDeepThink() {
    const saved = localStorage.getItem('deepThink');
    if (saved === '1' && currentMode === 'chat') {
        deepThinkEnabled = true;
        document.getElementById('deepThinkToggle').classList.add('active');
    }
})();

// ===== Marked Config =====
if (typeof marked !== 'undefined') {
    marked.setOptions({
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                try { return hljs.highlight(code, { language: lang }).value; } catch (e) {}
            }
            if (typeof hljs !== 'undefined') {
                try { return hljs.highlightAuto(code).value; } catch (e) {}
            }
            return code;
        },
        breaks: true,
        gfm: true,
    });

    const renderer = new marked.Renderer();
    renderer.code = function(code, language, escaped) {
        let codeText = '', lang = '';
        if (typeof code === 'object') {
            codeText = code.text || '';
            lang = code.lang || '';
        } else {
            codeText = code;
            lang = language || '';
        }
        let highlighted;
        if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
            try { highlighted = hljs.highlight(codeText, { language: lang }).value; } catch (e) { highlighted = escapeHtml(codeText); }
        } else if (typeof hljs !== 'undefined') {
            try { highlighted = hljs.highlightAuto(codeText).value; } catch (e) { highlighted = escapeHtml(codeText); }
        } else {
            highlighted = escapeHtml(codeText);
        }
        const langLabel = lang ? lang : 'code';
        const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
        return `<pre><div class="code-block-header"><span>${langLabel}</span><button class="code-copy-btn" onclick="copyCodeBlock('${codeId}', this)" aria-label="复制代码">复制</button></div><code id="${codeId}" class="hljs language-${lang}">${highlighted}</code></pre>`;
    };
    marked.setOptions({ renderer: renderer });
}

// ===== Toast =====
function showToast(msg, duration) {
    duration = duration || 2000;
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

// ===== Clipboard =====
function copyToClipboard(text, onSuccess, onFail) {
    // 优先尝试 Clipboard API（需要 HTTPS 或 localhost）
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            if (onSuccess) onSuccess();
        }).catch(() => {
            if (!fallbackCopy(text)) { if (onFail) onFail(); } else { if (onSuccess) onSuccess(); }
        });
        return;
    }
    // HTTP 环境：使用 fallback
    if (!fallbackCopy(text)) { if (onFail) onFail(); } else { if (onSuccess) onSuccess(); }
}

function fallbackCopy(text) {
    try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '0';
        ta.style.top = '0';
        ta.style.opacity = '0';
        ta.style.pointerEvents = 'none';
        ta.setAttribute('readonly', '');
        ta.style.fontSize = '16px'; // 防止 iOS 缩放
        document.body.appendChild(ta);
        ta.focus();
        ta.setSelectionRange(0, ta.value.length);
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
    } catch (e) { return false; }
}

// ===== Code Block Copy =====
function copyCodeBlock(codeId, btn) {
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;
    const text = codeEl.textContent;
    copyToClipboard(text, () => {
        btn.textContent = '已复制';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('copied'); }, 2000);
        showToast('代码已复制');
    }, () => { showToast('复制失败'); });
}

// ===== Model Management =====
async function loadModels() {
    try {
        const resp = await fetch('/api/v1/models');
        const data = await resp.json();
        const select = document.getElementById('modelSelect');
        select.innerHTML = '';
        data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id; opt.textContent = m.name; opt.title = m.desc;
            if (m.id === data.current) opt.selected = true;
            select.appendChild(opt);
        });
    } catch (e) { console.error('加载模型列表失败', e); }
}

async function switchModel() {
    const modelId = document.getElementById('modelSelect').value;
    try {
        const resp = await fetch('/api/v1/models/set', { method: 'POST', headers: apiHeaders(), body: JSON.stringify({ model_id: modelId }) });
        const data = await resp.json();
        if (data.success) {
            const select = document.getElementById('modelSelect');
            const name = select.options[select.selectedIndex].textContent;
            addMessageToUI('assistant', `✅ 已切换到模型: ${name}`);
        }
    } catch (e) { console.error('切换模型失败', e); }
}

// ===== Auth =====
// ===== Login Modal =====
function openLoginModal() {
    document.getElementById('loginModalTitle').textContent = '登录企业工作台';
    document.getElementById('loginModalSubtitle').textContent = '登录您的账号以继续';
    switchTab('login');
    document.getElementById('loginModal').classList.add('show');
    setTimeout(() => document.getElementById('loginUser').focus(), 100);
}

function closeLoginModal() {
    document.getElementById('loginModal').classList.remove('show');
    const loginMsg = document.getElementById('loginMsg');
    if (loginMsg) { loginMsg.textContent = ''; loginMsg.className = 'msg-box'; }
    const regMsg = document.getElementById('regMsg');
    if (regMsg) { regMsg.textContent = ''; regMsg.className = 'msg-box'; }
}

function openTrialModal() {
    document.getElementById('loginModalTitle').textContent = '开始免费试用';
    document.getElementById('loginModalSubtitle').textContent = '创建账号即可免费锻造你的第一个Agent';
    switchTab('register');
    document.getElementById('loginModal').classList.add('show');
    setTimeout(() => document.getElementById('regUser').focus(), 100);
}

function switchTab(tab) {
    document.querySelectorAll('.login-modal-body .tab').forEach(t => t.classList.remove('active'));
    const indicator = document.querySelector('.tab-indicator');
    if (tab === 'login') {
        document.querySelectorAll('.login-modal-body .tab')[0].classList.add('active');
        document.getElementById('loginForm').style.display = 'block';
        document.getElementById('registerForm').style.display = 'none';
        if (indicator) { indicator.style.left = '0'; indicator.style.width = '50%'; }
    } else {
        document.querySelectorAll('.login-modal-body .tab')[1].classList.add('active');
        document.getElementById('loginForm').style.display = 'none';
        document.getElementById('registerForm').style.display = 'block';
        if (indicator) { indicator.style.left = '50%'; indicator.style.width = '50%'; }
    }
}

// Close modal on overlay click (works with both old card layout and new fullscreen split layout)
document.addEventListener('click', function(e) {
    const modal = document.getElementById('loginModal');
    if (!modal || !modal.classList.contains('show')) return;
    // Close only if clicking the overlay background itself (not any child panel)
    if (e.target === modal) closeLoginModal();
});

// Close modals on Escape key — close the topmost active modal only
document.addEventListener('keydown', function(e) {
    if (e.key !== 'Escape') return;
    // Priority: rename > agent edit > docs > login (topmost first)
    const renameOverlay = document.getElementById('renameOverlay');
    if (renameOverlay && renameOverlay.classList.contains('show')) { cancelRename(); return; }
    const agentEdit = document.getElementById('agentEditModal');
    if (agentEdit && agentEdit.classList.contains('show')) { closeAgentEditModal(); return; }
    const docsModal = document.getElementById('docsModal');
    if (docsModal && docsModal.classList.contains('show')) { closeDocs(); return; }
    const loginModal = document.getElementById('loginModal');
    if (loginModal && loginModal.classList.contains('show')) { closeLoginModal(); return; }
});

async function doLogin() {
    const username = document.getElementById('loginUser').value.trim();
    const password = document.getElementById('loginPass').value.trim();
    const msgEl = document.getElementById('loginMsg');
    if (!username || !password) { msgEl.className = 'msg-box error'; msgEl.textContent = '请输入用户名和密码'; return; }
    try {
        const resp = await fetch('/api/v1/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
        const data = await resp.json();
        if (data.success) {
            currentUser = username;
            if (data.token) { authToken = data.token; localStorage.setItem('authToken', data.token); }
            msgEl.className = 'msg-box success'; msgEl.textContent = '登录成功！';
            setTimeout(async () => {
                document.getElementById('loginModal').classList.remove('show');
                document.getElementById('loginPage').classList.add('login-hidden');
                document.getElementById('chatPage').style.display = 'flex';
                document.body.classList.add('body-chat-mode');
                document.getElementById('sidebarUsername').textContent = username;
                document.getElementById('sidebarAvatar').textContent = username[0].toUpperCase();
                loadChatList();
                loadModels();
                await syncAgentsFromServer(true);  // [#12] 登录时强制同步一次，内部已调用 rebuildChatIdsFromServer（会GET /chats）
                renderMyAgents();
                updateKbUploadVisibility();
                updateHeaderKbVisibility();
                // [#14] 默认选中第一个智能体，避免进入空白的agent模式
                if (!currentAgentId && myAgents.length > 0) {
                    await switchToAgent(myAgents[0].id);
                }
            }, 500);
        } else { msgEl.className = 'msg-box error'; msgEl.textContent = data.message || '登录失败'; }
    } catch (e) { msgEl.className = 'msg-box error'; msgEl.textContent = '网络错误'; }
}

async function doRegister() {
    const username = document.getElementById('regUser').value.trim();
    const password = document.getElementById('regPass').value.trim();
    const msgEl = document.getElementById('regMsg');
    if (!username || !password) { msgEl.className = 'msg-box error'; msgEl.textContent = '请输入用户名和密码'; return; }
    try {
        const resp = await fetch('/api/v1/auth/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
        const data = await resp.json();
        if (data.success) {
            msgEl.className = 'msg-box success';
            msgEl.textContent = data.message || '注册成功！';
            // Auto-login after successful registration
            if (data.token) {
                authToken = data.token;
                localStorage.setItem('authToken', data.token);
            }
            currentUser = username;
            setTimeout(async () => {
                document.getElementById('loginModal').classList.remove('show');
                document.getElementById('loginPage').classList.add('login-hidden');
                document.getElementById('chatPage').style.display = 'flex';
                document.body.classList.add('body-chat-mode');
                document.getElementById('sidebarUsername').textContent = username;
                document.getElementById('sidebarAvatar').textContent = username[0].toUpperCase();
                loadChatList();
                loadModels();
                await syncAgentsFromServer(true);  // [#12] 注册后强制同步
                renderMyAgents();
                updateKbUploadVisibility();
                updateHeaderKbVisibility();
            }, 500);
        } else {
            msgEl.className = 'msg-box error';
            msgEl.textContent = data.message || '注册失败';
        }
    } catch (e) { msgEl.className = 'msg-box error'; msgEl.textContent = '网络错误'; }
}

function doLogout() {
    currentUser = null; authToken = null; selectedFile = null; currentChatId = null; allChats = []; currentAgentId = null; agentKbUploadMode = false;
    localStorage.removeItem('authToken');
    document.getElementById('chatPage').style.display = 'none';
    document.getElementById('loginPage').classList.remove('login-hidden');
    document.body.classList.remove('body-chat-mode');
    document.getElementById('chatMessages').innerHTML = '';
    document.getElementById('loginUser').value = '';
    document.getElementById('loginPass').value = '';
    updateHeaderKbVisibility();
}

// ===== Auto-login with JWT token =====
async function tryAutoLogin() {
    const token = localStorage.getItem('authToken');
    if (!token) return false;
    try {
        const resp = await fetch('/api/v1/auth/me', { headers: { 'Authorization': 'Bearer ' + token } });
        const data = await resp.json();
        if (data.valid && data.username) {
            currentUser = data.username;
            authToken = token;
            document.getElementById('loginPage').classList.add('login-hidden');
            document.getElementById('chatPage').style.display = 'flex';
            document.body.classList.add('body-chat-mode');
            document.getElementById('sidebarUsername').textContent = data.username;
            document.getElementById('sidebarAvatar').textContent = data.username[0].toUpperCase();
            loadChatList();
            loadModels();
            await syncAgentsFromServer(true);  // [#12] 自动登录时强制同步
            renderMyAgents();
            updateKbUploadVisibility();
            updateHeaderKbVisibility();
            // [#14] 默认选中第一个智能体，避免进入空白的agent模式
            if (!currentAgentId && myAgents.length > 0) {
                await switchToAgent(myAgents[0].id);
            }
            return true;
        }
    } catch (e) { console.warn('自动登录失败', e); }
    localStorage.removeItem('authToken');
    return false;
}

// ===== Centered Mode =====
function updateCenteredMode() {
    const content = document.getElementById('chatContent');
    const messages = document.getElementById('chatMessages');
    const hasMessages = messages.children.length > 0;
    content.classList.toggle('centered', !hasMessages);
}

// ===== Chat List =====
async function loadChatList() {
    if (!currentUser) return;
    try {
        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}`, { headers: apiHeaders() });
        const data = await resp.json();
        if (data.success) {
            allChats = data.chats;
            renderChatList();
            // 按当前模式恢复会话
            const modeChats = getModeChats();
            // 如果当前聊天仍然存在于全部聊天列表中，不要强制跳走
            // （避免智能体对话回复完成后，因过滤不同步导致跳转到空页面）
            const currentChatStillExists = currentChatId && allChats.some(c => c.chat_id === currentChatId);
            if (modeChats.length === 0 && !currentChatStillExists) {
                await createNewChat();
            } else if (!currentChatId || (!currentChatStillExists && !modeChats.some(c => c.chat_id === currentChatId))) {
                currentChatId = modeChats[0].chat_id;
                modeChatId[currentMode] = currentChatId;
                renderChatList();
                await loadChatHistory(currentChatId);
            }
        }
    } catch (e) { console.error('加载会话列表失败', e); }
}

function renderChatList() {
    const list = document.getElementById('chatList');
    list.innerHTML = '';
    // 只显示当前模式的会话
    const modeChats = getModeChats();
    modeChats.forEach(chat => {
        const item = document.createElement('div');
        item.className = `chat-item${chat.chat_id === currentChatId ? ' active' : ''}`;
        item.onclick = (e) => {
            if (e.target.closest('.chat-action-btn')) return;
            switchChat(chat.chat_id);
            closeSidebarOnMobile();
        };
        const timeStr = formatTime(chat.updated_at || chat.created_at);
        item.innerHTML = `
            <span class="chat-icon">💬</span>
            <span class="chat-title" title="${escapeHtml(chat.title)}">${escapeHtml(chat.title)}</span>
            <span class="chat-time">${timeStr}</span>
            <div class="chat-actions">
                <button class="chat-action-btn" onclick="openRename('${chat.chat_id}', '${escapeHtml(chat.title)}')" title="重命名" aria-label="重命名对话">✏️</button>
                <button class="chat-action-btn delete" onclick="deleteChatItem('${chat.chat_id}')" title="删除" aria-label="删除对话">🗑️</button>
            </div>
        `;
        list.appendChild(item);
    });
}

async function createNewChat() {
    if (!currentUser) return;
    try {
        const chatTitle = currentAgentId ? (myAgents.find(a => a.id === currentAgentId)?.name || '新对话') : '新对话';
        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}&title=${encodeURIComponent(chatTitle)}&mode=${currentMode}&agent_id=${currentAgentId || ''}`, { method: 'POST', headers: apiHeaders() });
        const data = await resp.json();
        if (data.success) {
            currentChatId = data.chat.chat_id;
            modeChatId[currentMode] = currentChatId;
            // Associate chat with current agent
            if (currentAgentId) {
                const agent = myAgents.find(a => a.id === currentAgentId);
                if (agent) {
                    if (!agent.chat_ids) agent.chat_ids = [];
                    agent.chat_ids.push(data.chat.chat_id);
                    agentActiveChatId[currentAgentId] = data.chat.chat_id;
                    saveAgentActiveChatIds();
                    saveAgents();
                }
            }
            await loadChatList();
            clearChatUI();
            closeSidebarOnMobile();
        }
    } catch (e) { console.error('创建会话失败', e); }
}

async function switchChat(chatId) {
    if (chatId === currentChatId) return;
    currentChatId = chatId;
    modeChatId[currentMode] = chatId;

    // Determine which agent owns this chat (check both local chat_ids and server agent_id)
    let belongsToAgent = null;
    const chatData = allChats.find(c => c.chat_id === chatId);
    myAgents.forEach(agent => {
        if (chatBelongsToAgent(chatData || { chat_id: chatId }, agent.id)) {
            belongsToAgent = agent.id;
        }
    });
    if (belongsToAgent) {
        currentAgentId = belongsToAgent;
        agentActiveChatId[currentAgentId] = chatId;
        saveAgentActiveChatIds();
    }

    renderChatList();
    updateHeaderKbVisibility();
    await loadChatHistory(chatId);
}

async function loadChatHistory(chatId) {
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';
    try {
        const resp = await fetch(`/api/v1/history/${chatId}`, { headers: apiHeaders() });
        const data = await resp.json();
        const messages = data.messages || [];
        if (messages.length > 0) {
            messages.forEach(m => addMessageToUI(m.role, m.content));
            scrollToBottom();
        }
        updateCenteredMode();
    } catch (e) { console.error('加载历史失败', e); }
}

async function deleteChatItem(chatId) {
    if (!confirm('确定删除这个对话？')) return;
    try {
        await fetch(`/api/v1/chats/${chatId}?username=${encodeURIComponent(currentUser)}`, { method: 'DELETE', headers: apiHeaders() });

        // Remove chat_id from all agents
        myAgents.forEach(agent => {
            if (agent.chat_ids) {
                agent.chat_ids = agent.chat_ids.filter(id => id !== chatId);
            }
            // Also clean agentActiveChatId
            if (agentActiveChatId[agent.id] === chatId) {
                agentActiveChatId[agent.id] = agent.chat_ids && agent.chat_ids.length > 0 ? agent.chat_ids[0] : null;
            }
        });
        saveAgentActiveChatIds();
        saveAgents();

        if (chatId === currentChatId) {
            currentChatId = null;
            modeChatId[currentMode] = null;
            clearChatUI();
        }
        await loadChatList();
        // 如果当前模式没有会话了，新建一个
        const modeChats = getModeChats();
        if (modeChats.length === 0) {
            await createNewChat();
        }
    } catch (e) { console.error('删除会话失败', e); }
}

function openRename(chatId, currentTitle) {
    renamingChatId = chatId;
    document.getElementById('renameInput').value = currentTitle;
    document.getElementById('renameOverlay').classList.add('show');
    setTimeout(() => document.getElementById('renameInput').focus(), 100);
}

function closeRename() {
    document.getElementById('renameOverlay').classList.remove('show');
    renamingChatId = null;
}

async function confirmRename() {
    const newTitle = document.getElementById('renameInput').value.trim();
    if (!newTitle || !renamingChatId) return;
    const username = currentUser || '';
    try {
        await fetch(`/api/v1/chats/${renamingChatId}/rename`, {
            method: 'PUT',
            headers: apiHeaders(),
            body: JSON.stringify({ username, chat_id: renamingChatId, new_title: newTitle })
        });
        document.getElementById('renameOverlay').classList.remove('show');
        await loadChatList();
    } catch (e) { showToast('重命名失败'); }
    renamingChatId = null;
}

function cancelRename() {
    document.getElementById('renameOverlay').classList.remove('show');
    renamingChatId = null;
}

function clearChatUI() {
    document.getElementById('chatMessages').innerHTML = '';
    updateCenteredMode();
}

async function clearCurrentChat() {
    if (!currentChatId) return;
    if (!confirm('确定清除当前对话的所有消息？')) return;
    try {
        await fetch(`/api/v1/history/${currentChatId}`, { method: 'DELETE', headers: apiHeaders() });
        clearChatUI();
    } catch (e) {}
}

// ===== Sidebar =====
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('mobile-open');
        overlay.classList.toggle('active');
    } else {
        sidebar.classList.toggle('collapsed');
    }
}
function closeSidebarMobile() {
    document.getElementById('sidebar').classList.remove('mobile-open');
    document.getElementById('sidebarOverlay').classList.remove('active');
}
function closeSidebarOnMobile() {
    if (window.innerWidth <= 768) setTimeout(closeSidebarMobile, 200);
}

// ===== Scroll =====
function setupScrollDetection() {
    const el = document.getElementById('chatMessages');
    el.addEventListener('scroll', () => {
        const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        userScrolledUp = distFromBottom > 100;
        const btn = document.getElementById('scrollBottomBtn');
        btn.classList.toggle('show', userScrolledUp);
    });
}

function scrollToBottom() {
    const el = document.getElementById('chatMessages');
    setTimeout(() => {
        el.scrollTop = el.scrollHeight;
        userScrolledUp = false;
        document.getElementById('scrollBottomBtn').classList.remove('show');
    }, 50);
}

function smartScrollToBottom() {
    if (!userScrolledUp) scrollToBottom();
}

// ===== Stop Generation =====
function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    isLoading = false;
    document.getElementById('sendBtn').style.display = '';
    document.getElementById('stopBtn').style.display = 'none';
    document.getElementById('sendBtn').disabled = false;
}

// ===== Thinking Status Texts =====
const THINKING_TEXTS = [
    '正在思考...',
    '分析问题中...',
    '整理思路...',
    '查找信息中...',
    '生成回答中...',
];
let thinkingTextIndex = 0;
let thinkingInterval = null;

// ===== Streaming Chat =====
function createStreamingBubble() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.innerHTML = `
        <button class="msg-action-btn" title="复制" onclick="copyMessage(this)" aria-label="复制消息">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
        </button>
        <button class="msg-action-btn" title="重新生成" onclick="regenerateMessage(this)" aria-label="重新生成">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
        </button>
    `;
    div.appendChild(bubble);
    div.appendChild(actions);
    container.appendChild(div);
    return bubble;
}

async function streamChat(url, options, bubble) {
    let fullText = '';
    let cursorEl = null;
    let thinkingEl = null;

    currentAbortController = new AbortController();
    if (options && !options.signal) {
        options.signal = currentAbortController.signal;
    }

    // Show stop button
    document.getElementById('sendBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = '';

    function addThinking() {
        if (thinkingEl) return;
        thinkingEl = document.createElement('div');
        thinkingEl.className = 'thinking-indicator';
        thinkingTextIndex = 0;
        thinkingEl.innerHTML = `<div class="spinner"></div><span class="think-status">${THINKING_TEXTS[0]}</span>`;
        bubble.appendChild(thinkingEl);
        smartScrollToBottom();
        // Rotate thinking text
        thinkingInterval = setInterval(() => {
            thinkingTextIndex = (thinkingTextIndex + 1) % THINKING_TEXTS.length;
            const statusEl = thinkingEl?.querySelector('.think-status');
            if (statusEl) statusEl.textContent = THINKING_TEXTS[thinkingTextIndex];
        }, 2000);
    }

    function removeThinking() {
        if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
        if (thinkingInterval) { clearInterval(thinkingInterval); thinkingInterval = null; }
    }

    function addToolTag(display, isDone) {
        removeThinking();
        const tag = document.createElement('span');
        if (isDone) {
            tag.className = 'tool-tag done';
            tag.innerHTML = `<span class="tool-icon">✓</span> ${escapeHtml(display)}`;
        } else {
            tag.className = 'tool-tag running';
            tag.innerHTML = `<span class="tool-spinner"></span> ${escapeHtml(display)}`;
        }
        bubble.appendChild(tag);
        bubble.appendChild(document.createTextNode(' '));
        smartScrollToBottom();
    }

    function addCursor() {
        if (cursorEl) return;
        removeThinking();
        cursorEl = document.createElement('span');
        cursorEl.className = 'stream-cursor';
        cursorEl.textContent = '▊';
        bubble.appendChild(cursorEl);
        smartScrollToBottom();
    }

    function appendToken(text) {
        removeThinking();
        if (cursorEl) {
            cursorEl.before(document.createTextNode(text));
        } else {
            bubble.appendChild(document.createTextNode(text));
        }
        smartScrollToBottom();
    }

    function finalize() {
        if (cursorEl) cursorEl.remove();
        cursorEl = null;
    }

    try {
        const resp = await fetch(url, options);

        if (!resp.ok) {
            removeThinking();
            const errData = await resp.json().catch(() => ({}));
            if (resp.status === 401) {
                showToast('登录已过期，请重新登录');
                doLogout();
                return;
            }
            bubble.innerHTML = escapeHtml(errData.detail || `请求失败 (${resp.status})`);
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;

                try {
                    const data = JSON.parse(jsonStr);
                    switch (data.type) {
                        case 'thinking': addThinking(); break;
                        case 'tool': addToolTag(data.display || data.name, false); break;
                        case 'tool_done': addToolTag(data.display || data.name, true); break;
                        case 'token': addCursor(); appendToken(data.content); fullText += data.content; break;
                        case 'done': finalize(); break;
                        case 'error': removeThinking(); finalize(); bubble.innerHTML += `<br><span style="color:var(--error)">${escapeHtml(data.content)}</span>`; break;
                    }
                } catch (e) { console.warn('SSE parse error:', e, jsonStr); }
            }
        }

        finalize();
        removeThinking();

        if (!fullText) {
            if (bubble.textContent.trim() === '') {
                bubble.innerHTML = '（未获取到回复）';
            }
        } else {
            renderBubbleMarkdown(bubble, fullText);
        }

    } catch (e) {
        removeThinking();
        finalize();
        if (e.name === 'AbortError') {
            if (fullText) {
                renderBubbleMarkdown(bubble, fullText);
                bubble.innerHTML += '<br><span style="color:var(--text-secondary);font-size:13px;">（已停止生成）</span>';
            } else {
                bubble.innerHTML = '<span style="color:var(--text-secondary)">已停止生成</span>';
            }
        } else {
            bubble.innerHTML = `<span style="color:var(--error)">网络错误，请重试</span>`;
        }
    } finally {
        currentAbortController = null;
        document.getElementById('sendBtn').style.display = '';
        document.getElementById('stopBtn').style.display = 'none';
    }
}

// ===== Markdown Rendering =====
function renderBubbleMarkdown(bubble, text) {
    if (typeof marked !== 'undefined' && text) {
        try {
            // 先用 marked 渲染 Markdown
            bubble.innerHTML = marked.parse(text);
            // 渲染后再替换下载链接为可点击按钮（避免 marked 过滤 HTML 标签）
            injectDownloadButtons(bubble);
            return;
        } catch (e) { console.warn('Markdown渲染失败', e); }
    }
    bubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
}

function injectDownloadButtons(container) {
    // 遍历所有文本节点，查找下载链接并替换为按钮
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    const nodesToReplace = [];
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.nodeValue && /\/api\/v1\/documents\/export-download\/[^ \n\)<"]+\.(docx|xlsx|pdf|txt)/.test(node.nodeValue)) {
            nodesToReplace.push(node);
        }
    }
    nodesToReplace.forEach(node => {
        const text = node.nodeValue;
        const urlMatch = text.match(/(\/api\/v1\/documents\/export-download\/[^ \n\)<"]+\.(docx|xlsx|pdf|txt))/);
        if (urlMatch) {
            const url = urlMatch[1];
            const btn = document.createElement('a');
            // 根据文件扩展名显示不同的按钮文字和样式
            const ext = url.split('.').pop().toLowerCase();
            const btnLabels = { docx: '点击下载Word文档', xlsx: '点击下载Excel表格', pdf: '点击下载PDF文档', txt: '点击下载文本文件' };
            btn.className = 'doc-download-btn' + (ext === 'xlsx' ? ' xlsx-btn' : '');
            btn.href = 'javascript:void(0)';
            btn.textContent = btnLabels[ext] || '点击下载文档';
            btn.onclick = function() { downloadExportFile(url); };
            // 替换整个文本节点为按钮
            const parent = node.parentNode;
            // 保留链接前面的文字
            const beforeText = text.substring(0, text.indexOf(url)).replace(/下载链接[：:]*\s*$/, '');
            if (beforeText.trim()) {
                parent.insertBefore(document.createTextNode(beforeText), node);
            }
            parent.insertBefore(btn, node);
            // 保留链接后面的文字
            const afterText = text.substring(text.indexOf(url) + url.length);
            if (afterText.trim()) {
                parent.insertBefore(document.createTextNode(afterText), node);
            }
            parent.removeChild(node);
        }
    });
}

// ===== 导出文件下载（支持中文文件名） =====
async function downloadExportFile(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            alert('下载失败：' + response.status + ' ' + response.statusText);
            return;
        }
        // 从Content-Disposition提取文件名
        const disposition = response.headers.get('Content-Disposition');
        // 根据URL中的扩展名决定默认文件名
        const urlExt = url.split('.').pop().toLowerCase();
        const defaultNames = { docx: '导出文档.docx', xlsx: '导出表格.xlsx', pdf: '导出文档.pdf', txt: '导出文本.txt' };
        let filename = defaultNames[urlExt] || '导出文档.docx';
        if (disposition) {
            const utf8Match = disposition.match(/filename\*=UTF-8''(.+)/i);
            if (utf8Match) {
                filename = decodeURIComponent(utf8Match[1]);
            } else {
                const plainMatch = disposition.match(/filename="?([^"]+)"?/);
                if (plainMatch) filename = plainMatch[1];
            }
        }
        // 从URL提取文件名（兜底：默认文件名未被服务端覆盖时才使用URL中的文件名）
        if (filename === defaultNames[urlExt] || filename === '导出文档.docx') {
            const urlParts = url.split('/');
            const lastPart = urlParts[urlParts.length - 1];
            if (lastPart) filename = decodeURIComponent(lastPart);
        }
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(blobUrl);
    } catch (e) {
        console.error('下载导出文件失败:', e);
        // 降级：直接在新标签页打开
        window.open(url, '_blank');
    }
}

// ===== Send Message =====
async function sendMessage() {
    if (isLoading) return;
    if (!currentChatId) { alert('请先创建或选择一个对话'); return; }
    const input = document.getElementById('msgInput');
    const message = input.value.trim();
    if (!message && !selectedFile) return;
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;

    document.getElementById('chatContent').classList.remove('centered');

    if (selectedFile && message) {
        const isImage = selectedFile.type.startsWith('image/');
        const icon = isImage ? '🖼️' : '📎';
        if (isImage && selectedFileBase64) {
            addMessageToUI('user', `${icon} ${selectedFile.name}\n${message}`, selectedFileBase64);
        } else {
            addMessageToUI('user', `${icon} ${selectedFile.name}\n${message}`);
        }
        input.value = ''; autoResize(input);
        const bubble = createStreamingBubble();
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('message', message);
        formData.append('session_id', currentChatId);
        formData.append('web_search', webSearchEnabled);
        formData.append('mode', currentMode);
        formData.append('deep_think', deepThinkEnabled);
        // 智能体ID和任务描述
        if (currentAgentId) {
            formData.append('agent_id', currentAgentId);
            const curAgent = myAgents.find(a => a.id === currentAgentId);
            if (curAgent) formData.append('agent_task', curAgent.task);
        } else {
            formData.append('agent_id', '');
        }
        // 知识库模式：📚激活时文件存入知识库，未激活时仅用于回答
        if (agentKbUploadMode && currentAgentId) {
            formData.append('store_to_kb', 'true');
        } else {
            formData.append('store_to_kb', 'false');
        }
        await streamChat('/api/v1/chat-with-file/stream', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} }, bubble);
        removeFile();
        await loadChatList();
    } else if (selectedFile && !message) {
        addMessageToUI('user', `[上传文档] ${selectedFile.name}`);
        showTyping(true);
        const formData = new FormData();
        formData.append('file', selectedFile);
        // Always pass agent_id when an agent is selected, not just when KB mode is on
        formData.append('agent_id', currentAgentId || '');
        // 知识库模式：📚激活时文件存入知识库
        if (agentKbUploadMode && currentAgentId) {
            formData.append('store_to_kb', 'true');
        }
        try {
            const resp = await fetch('/api/v1/upload', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} });
            const data = await resp.json();
            showTyping(false);
            const msg = data.detail?.message || (typeof data.detail === 'string' ? data.detail : '上传成功');
            addMessageToUI('assistant', msg);
        } catch (e) { showTyping(false); addMessageToUI('assistant', '上传失败，请重试'); }
        removeFile();
    } else {
        lastMessageText = message;
        addMessageToUI('user', message);
        input.value = ''; autoResize(input);
        const bubble = createStreamingBubble();
        await streamChat('/api/v1/chat/stream', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify({ message, session_id: currentChatId, web_search: webSearchEnabled, mode: currentMode, deep_think: deepThinkEnabled, agent_id: currentAgentId || '', agent_task: (currentAgentId && myAgents.find(a => a.id === currentAgentId)) ? myAgents.find(a => a.id === currentAgentId).task : '' })
        }, bubble);
        await loadChatList();
    }
    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
    scrollToBottom();
}

function sendQuick(text) { document.getElementById('msgInput').value = text; sendMessage(); }

function addMessageToUI(role, content, imageBase64) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (role === 'assistant') {
        renderBubbleMarkdown(bubble, content);
    } else {
        let htmlContent = escapeHtml(content).replace(/\n/g, '<br>');
        if (imageBase64) htmlContent += `<img class="chat-img" src="${imageBase64}" alt="上传的图片">`;
        bubble.innerHTML = htmlContent;
        bubble.style.whiteSpace = 'pre-wrap';
    }

    const actions = document.createElement('div');
    actions.className = 'message-actions';
    if (role === 'assistant') {
        actions.innerHTML = `
            <button class="msg-action-btn" title="复制" onclick="copyMessage(this)" aria-label="复制消息">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            </button>
            <button class="msg-action-btn" title="重新生成" onclick="regenerateMessage(this)" aria-label="重新生成">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
            </button>
        `;
    } else {
        actions.innerHTML = `
            <button class="msg-action-btn" title="复制" onclick="copyMessage(this)" aria-label="复制消息">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            </button>
        `;
    }

    div.appendChild(bubble);
    div.appendChild(actions);
    container.appendChild(div);

    document.getElementById('chatContent').classList.remove('centered');
    scrollToBottom();
}

// ===== Message Actions =====
function copyMessage(btn) {
    const messageDiv = btn.closest('.message');
    const bubble = messageDiv ? messageDiv.querySelector('.bubble') : null;
    if (!bubble) { showToast('复制失败：未找到消息内容'); return; }
    // 获取纯文本，排除代码块复制按钮的文字
    let text = bubble.innerText || bubble.textContent || '';
    // 去除代码块中的"复制"/"已复制"文字
    text = text.replace(/\n?复制\n?/g, '\n').replace(/\n?已复制\n?/g, '\n').trim();
    if (!text) { showToast('复制失败：内容为空'); return; }
    copyToClipboard(text, () => { showToast('已复制到剪贴板'); }, () => { showToast('复制失败，请手动复制'); });
}

async function regenerateMessage(btn) {
    if (isLoading) return;
    const messageDiv = btn.closest('.message');
    const prev = messageDiv.previousElementSibling;
    if (!prev || !prev.classList.contains('user')) { showToast('无法找到对应的用户消息'); return; }
    const userBubble = prev.querySelector('.bubble');
    const userText = userBubble.textContent || userBubble.innerText;
    messageDiv.remove();
    if (!currentChatId) return;
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;
    const bubble = createStreamingBubble();
    await streamChat('/api/v1/chat/stream', {
        method: 'POST',
        headers: apiHeaders(),
        body: JSON.stringify({ message: userText, session_id: currentChatId, web_search: webSearchEnabled, mode: currentMode, deep_think: deepThinkEnabled, agent_id: currentAgentId || '', agent_task: (currentAgentId && myAgents.find(a => a.id === currentAgentId)) ? myAgents.find(a => a.id === currentAgentId).task : '' })
    }, bubble);
    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
}

function showTyping(show) { document.getElementById('typingIndicator').style.display = show ? 'block' : 'none'; if (show) scrollToBottom(); }

// ===== File Handling =====
function onFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        if (file.size > MAX_FILE_SIZE) { showToast('文件大小不能超过 50MB'); event.target.value = ''; return; }
        setFilePreview(file);
    }
}

function setFilePreview(file) {
    selectedFile = file;
    selectedFileBase64 = null;
    const isImage = file.type.startsWith('image/');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileIcon').textContent = isImage ? '🖼️' : '📎';
    document.getElementById('fileBar').style.display = 'flex';
    document.getElementById('msgInput').placeholder = '针对此文件输入问题，或修改要求...';
    if (isImage) {
        const reader = new FileReader();
        reader.onload = function(e) { selectedFileBase64 = e.target.result; };
        reader.readAsDataURL(file);
    }
}

function removeFile() {
    selectedFile = null;
    selectedFileBase64 = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileBar').style.display = 'none';
    document.getElementById('fileIcon').textContent = '📎';
    document.getElementById('msgInput').placeholder = '输入问题，或粘贴/拖拽文件...';
}

// ===== Paste & Drag =====
document.addEventListener('DOMContentLoaded', function() {
    const msgInput = document.getElementById('msgInput');
    const inputContainer = document.querySelector('.input-container');

    msgInput.addEventListener('paste', function(e) {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) { if (file.size > MAX_FILE_SIZE) { showToast('图片大小不能超过 50MB'); return; } setFilePreview(file); showToast('已粘贴图片，输入问题后发送'); }
                return;
            }
            if (item.kind === 'file' && !item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) { setFilePreview(file); showToast('已粘贴文件，输入问题后发送'); }
                return;
            }
        }
    });

    inputContainer.addEventListener('dragover', function(e) { e.preventDefault(); e.stopPropagation(); inputContainer.style.borderColor = 'var(--accent)'; inputContainer.style.background = 'rgba(26,26,26,0.03)'; });
    inputContainer.addEventListener('dragleave', function(e) { e.preventDefault(); e.stopPropagation(); inputContainer.style.borderColor = ''; inputContainer.style.background = ''; });
    inputContainer.addEventListener('drop', function(e) { e.preventDefault(); e.stopPropagation(); inputContainer.style.borderColor = ''; inputContainer.style.background = ''; const files = e.dataTransfer.files; if (files.length > 0) { setFilePreview(files[0]); showToast('已添加文件，输入问题后发送'); } });
});

// ===== Knowledge Base Modal =====
async function showDocs() {
    document.getElementById('docsModal').classList.add('show');
    await loadDocList();
}
function closeDocs() { document.getElementById('docsModal').classList.remove('show'); document.getElementById('uploadProgress').style.display = 'none'; }

async function loadDocList() {
    const list = document.getElementById('docList');
    list.innerHTML = '<div class="doc-empty">加载中...</div>';
    try {
        // 按 agent_id 获取对应知识库的文档列表
        const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
        const resp = await fetch(`/api/v1/documents${agentParam}`, { headers: apiHeaders() });
        const data = await resp.json();
        list.innerHTML = '';
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                let icon = '📄';
                if (doc.endsWith('.pdf')) icon = '📕';
                else if (doc.endsWith('.docx')) icon = '📘';
                else if (doc.endsWith('.txt')) icon = '📝';
                item.innerHTML = `<span class="doc-icon">${icon}</span><span class="doc-name">${doc}</span><button class="doc-download-btn" onclick="downloadDocument('${doc.replace(/'/g, "\\'")}')" title="下载" aria-label="下载文档">📥</button><button class="doc-delete-btn" onclick="deleteDocument('${doc.replace(/'/g, "\\'")}', this)">删除</button>`;
                list.appendChild(item);
            });
        } else { list.innerHTML = '<div class="doc-empty">暂无文档，请上传</div>'; }
    } catch (e) { list.innerHTML = '<div class="doc-empty">加载失败</div>'; }
}

async function onKbFileSelected(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    for (let i = 0; i < files.length; i++) { await uploadToKnowledgeBase(files[i]); }
    document.getElementById('kbFileInput').value = '';
    await loadDocList();
}

async function deleteDocument(filename, btnEl) {
    if (!confirm(`确定要删除文档 "${filename}" 吗？此操作不可恢复！`)) return;
    const docItem = btnEl.closest('.doc-item');
    btnEl.disabled = true; btnEl.textContent = '删除中...';
    try {
        const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
        const resp = await fetch(`/api/v1/documents/${encodeURIComponent(filename)}${agentParam}`, { method: 'DELETE', headers: apiHeaders() });
        const data = await resp.json();
        if (resp.ok && data.status === 'success') {
            docItem.style.transition = 'all 0.3s'; docItem.style.opacity = '0'; docItem.style.transform = 'translateX(20px)';
            setTimeout(() => { docItem.remove(); const list = document.getElementById('docList'); if (list.children.length === 0) list.innerHTML = '<div class="doc-empty">暂无文档，请上传</div>'; }, 300);
            // 同步刷新右侧KB面板
            if (currentAgentId) loadKbDocs();
        } else { alert('删除失败：' + (data.detail || '未知错误')); btnEl.disabled = false; btnEl.textContent = '删除'; }
    } catch (e) { alert('删除失败：网络错误'); btnEl.disabled = false; btnEl.textContent = '删除'; }
}

async function uploadToKnowledgeBase(file) {
    const progressEl = document.getElementById('uploadProgress');
    const fileNameEl = document.getElementById('progressFileName');
    const barFill = document.getElementById('progressBarFill');
    const statusEl = document.getElementById('progressStatus');
    progressEl.style.display = 'block';
    const kbLabel = currentAgentId ? `智能体「${myAgents.find(a => a.id === currentAgentId)?.name || ''}」知识库` : '知识库';
    fileNameEl.textContent = `📎 ${file.name} → ${kbLabel}`;
    barFill.style.width = '10%';
    statusEl.textContent = '上传中...';
    statusEl.className = 'progress-status';
    const formData = new FormData();
    formData.append('file', file);
    if (currentAgentId) formData.append('agent_id', currentAgentId);
    try {
        barFill.style.width = '30%';
        const resp = await fetch('/api/v1/upload', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} });
        barFill.style.width = '80%';
        const data = await resp.json();
        if (resp.ok) { barFill.style.width = '100%'; statusEl.textContent = `✅ 上传成功！文档已索引到${kbLabel}`; statusEl.className = 'progress-status success'; }
        else { barFill.style.width = '100%'; barFill.style.background = 'var(--error)'; statusEl.textContent = '❌ 上传失败：' + (data.detail || '未知错误'); statusEl.className = 'progress-status error'; }
    } catch (e) { barFill.style.width = '100%'; barFill.style.background = 'var(--error)'; statusEl.textContent = '❌ 网络错误，请重试'; statusEl.className = 'progress-status error'; }
    setTimeout(() => { progressEl.style.display = 'none'; barFill.style.background = 'var(--accent)'; }, 3000);
}

function downloadDocument(filename) {
    // 在新标签页打开下载链接
    const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
    const url = `/api/v1/documents/${encodeURIComponent(filename)}/download${agentParam}`;
    window.open(url, '_blank');
}

// ===== Utility Functions =====
function formatTime(timestamp) {
    if (!timestamp) return '';
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return Math.floor(diff / 60) + '分钟前';
    if (diff < 86400) return Math.floor(diff / 3600) + '小时前';
    if (diff < 604800) return Math.floor(diff / 86400) + '天前';
    const d = new Date(timestamp * 1000);
    return `${d.getMonth() + 1}/${d.getDate()}`;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function handleKey(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); } }
function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }

// ===== Chat Search =====
function filterChats(query) {
    const items = document.querySelectorAll('.chat-item');
    query = query.toLowerCase().trim();
    items.forEach(item => {
        const title = item.querySelector('.chat-title');
        if (!title) return;
        const text = title.textContent.toLowerCase();
        item.style.display = (!query || text.includes(query)) ? '' : 'none';
    });
}

// ===== Export Chat =====
function toggleExportDropdown() {
    const dropdown = document.getElementById('exportDropdown');
    dropdown.classList.toggle('show');
    // Close when clicking outside
    if (dropdown.classList.contains('show')) {
        setTimeout(() => {
            document.addEventListener('click', closeExportDropdown, { once: true });
        }, 0);
    }
}

function closeExportDropdown(e) {
    const dropdown = document.getElementById('exportDropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        dropdown.classList.remove('show');
    }
}

async function exportChat(format) {
    if (!currentChatId) return;
    const dropdown = document.getElementById('exportDropdown');
    if (dropdown) dropdown.classList.remove('show');

    try {
        const resp = await fetch(`/api/v1/export/${currentChatId}?format=${format}`, { headers: apiHeaders() });
        if (!resp.ok) { showToast('导出失败'); return; }

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const ext = format === 'pdf' ? 'pdf' : 'md';
        a.download = `chat_${currentChatId.slice(0, 12)}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        showToast(`已导出为 ${format.toUpperCase()}`);
    } catch (e) {
        showToast('导出失败');
    }
}

// ===== Knowledge Base Panel =====
function toggleKbPanel() {
    const panel = document.getElementById('kbPanel');
    if (!panel) return;
    const wasShown = panel.classList.contains('show');
    panel.classList.toggle('show');
    
    if (!wasShown) {
        // Update agent name display
        const agentNameEl = document.getElementById('kbAgentName');
        if (currentAgentId) {
            const agent = myAgents.find(a => a.id === currentAgentId);
            if (agentNameEl) agentNameEl.textContent = agent ? agent.name : '';
        } else {
            if (agentNameEl) agentNameEl.textContent = '（未选择智能体）';
        }
        const uploadBtn = document.querySelector('.kb-panel-upload');
        if (uploadBtn) uploadBtn.style.display = currentAgentId ? '' : 'none';
        loadKbDocs();
        setTimeout(() => { document.addEventListener('click', closeKbPanel, { once: true }); }, 0);
    }
}

function closeKbPanel(e) {
    const panel = document.getElementById('kbPanel');
    if (panel && !panel.contains(e.target) && !e.target.closest('.kb-btn')) {
        panel.classList.remove('show');
    }
}

async function loadKbDocs() {
    const listEl = document.getElementById('kbDocList');
    if (!currentAgentId) {
        listEl.innerHTML = '<div class="kb-empty">请先选择一个智能体</div>';
        return;
    }
    listEl.innerHTML = '<div class="kb-empty">加载中...</div>';
    try {
        const resp = await fetch(`/api/v1/documents?agent_id=${encodeURIComponent(currentAgentId)}`, { headers: apiHeaders() });
        const data = await resp.json();
        console.log('[KB] loadKbDocs response:', JSON.stringify(data));
        // Handle multiple response formats - docs can be strings or objects
        let docs = data.documents || data.files || [];
        if (!Array.isArray(docs)) docs = [];
        // Extract filenames from objects if needed
        docs = docs.map(d => typeof d === 'string' ? d : (d.filename || d.name || d.title || String(d)));
        
        if (docs.length === 0) {
            listEl.innerHTML = '<div class="kb-empty">暂无文档，点击上方按钮上传</div>';
            return;
        }
        let html = '<div class="kb-doc-count">共 ' + docs.length + ' 个文档</div>';
        docs.forEach(docName => {
            const ext = docName.split('.').pop().toLowerCase();
            const icon = ext === 'pdf' ? '📕' : ext === 'docx' ? '📘' : '📄';
            html += '<div class="kb-doc-item">' +
                '<div class="kb-doc-info">' +
                '<span class="kb-doc-icon">' + icon + '</span>' +
                '<span class="kb-doc-name" title="' + escapeHtml(docName) + '">' + escapeHtml(docName) + '</span>' +
                '</div>' +
                '<button class="kb-doc-delete" onclick="deleteKbDoc(\'' + docName.replace(/'/g, "\\'") + '\')" title="删除文档">🗑️</button>' +
                '</div>';
        });
        listEl.innerHTML = html;
    } catch (e) {
        console.error('加载知识库文档列表失败', e);
        listEl.innerHTML = '<div class="kb-empty">加载失败，请重试</div>';
    }
}

async function uploadKbDoc(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    if (!currentAgentId) {
        showToast('请先选择一个智能体');
        input.value = '';
        return;
    }
    showToast('正在上传并索引...');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agent_id', currentAgentId);
    try {
        const resp = await fetch('/api/v1/upload', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} });
        const data = await resp.json();
        if (data.status === 'success') {
            const chunks = data.detail?.chunks || 0;
            showToast(`文档已上传，共 ${chunks} 个分块`);
            loadKbDocs();
        } else {
            showToast(data.detail || '上传失败');
        }
    } catch (e) {
        showToast('上传失败，请重试');
    }
    input.value = '';
}

async function deleteKbDoc(filename) {
    if (!confirm(`确定删除文档「${filename}」？`)) return;
    try {
        const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
        const resp = await fetch(`/api/v1/documents/${encodeURIComponent(filename)}${agentParam}`, { method: 'DELETE', headers: apiHeaders() });
        const data = await resp.json();
        if (data.status === 'success') {
            showToast('文档已删除');
            loadKbDocs();
        } else {
            showToast(data.detail?.message || data.message || '删除失败');
        }
    } catch (e) {
        showToast('删除失败，请重试');
    }
}

// ===== File Drag to Chat Area =====
(function() {
    const chatContent = document.getElementById('chatContent');
    if (!chatContent) return;
    chatContent.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); chatContent.classList.add('drag-over'); });
    chatContent.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); chatContent.classList.remove('drag-over'); });
    chatContent.addEventListener('drop', (e) => { e.preventDefault(); e.stopPropagation(); chatContent.classList.remove('drag-over'); const files = e.dataTransfer.files; if (files.length > 0) handleDroppedFile(files[0]); });
})();

function handleDroppedFile(file) {
    const validExts = ['.pdf','.txt','.docx','.png','.jpg','.jpeg','.gif','.bmp','.webp','.csv','.xlsx','.xls','.doc','.ppt','.pptx','.md','.json','.py','.js','.html','.css'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) { showToast('不支持的文件格式'); return; }
    if (file.size > 50 * 1024 * 1024) { showToast('文件大小超过50MB限制'); return; }
    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileBar').style.display = 'flex';
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => { selectedFileBase64 = e.target.result; };
        reader.readAsDataURL(file);
    } else { selectedFileBase64 = null; }
    showToast('文件已添加：' + file.name);
}

// ===== Mobile Keyboard =====
if (/Mobi|Android/i.test(navigator.userAgent)) {
    window.visualViewport && window.visualViewport.addEventListener('resize', () => {
        const chatContent = document.getElementById('chatContent');
        if (chatContent && document.activeElement && document.activeElement.tagName === 'TEXTAREA') {
            // Adjust layout for virtual keyboard
            const viewportHeight = window.visualViewport.height;
            chatContent.style.height = viewportHeight + 'px';
            setTimeout(() => scrollToBottom(), 100);
        } else {
            chatContent.style.height = '';
        }
    });
    window.visualViewport && window.visualViewport.addEventListener('scroll', () => {
        const chatContent = document.getElementById('chatContent');
        if (chatContent && document.activeElement && document.activeElement.tagName === 'TEXTAREA') {
            // Scroll input into view
            const inputArea = document.querySelector('.chat-input-area');
            if (inputArea) {
                inputArea.scrollIntoView({ block: 'end' });
            }
        }
    });
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', async function() {
    // Drag upload zone
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.addEventListener('dragover', function(e) { e.preventDefault(); e.stopPropagation(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', function(e) { e.preventDefault(); e.stopPropagation(); uploadZone.classList.remove('dragover'); });
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault(); e.stopPropagation(); uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            (async () => { for (let i = 0; i < files.length; i++) { await uploadToKnowledgeBase(files[i]); } await loadDocList(); })();
        }
    });

    // Scroll detection
    setupScrollDetection();

    // Centered mode init
    updateCenteredMode();

    // Try auto-login with saved JWT token
    await tryAutoLogin();

    // Landing page: nav scroll shadow effect
    const landingNav = document.getElementById('landingNav');
    const landingPage = document.getElementById('loginPage');
    if (landingPage && landingNav) {
        landingPage.addEventListener('scroll', function() {
            landingNav.classList.toggle('scrolled', landingPage.scrollTop > 10);
        });
    }

    // Landing page: smooth scroll for anchor links
    document.querySelectorAll('.nav-link, .footer-links a, .coze-footer-links a').forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target && landingPage) {
                    landingPage.scrollTo({
                        top: target.offsetTop - 64,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });

    // Sync agents when tab becomes visible (cross-browser prompt sync)
    // [#12] 不传force=true，受5秒防抖限制，避免频繁Alt-Tab触发大量请求
    document.addEventListener('visibilitychange', async function() {
        if (!document.hidden && currentUser && authToken) {
            await syncAgentsFromServer();
        }
    });

    // Landing page: scroll-reveal animation with IntersectionObserver
    const revealElements = document.querySelectorAll('.reveal');
    if (revealElements.length > 0 && 'IntersectionObserver' in window) {
        // Add .reveal-init to enable animation (content visible by default without it)
        revealElements.forEach(function(el) { el.classList.add('reveal-init'); });
        const revealObserver = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
        revealElements.forEach(function(el) { revealObserver.observe(el); });
    }
});
