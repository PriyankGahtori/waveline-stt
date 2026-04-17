// service_worker.js

// ── Recording tab helpers ─────────────────────────────────────────────────────
// recordingTabId is persisted to storage so it survives SW suspension/restart.

async function getRecordingTabId() {
  const res = await chrome.storage.local.get(['waveline_recording_tab_id']);
  return res.waveline_recording_tab_id || null;
}

async function setRecordingTabId(id) {
  if (id) {
    await chrome.storage.local.set({ waveline_recording_tab_id: id });
  } else {
    await chrome.storage.local.remove('waveline_recording_tab_id');
  }
}

async function sendToRecordingTab(msg) {
  const tabId = await getRecordingTabId();
  if (!tabId) return;
  // Extension pages in tabs receive via chrome.runtime.sendMessage with targetTabId filter
  chrome.runtime.sendMessage({ ...msg, targetTabId: tabId }).catch(() => {});
}

async function closeRecordingTab() {
  const tabId = await getRecordingTabId();
  if (tabId) {
    try { await chrome.tabs.remove(tabId); } catch {}
  }
  await setRecordingTabId(null);
}

// ── Message bus ──────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'START_RECORDING') {
    handleStart(msg).then(sendResponse).catch(e => sendResponse({ error: e.message }));
    return true;
  }

  if (msg.type === 'STOP_RECORDING') {
    handleStop().then(sendResponse).catch(e => sendResponse({ error: e.message }));
    return true;
  }

  if (msg.type === 'GET_STATE') {
    chrome.storage.local.get(['waveline_recording', 'waveline_notes', 'waveline_session_id'], (res) => {
      sendResponse({
        recording: !!res.waveline_recording,
        notes: res.waveline_notes || '',
        sessionId: res.waveline_session_id || null,
      });
    });
    return true;
  }

  if (msg.type === 'RECORDING_TAB_READY') {
    // Recording tab signals it's loaded and sends its own tab ID
    if (msg.tabId) setRecordingTabId(msg.tabId);
    return false;
  }

  // Persist + relay transcript line to popup
  if (msg.type === 'TRANSCRIPT_LINE') {
    chrome.storage.local.get(['waveline_notes'], (res) => {
      const prev = res.waveline_notes || '';
      const next = prev + (prev && !prev.endsWith('\n') ? '\n' : '') + msg.line + '\n';
      chrome.storage.local.set({ waveline_notes: next });
    });
    // Relay to popup (if open) — targetTabId not set so all extension pages get it
    chrome.runtime.sendMessage({ type: 'TRANSCRIPT_LINE', line: msg.line }).catch(() => {});
    return false;
  }

  if (msg.type === 'SESSION_LOST') {
    chrome.storage.local.set({ waveline_recording: false, waveline_session_id: null });
    closeRecordingTab();
    chrome.runtime.sendMessage({ type: 'SESSION_LOST' }).catch(() => {});
    return false;
  }

  if (msg.type === 'CHUNK_DEAD') {
    chrome.runtime.sendMessage({ type: 'CHUNK_DEAD', seq: msg.seq }).catch(() => {});
    return false;
  }

  if (msg.type === 'MIC_STATUS') {
    chrome.runtime.sendMessage({ type: 'MIC_STATUS', granted: msg.granted, reason: msg.reason }).catch(() => {});
    return false;
  }

  if (msg.type === 'QUEUE_DRAINED') {
    chrome.storage.local.get(['waveline_session_id', 'waveline_server_url'], async (res) => {
      const serverUrl = (res.waveline_server_url || 'http://localhost:8000').replace(/\/$/, '');
      const sessionId = res.waveline_session_id;
      if (sessionId) {
        try {
          const form = new FormData();
          form.append('session_id', sessionId);
          await fetch(`${serverUrl}/session/stop`, { method: 'POST', body: form });
        } catch (e) {
          console.warn('[sw] session/stop failed:', e);
        }
      }
      await closeRecordingTab();
      chrome.storage.local.set({ waveline_recording: false, waveline_session_id: null });
    });
    return false;
  }
});

// ── Start flow ───────────────────────────────────────────────────────────────

async function handleStart({ includeMic, model }) {
  // Get stream ID BEFORE opening recording tab (active tab must be the target)
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab) throw new Error('No active tab found.');

  const streamId = await new Promise((resolve, reject) => {
    chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id }, (id) => {
      if (chrome.runtime.lastError || !id)
        return reject(new Error(chrome.runtime.lastError?.message || 'tabCapture failed'));
      resolve(id);
    });
  });

  // Open recording tab in background
  const recTab = await new Promise((resolve, reject) => {
    chrome.tabs.create(
      { url: chrome.runtime.getURL('recording.html'), active: false },
      (t) => {
        if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
        resolve(t);
      }
    );
  });

  // Wait for recording tab to signal ready (it sends RECORDING_TAB_READY with its tabId)
  await new Promise((resolve, reject) => {
    const deadline = setTimeout(() => reject(new Error('Recording tab timed out')), 6000);
    const poll = setInterval(async () => {
      const id = await getRecordingTabId();
      if (id === recTab.id) { clearInterval(poll); clearTimeout(deadline); resolve(); }
    }, 100);
  });

  const { waveline_server_url } = await chrome.storage.local.get(['waveline_server_url']);
  const serverUrl = (waveline_server_url || 'http://localhost:8000').replace(/\/$/, '');
  const selectedModel = model || 'whisper';

  const form = new FormData();
  form.append('model', selectedModel);
  const sessionRes = await fetch(`${serverUrl}/session/start`, { method: 'POST', body: form });
  if (!sessionRes.ok) throw new Error(`Failed to start session: HTTP ${sessionRes.status}`);
  const { session_id: sessionId } = await sessionRes.json();

  chrome.storage.local.set({ waveline_session_id: sessionId, waveline_recording: true });

  // Send init to recording tab via runtime broadcast with targetTabId
  chrome.runtime.sendMessage({
    type: 'INIT_CAPTURE',
    targetTabId: recTab.id,
    streamId,
    includeMic,
    serverUrl,
    sessionId,
    model: selectedModel,
  }).catch(() => {});

  return { ok: true, sessionId };
}

// ── Stop flow ────────────────────────────────────────────────────────────────

async function handleStop() {
  await sendToRecordingTab({ type: 'STOP_CAPTURE' });
  return { ok: true };
}
