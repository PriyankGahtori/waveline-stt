// service_worker.js
// Owns the tab-capture stream ID and offscreen document lifecycle.

const OFFSCREEN_URL = chrome.runtime.getURL('offscreen.html');

let offscreenReady = false;

// ── Offscreen document helpers ──────────────────────────────────────────────

async function ensureOffscreen() {
  const existing = await chrome.offscreen.hasDocument?.() ?? false;
  if (existing) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['USER_MEDIA', 'AUDIO_PLAYBACK'],
    justification: 'Capture and transcribe meeting audio, and play it back so the tab audio is not muted.',
  });
}

async function closeOffscreen() {
  try {
    const has = await chrome.offscreen.hasDocument?.() ?? false;
    if (has) await chrome.offscreen.closeDocument();
  } catch {}
  offscreenReady = false;
}

// ── Message bus ─────────────────────────────────────────────────────────────

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

  // Relay transcript lines from offscreen → popup
  if (msg.type === 'TRANSCRIPT_LINE') {
    chrome.storage.local.get(['waveline_notes'], (res) => {
      const prev = res.waveline_notes || '';
      const next = prev + (prev && !prev.endsWith('\n') ? '\n' : '') + msg.line + '\n';
      chrome.storage.local.set({ waveline_notes: next });
    });
    return false;
  }

  // Relay dead chunk warning to popup
  if (msg.type === 'CHUNK_DEAD') {
    chrome.runtime.sendMessage({ type: 'CHUNK_DEAD', seq: msg.seq }).catch(() => {});
    return false;
  }

  if (msg.type === 'OFFSCREEN_READY') {
    offscreenReady = true;
    return false;
  }

  // Offscreen signals queue drained after stop — now safe to close
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
      await closeOffscreen();
      chrome.storage.local.set({ waveline_recording: false, waveline_session_id: null });
    });
    return false;
  }
});

// ── Start flow ───────────────────────────────────────────────────────────────

async function handleStart({ includeMic, model }) {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab) throw new Error('No active tab found.');

  const streamId = await new Promise((resolve, reject) => {
    chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id }, (id) => {
      if (chrome.runtime.lastError || !id) return reject(new Error(chrome.runtime.lastError?.message || 'tabCapture failed'));
      resolve(id);
    });
  });

  await ensureOffscreen();

  if (!offscreenReady) {
    await new Promise((resolve, reject) => {
      const deadline = setTimeout(() => reject(new Error('Offscreen timed out')), 3000);
      const check = setInterval(() => {
        if (offscreenReady) { clearInterval(check); clearTimeout(deadline); resolve(); }
      }, 50);
    });
  }

  const { waveline_server_url } = await chrome.storage.local.get(['waveline_server_url']);
  const serverUrl = (waveline_server_url || 'http://localhost:8000').replace(/\/$/, '');
  const selectedModel = model || 'whisper';

  // Create session on backend
  const form = new FormData();
  form.append('model', selectedModel);
  const sessionRes = await fetch(`${serverUrl}/session/start`, { method: 'POST', body: form });
  if (!sessionRes.ok) throw new Error(`Failed to start session: HTTP ${sessionRes.status}`);
  const { session_id: sessionId } = await sessionRes.json();

  chrome.storage.local.set({ waveline_session_id: sessionId });

  await chrome.runtime.sendMessage({
    type: 'INIT_CAPTURE',
    streamId,
    includeMic,
    serverUrl,
    sessionId,
    model: selectedModel,
  });

  chrome.storage.local.set({ waveline_recording: true });
  return { ok: true, sessionId };
}

// ── Stop flow ────────────────────────────────────────────────────────────────

async function handleStop() {
  // Tell offscreen to flush audio and drain queue.
  // Actual close + session/stop happens when QUEUE_DRAINED message arrives.
  await chrome.runtime.sendMessage({ type: 'STOP_CAPTURE' }).catch(() => {});
  return { ok: true };
}
