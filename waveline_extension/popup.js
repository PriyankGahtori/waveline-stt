// popup.js — UI layer. All recording happens in offscreen.js via service_worker.js.

const startBtn        = document.getElementById('startBtn');
const stopBtn         = document.getElementById('stopBtn');
const clearBtn        = document.getElementById('clearBtn');
const exportBtn       = document.getElementById('exportBtn');
const micBtn          = document.getElementById('micBtn');
const notesEl         = document.getElementById('notes');
const statusEl        = document.getElementById('status');
const modelSelect     = document.getElementById('modelSelect');
const connStatus      = document.getElementById('connStatus');
const connText        = document.getElementById('connText');
const deadWarning     = document.getElementById('deadWarning');
const deadWarningText = document.getElementById('deadWarningText');
const charCount       = document.getElementById('charCount');
const themeToggle     = document.getElementById('themeToggle');
const iconSun         = document.getElementById('iconSun');
const iconMoon        = document.getElementById('iconMoon');
const waveformEl      = document.getElementById('waveform');
const sourceIndicators = document.getElementById('sourceIndicators');
const optionsRow      = document.getElementById('optionsRow');
const tabChip         = document.getElementById('tabChip');
const micChip         = document.getElementById('micChip');

let micEnabled = false;

// ── Theme ────────────────────────────────────────────────────────────────────

function applyTheme(dark) {
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : '');
  iconSun.style.display  = dark ? 'block' : 'none';
  iconMoon.style.display = dark ? 'none'  : 'block';
}

chrome.storage.local.get(['waveline_dark_theme'], (res) => {
  applyTheme(!!res.waveline_dark_theme);
});

themeToggle.addEventListener('click', () => {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  applyTheme(!isDark);
  chrome.storage.local.set({ waveline_dark_theme: !isDark });
});

// ── Char count ───────────────────────────────────────────────────────────────

function updateCharCount() {
  const len = notesEl.value.length;
  charCount.textContent = len === 0 ? '0 chars' : `${len.toLocaleString()} chars`;
}
notesEl.addEventListener('input', updateCharCount);

// ── Init: load saved state ───────────────────────────────────────────────────

chrome.runtime.sendMessage({ type: 'GET_STATE' }, (res) => {
  if (chrome.runtime.lastError) return;
  notesEl.value = res.notes || '';
  updateCharCount();
  // We don't know mic state on reopen — show tab-only indicator if recording
  setUI(!!res.recording, false);
});

chrome.storage.local.get(['waveline_model'], (res) => {
  if (res.waveline_model && modelSelect) modelSelect.value = res.waveline_model;
});

// ── Live transcript relay from SW ────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'TRANSCRIPT_LINE') {
    const line = msg.line || '';
    if (notesEl.value && !notesEl.value.endsWith('\n')) notesEl.value += '\n';
    notesEl.value += line + '\n';
    notesEl.scrollTop = notesEl.scrollHeight;
    updateCharCount();
  }
  if (msg.type === 'CHUNK_DEAD') {
    deadWarningText.textContent = `Chunk #${msg.seq} permanently failed — gap in transcript.`;
    deadWarning.style.display = 'flex';
  }
  if (msg.type === 'SESSION_LOST') {
    setUI(false, false);
    setStatus('Session lost — server restarted. Start a new recording.', 'error');
  }
  if (msg.type === 'MIC_STATUS') {
    if (msg.granted) {
      setStatus('Recording · mic active', 'ok');
    } else {
      const reason = msg.reason === 'NotAllowedError' ? 'Mic permission denied' : 'Mic unavailable';
      setStatus(`${reason} — tab audio only`, 'error');
    }
  }
});

// ── Server URL settings ───────────────────────────────────────────────────────

const serverUrlEl = document.getElementById('serverUrl');
const saveUrlBtn  = document.getElementById('saveUrlBtn');
const urlStatusEl = document.getElementById('urlStatus');
const DEFAULT_URL = 'http://localhost:8000';

chrome.storage.local.get(['waveline_server_url'], (res) => {
  serverUrlEl.value = res.waveline_server_url || DEFAULT_URL;
});

saveUrlBtn.addEventListener('click', () => {
  const raw = serverUrlEl.value.trim().replace(/\/$/, '');
  if (!raw) { urlStatusEl.textContent = 'URL cannot be empty.'; urlStatusEl.style.color = 'var(--red)'; return; }
  try { new URL(raw); } catch {
    urlStatusEl.textContent = 'Invalid URL.'; urlStatusEl.style.color = 'var(--red)'; return;
  }
  chrome.storage.local.set({ waveline_server_url: raw }, () => {
    urlStatusEl.textContent = 'Saved ✓';
    urlStatusEl.style.color = 'var(--green)';
    setTimeout(() => { urlStatusEl.textContent = ''; }, 2000);
    checkConnection(raw);
  });
});

// ── Model selector ────────────────────────────────────────────────────────────

if (modelSelect) {
  modelSelect.addEventListener('change', () => {
    chrome.storage.local.set({ waveline_model: modelSelect.value });
  });
}


// ── Connection health check ───────────────────────────────────────────────────

async function checkConnection(baseUrl) {
  if (!connStatus) return;
  const url = (baseUrl || DEFAULT_URL).replace(/\/$/, '');
  connText.textContent = 'Checking…';
  connStatus.className = 'conn-badge';
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (res.ok && data.ok) {
      const loaded = Object.entries(data.models || {})
        .filter(([, v]) => v)
        .map(([k]) => k)
        .join(' · ');
      connText.textContent = loaded ? `Connected · ${loaded}` : 'Connected';
      connStatus.className = 'conn-badge conn-ok';
    } else {
      connText.textContent = 'Server error';
      connStatus.className = 'conn-badge conn-warn';
    }
  } catch {
    connText.textContent = 'Unreachable';
    connStatus.className = 'conn-badge conn-err';
  }
}

chrome.storage.local.get(['waveline_server_url'], (res) => {
  checkConnection(res.waveline_server_url || DEFAULT_URL);
});

// ── Notes persistence ────────────────────────────────────────────────────────

const saveNotes = debounce(() => {
  chrome.storage.local.set({ waveline_notes: notesEl.value });
}, 300);
notesEl.addEventListener('input', saveNotes);

// ── Mic button ───────────────────────────────────────────────────────────────

micBtn.addEventListener('click', () => {
  micEnabled = !micEnabled;
  micBtn.classList.toggle('active', micEnabled);
  micBtn.setAttribute('aria-pressed', String(micEnabled));
  if (!micEnabled) setStatus('');
});

// ── Buttons ──────────────────────────────────────────────────────────────────

startBtn.addEventListener('click', async () => {
  if (deadWarning) deadWarning.style.display = 'none';

  const useMic = micEnabled;
  setUI(true, useMic);
  setStatus(useMic ? 'Requesting mic permission…' : 'Starting…');
  const model = modelSelect ? modelSelect.value : 'whisper';
  chrome.runtime.sendMessage(
    { type: 'START_RECORDING', includeMic: useMic, model },
    (res) => {
      if (res?.error) {
        setStatus('Error: ' + res.error, 'error');
        setUI(false, false);
      } else {
        setStatus('');
      }
    }
  );
});

stopBtn.addEventListener('click', () => {
  setStatus('Stopping…');
  chrome.runtime.sendMessage({ type: 'STOP_RECORDING' }, () => {
    setUI(false, false);
    setStatus('Stopped.');
  });
});

clearBtn.addEventListener('click', () => {
  notesEl.value = '';
  updateCharCount();
  chrome.storage.local.set({ waveline_notes: '' });
});

exportBtn.addEventListener('click', () => {
  const blob = new Blob([notesEl.value || ''], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  chrome.downloads.download({ url, filename: `waveline-${Date.now()}.txt`, saveAs: true });
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function setUI(isRecording, useMic) {
  startBtn.disabled = isRecording;
  stopBtn.disabled  = !isRecording;
  if (modelSelect) modelSelect.disabled = isRecording;

  if (isRecording) {
    // Swap options row for source indicators
    optionsRow.style.display = 'none';
    sourceIndicators.style.display = 'flex';
    tabChip.classList.add('active');
    if (useMic) {
      micChip.style.display = 'flex';
      micChip.classList.add('active');
    } else {
      micChip.style.display = 'none';
    }
    waveformEl.classList.add('active');
    document.getElementById('recIndicator').classList.add('active');
  } else {
    optionsRow.style.display = 'flex';
    sourceIndicators.style.display = 'none';
    tabChip.classList.remove('active');
    micChip.classList.remove('active');
    waveformEl.classList.remove('active');
    document.getElementById('recIndicator').classList.remove('active');
    // Reset mic button
    micEnabled = false;
    micBtn.classList.remove('active');
    micBtn.setAttribute('aria-pressed', 'false');
  }
}

function setStatus(msg, type) {
  if (!statusEl) return;
  statusEl.textContent = msg;
  statusEl.className = 'status-text' + (type ? ` ${type}` : '');
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}
