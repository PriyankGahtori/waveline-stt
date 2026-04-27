// popup.js — UI layer. All recording happens in offscreen.js via service_worker.js.

const startBtn    = document.getElementById('startBtn');
const stopBtn     = document.getElementById('stopBtn');
const clearBtn    = document.getElementById('clearBtn');
const exportBtn   = document.getElementById('exportBtn');
const notesEl     = document.getElementById('notes');
const statusEl    = document.getElementById('status');
const modelSelect = document.getElementById('modelSelect');
const connStatus  = document.getElementById('connStatus');
const connText    = document.getElementById('connText');
const deadWarning = document.getElementById('deadWarning');
const deadWarningText = document.getElementById('deadWarningText');
const charCount   = document.getElementById('charCount');
const themeToggle = document.getElementById('themeToggle');
const iconSun     = document.getElementById('iconSun');
const iconMoon    = document.getElementById('iconMoon');
const waveformEl  = document.getElementById('waveform');
const idleRow     = document.getElementById('idleRow');
const recordingRow = document.getElementById('recordingRow');

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
  if (chrome.runtime.lastError || !res) return;
  notesEl.value = res.notes || '';
  updateCharCount();
  if (res.recording) setUI(true);
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
  if (msg.type === 'QUEUE_DRAINED') {
    startBtn.disabled = false;
    setStatus('');
  }
  if (msg.type === 'SESSION_LOST') {
    startBtn.disabled = false;
    setUI(false);
    setStatus('Server restarted — session lost. Start a new recording.', 'error');
  }
});

// ── Server URL settings ───────────────────────────────────────────────────────

const serverUrlEl        = document.getElementById('serverUrl');
const saveUrlBtn         = document.getElementById('saveUrlBtn');
const urlStatusEl        = document.getElementById('urlStatus');
const transportSelectEl  = document.getElementById('transportSelect');
const micToggleEl        = document.getElementById('micToggle');
const silenceThresholdEl = document.getElementById('silenceThreshold');
const silenceValEl       = document.getElementById('silenceThresholdVal');
const languageEl         = document.getElementById('languageInput');
const DEFAULT_URL        = 'http://localhost:8000';

chrome.storage.local.get(['waveline_server_url', 'waveline_include_mic', 'waveline_silence_threshold', 'waveline_transport', 'waveline_language'], (res) => {
  serverUrlEl.value = res.waveline_server_url || DEFAULT_URL;
  if (transportSelectEl) transportSelectEl.value = res.waveline_transport || 'ws';
  if (micToggleEl) micToggleEl.checked = !!res.waveline_include_mic;
  const thresh = res.waveline_silence_threshold ?? 0.003;
  if (silenceThresholdEl) { silenceThresholdEl.value = thresh; silenceValEl.textContent = thresh; }
  if (languageEl) languageEl.value = res.waveline_language || '';
});

if (transportSelectEl) {
  transportSelectEl.addEventListener('change', () => {
    chrome.storage.local.set({ waveline_transport: transportSelectEl.value });
  });
}

if (languageEl) {
  languageEl.addEventListener('change', () => {
    chrome.storage.local.set({ waveline_language: languageEl.value.trim().toLowerCase() });
  });
}

if (micToggleEl) {
  micToggleEl.addEventListener('change', () => {
    chrome.storage.local.set({ waveline_include_mic: micToggleEl.checked });
  });
}

if (silenceThresholdEl) {
  silenceThresholdEl.addEventListener('input', () => {
    const v = parseFloat(silenceThresholdEl.value);
    silenceValEl.textContent = v.toFixed(3);
    chrome.storage.local.set({ waveline_silence_threshold: v });
  });
}

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
      const models = data.models || {};
      // Update model selector: disable unavailable, auto-select first available
      if (modelSelect) {
        Array.from(modelSelect.options).forEach(opt => {
          opt.disabled = !models[opt.value];
        });
        if (models[modelSelect.value] === false || !models[modelSelect.value]) {
          const first = Array.from(modelSelect.options).find(o => !o.disabled);
          if (first) { modelSelect.value = first.value; chrome.storage.local.set({ waveline_model: first.value }); }
        }
      }
      const loaded = Object.entries(models).filter(([, v]) => v).map(([k]) => k).join(' · ');
      connText.textContent = loaded ? `Connected · ${loaded}` : 'Connected';
      connStatus.className = 'conn-badge conn-ok';
      // Show server default language as placeholder on the language input
      if (languageEl && data.defaults?.whisper_language) {
        languageEl.placeholder = `server default: ${data.defaults.whisper_language}`;
      }
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

// ── Buttons ──────────────────────────────────────────────────────────────────

startBtn.addEventListener('click', async () => {
  if (deadWarning) deadWarning.style.display = 'none';
  setStatus('Starting…');
  const model = modelSelect ? modelSelect.value : 'whisper';
  chrome.storage.local.get(['waveline_include_mic', 'waveline_silence_threshold', 'waveline_transport', 'waveline_language'], (prefs) => {
    chrome.runtime.sendMessage(
      {
        type: 'START_RECORDING',
        model,
        includeMic: !!prefs.waveline_include_mic,
        silenceThreshold: prefs.waveline_silence_threshold ?? 0.003,
        transport: prefs.waveline_transport || 'ws',
        language: prefs.waveline_language || '',  // '' = use server default
      },
      (res) => {
        if (res?.error) {
          setStatus('Error: ' + res.error, 'error');
        } else {
          setUI(true);
          setStatus('');
        }
      }
    );
  });
});

stopBtn.addEventListener('click', () => {
  setUI(false);
  setStatus('Finishing…');
  startBtn.disabled = true; // block re-start until drain completes
  chrome.runtime.sendMessage({ type: 'STOP_RECORDING' }).catch(() => {});
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

function setUI(isRecording) {
  idleRow.style.display      = isRecording ? 'none' : 'flex';
  recordingRow.style.display = isRecording ? 'flex' : 'none';
  if (modelSelect) modelSelect.disabled = isRecording;
  if (isRecording) {
    waveformEl.classList.add('active');
  } else {
    waveformEl.classList.remove('active');
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
