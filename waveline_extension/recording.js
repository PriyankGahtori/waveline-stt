// recording.js — runs in recording.html (a real Chrome tab).
// Full page context means getUserMedia mic prompts work correctly.

let audioCtx = null;
let workletNode = null;
let tabStream = null;
let micStream = null;
let serverUrl = 'http://localhost:8000';
let sessionId = null;
let sessionModel = 'whisper';

const statusEl  = document.getElementById('status');
const waveformEl = document.getElementById('waveform');

// ── Send queue ────────────────────────────────────────────────────────────────

const sendQueue = new Map();
let seqCounter = 0;
let queueLoopRunning = false;
let draining = false;

const MAX_RETRIES      = 10;
const MAX_BACKOFF_MS   = 30000;
const QUEUE_INTERVAL_MS = 500;
const MAX_PARALLEL_SENDS = 3;

function enqueueChunk(blob) {
  const seq = seqCounter++;
  sendQueue.set(seq, { blob, retries: 0, nextRetryAt: 0 });
  if (!queueLoopRunning) startQueueLoop();
}

function startQueueLoop() {
  queueLoopRunning = true;
  processQueue();
}

async function processQueue() {
  while (true) {
    const now = Date.now();
    const eligible = [];
    for (const [seq, entry] of sendQueue) {
      if (entry.nextRetryAt <= now) eligible.push([seq, entry]);
      if (eligible.length >= MAX_PARALLEL_SENDS) break;
    }
    if (eligible.length > 0) {
      await Promise.all(eligible.map(([seq, entry]) => trySend(seq, entry)));
    }
    if (draining && sendQueue.size === 0) {
      queueLoopRunning = false;
      cleanup();
      chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' });
      return;
    }
    await sleep(QUEUE_INTERVAL_MS);
  }
}

async function trySend(seq, entry) {
  try {
    const form = new FormData();
    form.append('audio', entry.blob, `chunk-${seq}.wav`);
    form.append('session_id', sessionId);
    form.append('seq', String(seq));
    form.append('model', sessionModel);

    const res = await fetch(`${serverUrl}/transcribe`, { method: 'POST', body: form });

    if (res.status === 404 || res.status === 409) {
      sendQueue.clear();
      chrome.runtime.sendMessage({ type: 'SESSION_LOST' });
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    sendQueue.delete(seq);
    const line = (data?.text || '').trim();
    if (line) chrome.runtime.sendMessage({ type: 'TRANSCRIPT_LINE', line });
  } catch (e) {
    entry.retries++;
    if (entry.retries >= MAX_RETRIES) {
      sendQueue.delete(seq);
      chrome.runtime.sendMessage({ type: 'CHUNK_DEAD', seq });
    } else {
      const backoff = Math.min(1000 * Math.pow(2, entry.retries - 1), MAX_BACKOFF_MS);
      entry.nextRetryAt = Date.now() + backoff;
    }
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Message bus ───────────────────────────────────────────────────────────────

let myTabId = null;

// Get this tab's ID so we can filter targeted messages
chrome.tabs.getCurrent((tab) => {
  myTabId = tab?.id ?? null;
  // Signal SW we're ready, including our tab ID
  chrome.runtime.sendMessage({ type: 'RECORDING_TAB_READY', tabId: myTabId });
});

chrome.runtime.onMessage.addListener((msg) => {
  // Only handle messages targeted at this tab (or broadcast messages)
  if (msg.targetTabId && msg.targetTabId !== myTabId) return;

  if (msg.type === 'INIT_CAPTURE') {
    serverUrl    = msg.serverUrl || serverUrl;
    sessionId    = msg.sessionId;
    sessionModel = msg.model || 'whisper';
    initCapture(msg.streamId, msg.includeMic).catch(e => {
      setStatus('Error: ' + e.message, 'error');
      chrome.runtime.sendMessage({ type: 'CAPTURE_ERROR', error: e.message });
    });
  }
  if (msg.type === 'STOP_CAPTURE') {
    stopCapture();
  }
});

// ── Audio init ────────────────────────────────────────────────────────────────

async function initCapture(streamId, includeMic) {
  setStatus('Connecting to tab audio…');

  tabStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  if (includeMic) {
    setStatus('Requesting microphone…');
    try {
      // This WILL show Chrome's permission prompt since we're a real page.
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000, channelCount: 1 },
        video: false,
      });
      chrome.runtime.sendMessage({ type: 'MIC_STATUS', granted: true });
    } catch (e) {
      setStatus('Mic denied — recording tab audio only', 'error');
      chrome.runtime.sendMessage({ type: 'MIC_STATUS', granted: false, reason: e.name });
    }
  }

  audioCtx = new AudioContext({ sampleRate: 16000 });
  await audioCtx.audioWorklet.addModule(chrome.runtime.getURL('pcm-worklet.js'));

  // GainNode as summing bus — both sources mix to mono automatically
  const mixBus = audioCtx.createGain();
  audioCtx.createMediaStreamSource(tabStream).connect(mixBus);
  if (micStream) audioCtx.createMediaStreamSource(micStream).connect(mixBus);

  workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture', {
    processorOptions: { chunkFrames: 16000 * 4 },
  });

  workletNode.port.onmessage = (e) => {
    if (e.data?.type === 'chunk') {
      enqueueChunk(encodeWavPCM16(e.data.samples, 16000));
    }
    if (e.data?.type === 'flushed') {
      draining = true;
      if (!queueLoopRunning) { cleanup(); chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' }); }
    }
  };

  mixBus.connect(workletNode);

  // Play tab audio back so the user still hears it
  const audioEl = new Audio();
  audioEl.srcObject = tabStream;
  audioEl.play().catch(() => {});

  const micLabel = micStream ? ' + mic' : '';
  setStatus(`Recording · tab${micLabel}`, 'ok');
  waveformEl.classList.add('active');
}

// ── Stop ──────────────────────────────────────────────────────────────────────

function stopCapture() {
  waveformEl.classList.remove('active');
  setStatus('Flushing…');
  // Release mic immediately so Chrome stops showing the mic indicator
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (workletNode) {
    workletNode.port.postMessage({ type: 'flush' });
  } else {
    draining = true;
    if (sendQueue.size === 0) { cleanup(); chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' }); }
  }
}

function cleanup() {
  try { workletNode?.disconnect(); } catch {}
  try { audioCtx?.close(); } catch {}
  try { tabStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { micStream?.getTracks().forEach(t => t.stop()); } catch {}
  audioCtx = workletNode = tabStream = micStream = null;
  setStatus('Done.');
}

function setStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className = 'status-msg' + (type ? ` ${type}` : '');
}

// ── WAV encoder ───────────────────────────────────────────────────────────────

function encodeWavPCM16(float32Samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32Samples.length * 2);
  const view = new DataView(buffer);
  let o = 0;
  const str = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(o++, s.charCodeAt(i)); };
  const u32 = (v) => { view.setUint32(o, v, true); o += 4; };
  const u16 = (v) => { view.setUint16(o, v, true); o += 2; };
  str('RIFF'); u32(36 + float32Samples.length * 2);
  str('WAVE'); str('fmt '); u32(16); u16(1); u16(1);
  u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16);
  str('data'); u32(float32Samples.length * 2);
  for (let i = 0; i < float32Samples.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}
