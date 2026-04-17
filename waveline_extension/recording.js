// recording.js — runs inside recording.html (a real Chrome tab).
// Opened by the service worker; handles tab + mic capture and transcription.

let audioCtx = null;
let workletNode = null;
let tabStream = null;
let micStream = null;
let serverUrl = 'http://localhost:8000';
let sessionId = null;
let sessionModel = 'whisper';

const statusEl   = document.getElementById('status');
const waveformEl = document.getElementById('waveform');

// ── Send queue ────────────────────────────────────────────────────────────────

const sendQueue = new Map();
let seqCounter = 0;
let queueLoopRunning = false;
let draining = false;

const MAX_RETRIES       = 10;
const MAX_BACKOFF_MS    = 30000;
const QUEUE_INTERVAL_MS = 500;
const MAX_PARALLEL_SENDS = 3;

function enqueueChunk(blob) {
  const seq = seqCounter++;
  sendQueue.set(seq, { blob, retries: 0, nextRetryAt: 0 });
  if (!queueLoopRunning) { queueLoopRunning = true; processQueue(); }
}

async function processQueue() {
  while (true) {
    const now = Date.now();
    const eligible = [];
    for (const [seq, entry] of sendQueue) {
      if (entry.nextRetryAt <= now) eligible.push([seq, entry]);
      if (eligible.length >= MAX_PARALLEL_SENDS) break;
    }
    if (eligible.length) await Promise.all(eligible.map(([s, e]) => trySend(s, e)));
    if (draining && sendQueue.size === 0) {
      queueLoopRunning = false;
      cleanup();
      chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' });
      return;
    }
    await new Promise(r => setTimeout(r, QUEUE_INTERVAL_MS));
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
      entry.nextRetryAt = Date.now() + Math.min(1000 * 2 ** (entry.retries - 1), MAX_BACKOFF_MS);
    }
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

// Read all config from URL params — SW passed everything here synchronously
const _params = new URLSearchParams(location.search);
const _streamId = _params.get('streamId');
sessionId    = _params.get('sessionId');
serverUrl    = _params.get('serverUrl') || serverUrl;
sessionModel = _params.get('model') || 'whisper';
const _includeMic = _params.get('mic') === '1';

if (_streamId && sessionId) {
  initCapture(_streamId, _includeMic).catch(e => setStatus('Error: ' + e.message, 'error'));
} else {
  setStatus('Missing session config.', 'error');
}

// ── Message bus ───────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'STOP_CAPTURE') stopCapture();
});

// ── Audio capture ─────────────────────────────────────────────────────────────

async function initCapture(streamId, includeMic) {
  setStatus('Connecting to tab audio…');

  tabStream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } },
    video: false,
  });

  if (includeMic) {
    setStatus('Requesting microphone…');
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000, channelCount: 1 },
        video: false,
      });
      chrome.runtime.sendMessage({ type: 'MIC_STATUS', granted: true });
    } catch (e) {
      setStatus('Mic denied — tab audio only', 'error');
      chrome.runtime.sendMessage({ type: 'MIC_STATUS', granted: false });
    }
  }

  audioCtx = new AudioContext({ sampleRate: 16000 });
  await audioCtx.audioWorklet.addModule(chrome.runtime.getURL('pcm-worklet.js'));

  const mixBus = audioCtx.createGain();
  audioCtx.createMediaStreamSource(tabStream).connect(mixBus);
  if (micStream) audioCtx.createMediaStreamSource(micStream).connect(mixBus);

  workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture', {
    processorOptions: { chunkFrames: 16000 * 4 },
  });

  workletNode.port.onmessage = (e) => {
    if (e.data?.type === 'chunk') enqueueChunk(encodeWav(e.data.samples, 16000));
    if (e.data?.type === 'flushed') {
      draining = true;
      if (!queueLoopRunning) { cleanup(); chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' }); }
    }
  };

  mixBus.connect(workletNode);

  // Play tab audio back so user still hears it
  const el = new Audio(); el.srcObject = tabStream; el.play().catch(() => {});

  const label = micStream ? 'Tab + Mic' : 'Tab audio';
  setStatus(`Recording · ${label}`, 'ok');
  waveformEl.classList.add('active');
}

function stopCapture() {
  waveformEl.classList.remove('active');
  setStatus('Stopping…');
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
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
  audioCtx = workletNode = tabStream = null;
  setStatus('Done.');
}

function setStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className = 'status-msg' + (type ? ` ${type}` : '');
}

// ── WAV encoder ───────────────────────────────────────────────────────────────

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  let o = 0;
  const str = s => { for (let i = 0; i < s.length; i++) v.setUint8(o++, s.charCodeAt(i)); };
  const u32 = n => { v.setUint32(o, n, true); o += 4; };
  const u16 = n => { v.setUint16(o, n, true); o += 2; };
  str('RIFF'); u32(36 + samples.length * 2);
  str('WAVE'); str('fmt '); u32(16); u16(1); u16(1);
  u32(rate); u32(rate * 2); u16(2); u16(16);
  str('data'); u32(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
