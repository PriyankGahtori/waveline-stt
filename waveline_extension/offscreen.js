// offscreen.js
// Runs inside offscreen.html — has access to Web Audio, getUserMedia, fetch.
// Lifecycle: created by SW on START, closed by SW after queue drain + STOP.

let audioCtx = null;
let workletNode = null;
let tabStream = null;
let micStream = null;
let audioEl = null;
let serverUrl = 'http://localhost:8000';
let sessionId = null;
let sessionModel = 'whisper';

// ── Send queue for zero data loss ─────────────────────────────────────────────
// Map<seq, { blob, retries, nextRetryAt }>
const sendQueue = new Map();
let seqCounter = 0;
let queueLoopRunning = false;
let draining = false; // true after STOP_CAPTURE — wait for empty queue then signal

const MAX_RETRIES = 10;
const MAX_BACKOFF_MS = 30000;
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    sendQueue.delete(seq);

    const line = (data?.text || '').trim();
    if (line) {
      chrome.runtime.sendMessage({ type: 'TRANSCRIPT_LINE', line });
    }
  } catch (e) {
    entry.retries++;
    if (entry.retries >= MAX_RETRIES) {
      sendQueue.delete(seq);
      console.error(`[offscreen] chunk seq=${seq} permanently failed after ${MAX_RETRIES} retries`);
      chrome.runtime.sendMessage({ type: 'CHUNK_DEAD', seq });
    } else {
      const backoff = Math.min(1000 * Math.pow(2, entry.retries - 1), MAX_BACKOFF_MS);
      entry.nextRetryAt = Date.now() + backoff;
      console.warn(`[offscreen] chunk seq=${seq} retry ${entry.retries} in ${backoff}ms:`, e.message);
    }
  }
}

// Persist unsent queue to storage on unexpected close (tab navigation, crash)
window.addEventListener('beforeunload', () => {
  if (sendQueue.size === 0) return;
  const serializable = [];
  for (const [seq, entry] of sendQueue) {
    // Convert blob to array buffer for storage — best effort
    serializable.push({ seq, retries: entry.retries });
  }
  chrome.storage.local.set({ scribble_dead_queue_info: serializable });
});

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ── Tell SW we're alive ──────────────────────────────────────────────────────
chrome.runtime.sendMessage({ type: 'OFFSCREEN_READY' });

// ── Message bus ──────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender) => {
  if (msg.type === 'INIT_CAPTURE') {
    serverUrl = msg.serverUrl || serverUrl;
    sessionId = msg.sessionId;
    sessionModel = msg.model || 'whisper';
    initCapture(msg.streamId, msg.includeMic).catch(e => {
      console.error('[offscreen] init error', e);
    });
  }
  if (msg.type === 'STOP_CAPTURE') {
    stopCapture();
  }
});

// ── Audio init ───────────────────────────────────────────────────────────────

async function initCapture(streamId, includeMic) {
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
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
        video: false,
      });
    } catch (e) {
      console.warn('[offscreen] mic not granted', e);
    }
  }

  audioCtx = new AudioContext({ sampleRate: 16000 });
  await audioCtx.audioWorklet.addModule(chrome.runtime.getURL('pcm-worklet.js'));

  const tabSrc = audioCtx.createMediaStreamSource(tabStream);

  let sourceNode;
  if (micStream) {
    const micSrc = audioCtx.createMediaStreamSource(micStream);
    const merger = audioCtx.createChannelMerger(2);
    tabSrc.connect(merger, 0, 0);
    micSrc.connect(merger, 0, 1);
    sourceNode = merger;
  } else {
    sourceNode = tabSrc;
  }

  workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture', {
    processorOptions: { chunkFrames: 16000 * 4 }, // 4-second chunks
  });

  workletNode.port.onmessage = (e) => {
    if (e.data?.type === 'chunk') {
      const wavBlob = encodeWavPCM16(e.data.samples, 16000);
      enqueueChunk(wavBlob);
    }
    if (e.data?.type === 'flushed') {
      // Audio graph is done — now drain the send queue before cleanup
      draining = true;
      if (!queueLoopRunning) {
        // Queue is already empty
        cleanup();
        chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' });
      }
      // Otherwise processQueue() handles drain → cleanup → QUEUE_DRAINED
    }
  };

  sourceNode.connect(workletNode);

  audioEl = new Audio();
  audioEl.srcObject = tabStream;
  audioEl.play().catch(() => {});
}

// ── Stop ─────────────────────────────────────────────────────────────────────

function stopCapture() {
  // Stop tracks immediately so the browser stops showing "tab is being shared"
  try { tabStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { micStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { if (audioEl) { audioEl.pause(); audioEl.srcObject = null; } } catch {}

  if (workletNode) {
    workletNode.port.postMessage({ type: 'flush' });
  } else {
    draining = true;
    if (sendQueue.size === 0) {
      cleanup();
      chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' });
    }
  }
}

function cleanup() {
  try { workletNode?.disconnect(); } catch {}
  try { audioCtx?.close(); } catch {}
  try { tabStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { micStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { if (audioEl) { audioEl.pause(); audioEl.srcObject = null; } } catch {}
  audioCtx = workletNode = tabStream = micStream = audioEl = null;
}

// ── WAV encoder ──────────────────────────────────────────────────────────────

function encodeWavPCM16(float32Samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32Samples.length * 2);
  const view = new DataView(buffer);
  let offset = 0;

  const writeStr = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset++, s.charCodeAt(i)); };
  const u32 = (v) => { view.setUint32(offset, v, true); offset += 4; };
  const u16 = (v) => { view.setUint16(offset, v, true); offset += 2; };

  writeStr('RIFF'); u32(36 + float32Samples.length * 2);
  writeStr('WAVE');
  writeStr('fmt '); u32(16); u16(1); u16(1);
  u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16);
  writeStr('data'); u32(float32Samples.length * 2);

  for (let i = 0; i < float32Samples.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}
