// offscreen.js — audio capture + streaming transcription via WebSocket

let audioCtx = null;
let workletNode = null;
let tabStream = null;
let micStream = null;
let audioEl = null;
let serverUrl = 'http://localhost:8000';
let sessionId = null;
let sessionModel = 'whisper';

// ── Transport ─────────────────────────────────────────────────────────────────
// 'ws'   = WebSocket preferred, HTTP fallback on failure
// 'http' = HTTP only, no WebSocket
let useWebSocket = true; // configurable from popup

// ── WebSocket streaming ───────────────────────────────────────────────────────
let ws = null;
let wsReady = false;
let wsReconnectTimer = null;
const WS_MAX_RECONNECT_MS = 8000;
let wsReconnectDelay = 500;

// Silence gate — configurable from popup settings, default 0.003
let SILENCE_RMS_THRESHOLD = 0.003;

// ── HTTP fallback send queue (used if WS unavailable) ────────────────────────
const sendQueue = new Map();
let seqCounter = 0;
let queueLoopRunning = false;
let draining = false;

const MAX_RETRIES = 8;
const MAX_BACKOFF_MS = 16000;
const QUEUE_INTERVAL_MS = 80; // was 500ms — much more responsive
const MAX_PARALLEL_SENDS = 4;

// ── WebSocket connect ─────────────────────────────────────────────────────────

function wsUrl() {
  return serverUrl.replace(/^http/, 'ws') + `/ws/transcribe/${sessionId}?model=${sessionModel}`;
}

function connectWs() {
  if (!sessionId || !useWebSocket) return;
  clearTimeout(wsReconnectTimer);
  try {
    ws = new WebSocket(wsUrl());
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      wsReady = true;
      wsReconnectDelay = 500;
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.text) {
          chrome.runtime.sendMessage({ type: 'TRANSCRIPT_LINE', line: data.text.trim() });
        }
        if (data.seq !== undefined) {
          sendQueue.delete(data.seq); // ack
        }
      } catch {}
    };

    ws.onerror = () => { wsReady = false; };

    ws.onclose = () => {
      wsReady = false;
      ws = null;
      if (!draining && sessionId) {
        wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_MAX_RECONNECT_MS);
        wsReconnectTimer = setTimeout(connectWs, wsReconnectDelay);
      }
    };
  } catch {
    wsReady = false;
  }
}

function closeWs() {
  clearTimeout(wsReconnectTimer);
  wsReady = false;
  try { ws?.close(); } catch {}
  ws = null;
}

// ── Chunk dispatch ────────────────────────────────────────────────────────────

function enqueueChunk(blob, rms) {
  // Skip silent chunks to reduce server load and latency
  if (rms < SILENCE_RMS_THRESHOLD) return;

  const seq = seqCounter++;

  // Try WebSocket first (lowest latency)
  if (wsReady && ws?.readyState === WebSocket.OPEN) {
    const reader = new FileReader();
    reader.onload = () => {
      if (ws?.readyState === WebSocket.OPEN) {
        // Send seq as 4-byte header + wav bytes
        const wavBytes = new Uint8Array(reader.result);
        const pkt = new Uint8Array(4 + wavBytes.length);
        new DataView(pkt.buffer).setUint32(0, seq, true);
        pkt.set(wavBytes, 4);
        ws.send(pkt.buffer);
      } else {
        // WS closed mid-send — fall back to HTTP
        sendQueue.set(seq, { blob, retries: 0, nextRetryAt: 0 });
        if (!queueLoopRunning) startQueueLoop();
      }
    };
    reader.readAsArrayBuffer(blob);
    return;
  }

  // HTTP fallback
  sendQueue.set(seq, { blob, retries: 0, nextRetryAt: 0 });
  if (!queueLoopRunning) startQueueLoop();
}

// ── HTTP fallback queue ───────────────────────────────────────────────────────

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
      onFullyDrained();
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
      const backoff = Math.min(500 * Math.pow(2, entry.retries - 1), MAX_BACKOFF_MS);
      entry.nextRetryAt = Date.now() + backoff;
    }
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Drain + cleanup ───────────────────────────────────────────────────────────

function onFullyDrained() {
  closeWs();
  cleanup();
  chrome.runtime.sendMessage({ type: 'QUEUE_DRAINED' });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

chrome.runtime.sendMessage({ type: 'OFFSCREEN_READY' });

chrome.runtime.onMessage.addListener((msg, _sender) => {
  if (msg.type === 'INIT_CAPTURE') {
    serverUrl = msg.serverUrl || serverUrl;
    sessionId = msg.sessionId;
    sessionModel = msg.model || 'whisper';
    if (typeof msg.silenceThreshold === 'number') SILENCE_RMS_THRESHOLD = msg.silenceThreshold;
    useWebSocket = msg.transport !== 'http';
    connectWs();
    initCapture(msg.streamId, msg.includeMic).catch(e => {
      console.error('[offscreen] init error', e);
    });
  }
  if (msg.type === 'STOP_CAPTURE') {
    stopCapture();
  }
});

// ── Audio init ────────────────────────────────────────────────────────────────

async function initCapture(streamId, includeMic) {
  tabStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId },
    },
    video: false,
  });

  if (includeMic) {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
        video: false,
      });
    } catch {}
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
    processorOptions: {
      chunkFrames: 16000 * 4, // 4s chunks — matches Voxtral's ~3.5s inference time
      silenceThreshold: SILENCE_RMS_THRESHOLD,
    },
  });

  workletNode.port.onmessage = (e) => {
    if (e.data?.type === 'chunk') {
      const wavBlob = encodeWavPCM16(e.data.samples, 16000);
      enqueueChunk(wavBlob, e.data.rms ?? 1);
    }
    if (e.data?.type === 'flushed') {
      draining = true;
      if (sendQueue.size === 0 && !wsHasPending()) {
        onFullyDrained();
      } else if (!queueLoopRunning && sendQueue.size > 0) {
        startQueueLoop();
      }
      // If WS path is used and has pending, wait for ws.onclose or explicit drain signal
      if (wsReady) {
        // Send a close frame so server knows to flush
        try { ws?.close(1000, 'done'); } catch {}
      }
    }
  };

  sourceNode.connect(workletNode);

  audioEl = new Audio();
  audioEl.srcObject = tabStream;
  audioEl.play().catch(() => {});
}

function wsHasPending() {
  // We don't track individual WS sends, so conservatively return false
  // The WS onclose handler or HTTP fallback covers the remainder
  return false;
}

// ── Stop ──────────────────────────────────────────────────────────────────────

function stopCapture() {
  try { tabStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { micStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { if (audioEl) { audioEl.pause(); audioEl.srcObject = null; } } catch {}

  if (workletNode) {
    workletNode.port.postMessage({ type: 'flush' });
  } else {
    draining = true;
    if (sendQueue.size === 0) {
      onFullyDrained();
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

// ── WAV encoder ───────────────────────────────────────────────────────────────

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
