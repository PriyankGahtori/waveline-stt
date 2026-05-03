# Waveline — Real-Time Speech-to-Text for Browser Audio

> Local, private, zero-loss speech transcription for any browser tab. Chrome extension + FastAPI backend with Whisper and Voxtral model support, session management, audio file saving, and MCP server exposure.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Server Configuration](#server-configuration)
- [Chrome Extension Setup](#chrome-extension-setup)
- [Extension Settings Guide](#extension-settings-guide)
- [API Reference](#api-reference)
- [MCP Server](#mcp-server)
- [Output Files](#output-files)
- [Zero-Loss Design](#zero-loss-design)
- [Known Limitations](#known-limitations)

---

## Features

- **Real-time transcription** of browser tab audio (meetings, YouTube, any tab)
- **WebSocket streaming** — single persistent connection per session, results returned as soon as each chunk is transcribed
- **HTTP fallback** — automatically falls back to per-chunk HTTP POST if WebSocket is unavailable
- **Silence gate** — silent audio chunks are detected client-side and never sent, reducing latency and server load
- **1.5-second chunks** — transcription starts 1.5 s after speech, down from 4 s
- **Zero data loss** — retry queue with exponential backoff; chunks are never silently dropped
- **Dual model support** — switch between OpenAI Whisper and Mistral Voxtral from the extension popup
- **Session management** — each recording is isolated under a UUID; safe for concurrent multi-user use
- **Audio file saving** — every chunk saved as WAV; merged incrementally into a single file (only new chunks appended each time)
- **Transcript persistence** — full transcript written to disk in chronological order on session stop
- **Async I/O** — audio merge and transcript write run concurrently in a thread pool, never blocking the event loop
- **MCP server** — all session operations exposed as tools for Claude Code and other MCP clients
- **Live connection status** — popup shows whether backend is reachable and which models are loaded
- **Dead-chunk warning** — user notified if a chunk permanently fails after 8 retries
- **Microphone mixing** — optional mic capture merged with tab audio for in-person + remote scenarios

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Chrome Extension (Manifest V3)                                      │
│                                                                       │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  popup.js   │◄──►│ service_worker.js│◄──►│   offscreen.js    │   │
│  │             │    │                  │    │                    │   │
│  │ • Model     │    │ • Session UUID   │    │ • Web Audio API    │   │
│  │   selector  │    │ • /session/start │    │ • PCM worklet      │   │
│  │ • Transport │    │ • /session/stop  │    │ • WebSocket send   │   │
│  │   selector  │    │ • Message relay  │    │ • HTTP fallback     │   │
│  │ • Mic toggle│    │                  │    │ • Silence gate      │   │
│  │ • Silence   │    │                  │    │ • Retry queue       │   │
│  │   gate      │    │                  │    │ • WAV encoder       │   │
│  │ • Conn stat │    │                  │    │                    │   │
│  │ • Transcript│    │                  │    │                    │   │
│  └─────────────┘    └──────────────────┘    └────────┬───────────┘  │
│                                                       │              │
│                                              pcm-worklet.js          │
│                                              (AudioWorklet)          │
└───────────────────────────────────────────────────────┼─────────────┘
                                                        │ WS /ws/transcribe/{id}
                                                        │ or POST /transcribe
                                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  server.py  (FastAPI :8000 + FastMCP :8001)                          │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Model        │  │ Session      │  │ MCP Tools                │   │
│  │ Registry     │  │ Store        │  │                          │   │
│  │              │  │              │  │ • start_session          │   │
│  │ • Whisper    │  │ • chunks{}   │  │ • stop_session           │   │
│  │   (faster-   │  │ • audio_dir  │  │ • get_transcript         │   │
│  │   whisper)   │  │ • merged_path│  │ • get_audio_path         │   │
│  │ • Voxtral    │  │ • closed     │  │ • set_model              │   │
│  │   (mlx-audio)│  │              │  │                          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
│                                                                       │
│  IO thread pool (4 threads) — merge + transcript write off-loop      │
│                                                                       │
│  recordings/                                                          │
│  └── {session_id}/                                                   │
│      ├── chunks/chunk_00000.wav … chunk_NNNNN.wav                    │
│      ├── merged.wav                                                  │
│      └── transcript.txt                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Start**: popup reads settings (transport, mic, silence threshold) → sends `START_RECORDING` → service worker calls `POST /session/start` → gets `session_id` back
2. **Capture**: offscreen document creates Web Audio graph (tab + optional mic) → AudioWorklet buffers 1.5-second PCM chunks and computes RMS
3. **Silence gate**: chunks with RMS below the threshold are discarded immediately — no network round trip
4. **Send (WebSocket)**: if transport = `ws`, chunk is sent as a binary frame (4-byte seq + WAV bytes) over a persistent WebSocket connection; result JSON arrives back immediately
5. **Send (HTTP fallback)**: if WebSocket is unavailable or transport = `http`, chunk goes via `POST /transcribe`; up to 4 in-flight simultaneously
6. **Transcribe**: server saves chunk WAV, runs Whisper/Voxtral, returns transcript text; Whisper uses VAD + no timestamp mode for speed
7. **Stop**: worklet flushes partial chunk → queue fully drains → service worker calls `POST /session/stop` → server concurrently merges WAV and writes transcript in IO thread pool → popup Start button re-enabled

---

## Project Structure

```
stt_plugin/
├── server.py                        # Unified backend (Whisper + Voxtral + MCP)
├── app.py                           # Legacy: Whisper-only server
├── server_voxtral_sst_v2.py         # Legacy: Voxtral-only server
├── requirements.txt
├── README.md
├── recordings/                      # Created at runtime
│   └── {session_id}/
│       ├── chunks/
│       ├── merged.wav
│       └── transcript.txt
└── waveline_extension/
    ├── manifest.json
    ├── icons/                       # icon16.png, icon48.png, icon128.png
    ├── popup.html                   # Extension UI
    ├── popup.js                     # UI logic + settings wiring
    ├── style.css
    ├── service_worker.js            # Background: session lifecycle, message relay
    ├── offscreen.html               # Offscreen document host
    ├── offscreen.js                 # Audio capture, WebSocket/HTTP send, silence gate
    └── pcm-worklet.js               # AudioWorklet: 1.5s PCM buffering + RMS
```

---

## Requirements

- **Python 3.13**
- **macOS** (Voxtral uses Apple MLX framework; Whisper works on any OS)
- **Google Chrome** 116+ (Manifest V3, Offscreen Documents API)
- ~4 GB free memory for Voxtral (4-bit quantized); ~1.6 GB disk/cache for the default Hindi Whisper model

---

## Installation

```bash
git clone <repo>
cd stt_plugin

# Create virtual environment with system Python 3.13
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv

# Install all dependencies
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs: `fastapi`, `uvicorn`, `faster-whisper`, `mlx-audio`, `mlx-whisper`, `mcp[cli]`, `python-multipart`.

---

## Running the Server

### Default models from `.env`

```bash
.venv/bin/python3 server.py
```

The server loads `.env` from the same directory as `server.py`. Shell environment variables override values from `.env`.
For a new checkout, copy `.env.example` to `.env` and edit the model flags there.

### Whisper only (faster startup)

```bash
LOAD_VOXTRAL=false LOAD_VAANI=false .venv/bin/python3 server.py
```

### Whisper with auto language detection

```bash
WHISPER_LANGUAGE= LOAD_VOXTRAL=false LOAD_VAANI=false .venv/bin/python3 server.py
```

### Whisper with a different model

```bash
WHISPER_MODEL=medium \
WHISPER_LANGUAGE=en \
LOAD_VOXTRAL=false \
LOAD_VAANI=false \
.venv/bin/python3 server.py
```

### Voxtral only (~4 GB RAM, Apple Silicon only)

```bash
LOAD_WHISPER=false LOAD_VAANI=false .venv/bin/python3 server.py
```

### Vaani only (Hindi, Apple Silicon only)

```bash
LOAD_WHISPER=false LOAD_VOXTRAL=false .venv/bin/python3 server.py
```

### With audio merge interval

```bash
MERGE_INTERVAL_SECS=30 .venv/bin/python3 server.py
```

### Verify server is ready

```bash
curl http://localhost:8000/health
# {"ok":true,"models":{"whisper":true,"voxtral":true,"vaani":true}}
```

> **Note:** The default Whisper model `collabora/faster-whisper-medium-hindi` downloads ~1.54 GB on first run. Voxtral downloads ~2.5 GB on first run; subsequent startups take ~5 s from cache.

---

## Server Configuration

Settings are read from a root `.env` file first, then from shell environment variables. Shell values take precedence.

| Variable | Default | Description |
|---|---|---|
| `LOAD_WHISPER` | `true` | Load Whisper model on startup |
| `LOAD_VOXTRAL` | `true` | Load Voxtral model on startup |
| `LOAD_VAANI` | `true` | Load Vaani model on startup |
| `WHISPER_MODEL` | `collabora/faster-whisper-medium-hindi` | Whisper model ID or size (`medium`, `large-v3`, HuggingFace ID) |
| `WHISPER_LANGUAGE` | `hi` | Language code (`hi`, `en`); set empty for auto-detect |
| `COMPUTE_TYPE` | `float32` | Whisper compute type: `float32` / `int8` |
| `BEAM_SIZE` | `1` | Whisper beam width (higher = more accurate, slower) |
| `VOXTRAL_MODEL` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | Voxtral model path |
| `VAANI_MODEL` | `ARTPARK-IISc/whisper-medium-vaani-hindi` | Vaani model path |
| `MAX_TOKENS` | `4096` | Maximum tokens for Voxtral output |
| `TEMPERATURE` | `0.0` | Voxtral sampling temperature (0 = deterministic) |
| `RECORDINGS_DIR` | `./recordings` | Root directory for session audio and transcripts |
| `MERGE_INTERVAL_SECS` | `0` | Auto-merge chunks every N seconds; `0` = merge on stop only |
| `MAX_UPLOAD_MB` | `100` | Maximum audio chunk upload size in MB |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | FastAPI HTTP port |
| `MCP_PORT` | `8001` | MCP SSE server port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `LOG_DIR` | `.` | Directory for rotating log files (daily, 30 days) |

**Example with multiple overrides:**

```bash
WHISPER_MODEL=medium \
WHISPER_LANGUAGE=en \
RECORDINGS_DIR=~/my-recordings \
MERGE_INTERVAL_SECS=60 \
LOG_DIR=~/logs \
.venv/bin/python3 server.py
```

---

## Chrome Extension Setup

### Install

1. Open `chrome://extensions` in Chrome
2. Enable **Developer mode** (toggle, top-right)
3. Click **Load unpacked**
4. Select the `waveline_extension/` folder
5. The **Waveline** icon appears in your toolbar — pin it via the puzzle-piece menu

### Quick start

1. Confirm the backend is running: `curl localhost:8000/health`
2. Open any tab with audio — YouTube, a meeting, a podcast
3. Click the **Waveline** icon
4. Select your **model** and configure **Settings** if needed (see below)
5. Click **Record** — transcript lines appear in real time
6. Click **Stop** — the button stays disabled until all audio is uploaded, then re-enables
7. Use **Export** to download the transcript as `.txt`

---

## Extension Settings Guide

Open the **Settings** section at the bottom of the popup to access all options. Settings are saved automatically and apply to the next recording.

---

### Backend URL

**Default:** `http://localhost:8000`

The address of your running `server.py`. Change this if:
- The server is on a different machine (e.g. `http://192.168.1.10:8000`)
- You changed the `PORT` environment variable
- You're routing through a reverse proxy

After editing, click **Save** — the connection indicator updates immediately.

---

### Transport

**Default:** `WebSocket (low latency, auto-fallback to HTTP)`

Controls how audio chunks are sent to the server.

| Option | When to use |
|---|---|
| **WebSocket** | Default. One persistent connection per session. Lowest latency — result arrives as soon as each chunk is transcribed. Automatically falls back to HTTP if the WebSocket connection fails. |
| **HTTP only** | Use if you're behind a corporate proxy or nginx that doesn't support WebSocket upgrades. Also useful for debugging — HTTP requests are visible in Chrome DevTools → Network. |

> **Tip:** If you see frequent reconnection attempts in the server logs (`WS session=... closed with error`), switch to HTTP only.

---

### Include Microphone

**Default:** off

When enabled, your microphone audio is mixed with the tab audio before transcription. Useful for:
- In-person meetings where you speak and others are on-screen
- Dictating notes while a video plays

Chrome will prompt for microphone permission the first time this is enabled. If you deny it, recording continues with tab audio only.

---

### Silence Gate

**Default:** `0.003` — Range: `0.000` to `0.020`

Controls the RMS (volume) threshold below which a chunk is considered silence and **not sent to the server**. The current value is shown next to the slider label.

| Value | Effect |
|---|---|
| `0.000` | Gate off — every chunk is sent, including silence |
| `0.003` | Default — skips near-silent periods (background hum, idle mic) |
| `0.008–0.012` | Aggressive — only sends clearly audible speech; good for noisy rooms |
| `0.020` | Maximum — may skip quiet speech; not recommended |

**When to raise it:**
- Server logs show many empty-text transcription results
- You're in a noisy environment and silent chunks are slowing down the queue

**When to lower it:**
- Soft-spoken audio is being skipped
- You notice gaps in the transcript that correspond to quieter speech

---

### Popup UI reference

| Element | Description |
|---|---|
| **Record** | Start a new recording session. Disabled while the previous session is still draining. |
| **Stop** | Stop recording. Shows "Finishing…" while remaining chunks upload; re-enables Record when complete. |
| **Model selector** | Choose Whisper or Voxtral. Disabled during recording. Unavailable models are greyed out. |
| **Connection status** | Live indicator — Connected / Unreachable — updates on URL save and popup open. |
| **Transcript area** | Live transcript; editable; persists across popup close/reopen. |
| **Export** | Download transcript as `.txt`. |
| **Clear** | Clear transcript from UI and local storage. |
| **Settings › Backend URL** | Server address with save + live connection check. |
| **Settings › Transport** | WebSocket (default) or HTTP only. |
| **Settings › Include microphone** | Mix mic audio into the recording. |
| **Settings › Silence gate** | RMS threshold slider for skipping silent chunks. |

---

## API Reference

### `POST /session/start`

Start a new recording session.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | `whisper` | `whisper` or `voxtral` |
| `user_tag` | string | `""` | Optional identifier for multi-user setups |

**Response:**
```json
{ "session_id": "uuid", "model": "whisper" }
```

---

### `POST /session/stop`

Finalize a session. Concurrently merges all audio chunks into `merged.wav` and writes `transcript.txt` in the IO thread pool.

**Form fields:** `session_id` (string, required)

**Response:**
```json
{
  "session_id": "uuid",
  "merged_path": "recordings/uuid/merged.wav",
  "transcript_path": "recordings/uuid/transcript.txt",
  "chunk_count": 12
}
```

---

### `POST /transcribe`

Upload a single audio chunk for transcription. **Idempotent** — sending the same `seq` twice returns the cached result without re-transcribing.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Session UUID |
| `seq` | integer | Chunk sequence number (0-indexed, monotonic) |
| `model` | string | `whisper` or `voxtral` |
| `audio` | file | WAV audio chunk |

**Response:**
```json
{ "ok": true, "seq": 0, "text": "transcribed text here" }
```

**Error codes:** `404` session not found · `409` session closed · `503` model not loaded · `413` file too large

---

### `WebSocket /ws/transcribe/{session_id}?model=whisper`

Streaming transcription over a persistent WebSocket connection.

**Client → server:** binary frame = 4-byte little-endian sequence number + WAV bytes

**Server → client:** JSON frame per chunk
```json
{ "seq": 0, "text": "transcribed text here" }
```

The extension uses this automatically when Transport = WebSocket. Falls back to `POST /transcribe` if the connection fails.

---

### `GET /transcript/{session_id}`

Retrieve the full transcript in sequence order.

**Response:**
```json
{ "session_id": "uuid", "text": "line 1\nline 2\n...", "chunk_count": 12 }
```

---

### `GET /audio/{session_id}`

Download the merged WAV file. Triggers an on-demand merge if not yet done.

**Response:** `audio/wav` file download

---

### `GET /health`

Check server status and loaded models.

**Response:**
```json
{ "ok": true, "models": { "whisper": true, "voxtral": false } }
```

Returns `503` if no models are loaded.

---

## MCP Server

Runs on port 8001 alongside FastAPI. Exposes session operations as tools for Claude Code and other MCP clients.

### Available Tools

| Tool | Arguments | Description |
|---|---|---|
| `start_session` | `model`, `user_tag` | Start a new STT session |
| `stop_session` | `session_id` | Stop session, merge audio, write transcript |
| `get_transcript` | `session_id` | Get full transcript text |
| `get_audio_path` | `session_id` | Get path to merged WAV file |
| `set_model` | `session_id`, `model` | Switch model for an active session |

### Connect from Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "stt": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

---

## Output Files

Each session creates an isolated directory:

```
recordings/
└── {session_id}/              # UUID, e.g. c7ea4cdf-dd2a-499b-bff2-1c7b2a27e8b7
    ├── chunks/
    │   ├── chunk_00000.wav    # 1.5-second PCM WAV, 16kHz mono 16-bit
    │   ├── chunk_00001.wav
    │   └── ...
    ├── merged.wav             # All chunks merged in sequence order (incremental append)
    └── transcript.txt         # Full transcript, one chunk per line
```

- Chunks are saved immediately on receipt — audio is preserved even if transcription fails
- `merged.wav` is built incrementally — only new chunks are appended each merge cycle, not rewritten from scratch
- Merge and transcript write run concurrently in a background thread pool
- `merged.wav` uses Python's stdlib `wave` module — no ffmpeg dependency
- Multiple concurrent sessions never share directories

---

## Zero-Loss Design

### Silence gate (client-side)

The AudioWorklet computes the RMS of each 1.5-second chunk. If RMS < threshold, the chunk is discarded before any network I/O — no server round trip, no wasted inference.

### Send queue with WebSocket + HTTP fallback

```
chunk ready (RMS ≥ threshold)
    │
    ▼
WebSocket open?
    ├─ yes → send binary frame → result JSON back immediately → done
    │
    └─ no  → HTTP queue (up to 4 parallel sends)
                 │
                 ├─ success → remove from queue
                 │
                 └─ failure → retry with backoff (500ms → 1s → 2s → … → 16s max)
                                   │
                               retries = 8
                                   │
                                   ▼
                             mark DEAD, notify popup
                             (never silently dropped)
```

### Drain before close

When Stop is clicked:
1. AudioWorklet flushes its partial buffer
2. The send queue (HTTP path) fully drains
3. WebSocket sends a close frame so the server can flush
4. Service worker calls `POST /session/stop`
5. Popup Start button re-enables

The Start button is disabled during this window — a new recording cannot begin until the previous session is committed.

### Idempotent server

Each chunk carries a `seq` number. If a chunk is retried after a successful server write (e.g. the network response was lost), the server returns the cached result without re-transcribing or re-saving.

---

## Known Limitations

| Limitation | Notes |
|---|---|
| **Voxtral: Apple Silicon only** | MLX framework requires Apple M-series hardware |
| **Tab audio requires user gesture** | Chrome's `tabCapture` API requires the extension popup to be opened by a real click — cannot be automated via CDP |
| **No authentication** | Server accepts all requests; intended for local/team use behind a firewall |
| **In-memory session store** | Sessions are lost on server restart; recordings on disk remain intact |
| **Whisper language is fixed per server process** | Change `WHISPER_LANGUAGE` and restart the server to switch; set it empty for auto-detect |
| **MCP port conflict** | If port 8001 is in use, MCP server fails silently; FastAPI still runs |
| **WebSocket behind proxies** | Some corporate proxies strip the `Upgrade` header — use Transport = HTTP only in that case |

---

## Legacy Servers

`app.py` and `server_voxtral_sst_v2.py` are kept for reference but superseded by `server.py`.

| File | Model | Sessions | Audio saving | MCP |
|---|---|---|---|---|
| `server.py` | Whisper + Voxtral | ✅ | ✅ | ✅ |
| `app.py` | Whisper only | ✗ | ✗ | ✗ |
| `server_voxtral_sst_v2.py` | Voxtral only | ✗ | ✗ | ✗ |
