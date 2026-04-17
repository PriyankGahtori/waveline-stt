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
- [Configuration](#configuration)
- [Chrome Extension Setup](#chrome-extension-setup)
- [API Reference](#api-reference)
- [MCP Server](#mcp-server)
- [Output Files](#output-files)
- [Zero-Loss Design](#zero-loss-design)
- [Tests Performed](#tests-performed)
- [Known Limitations](#known-limitations)

---

## Features

- **Real-time transcription** of browser tab audio (meetings, YouTube, any tab)
- **Zero data loss** — retry queue with exponential backoff; chunks are never silently dropped
- **Dual model support** — switch between OpenAI Whisper and Mistral Voxtral from the extension popup
- **Session management** — each recording is isolated under a UUID; safe for concurrent multi-user use
- **Audio file saving** — every chunk saved as WAV; merged into a single file on stop or on a configurable interval
- **Transcript persistence** — full transcript written to disk in chronological order on session stop
- **MCP server** — all session operations exposed as tools for Claude Code and other MCP clients
- **Live connection status** — popup shows whether backend is reachable and which models are loaded
- **Dead-chunk warning** — user notified if a chunk permanently fails after 10 retries
- **Microphone mixing** — optional mic capture merged with tab audio for in-person + remote scenarios

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Chrome Extension (Manifest V3)                                  │
│                                                                   │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │  popup.js   │◄──►│ service_worker.js│◄──►│  offscreen.js   │ │
│  │             │    │                  │    │                  │ │
│  │ • Model     │    │ • Session UUID   │    │ • Web Audio API  │ │
│  │   selector  │    │ • /session/start │    │ • PCM worklet    │ │
│  │ • Conn stat │    │ • /session/stop  │    │ • Retry queue    │ │
│  │ • Transcript│    │ • Message relay  │    │ • WAV encoder    │ │
│  └─────────────┘    └──────────────────┘    └────────┬────────┘ │
│                                                       │          │
│                                              pcm-worklet.js      │
│                                              (AudioWorklet)      │
└───────────────────────────────────────────────────────┼─────────┘
                                                        │ POST /transcribe
                                                        │ (session_id, seq, model, audio)
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  server.py  (FastAPI :8000 + FastMCP :8001)                      │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Model        │  │ Session      │  │ MCP Tools              │ │
│  │ Registry     │  │ Store        │  │                        │ │
│  │              │  │              │  │ • start_session        │ │
│  │ • Whisper    │  │ • chunks{}   │  │ • stop_session         │ │
│  │   (faster-   │  │ • audio_dir  │  │ • get_transcript       │ │
│  │   whisper)   │  │ • merged_path│  │ • get_audio_path       │ │
│  │ • Voxtral    │  │ • closed     │  │ • set_model            │ │
│  │   (mlx-audio)│  │              │  │                        │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                   │
│  recordings/                                                      │
│  └── {session_id}/                                               │
│      ├── chunks/chunk_00000.wav … chunk_NNNNN.wav                │
│      ├── merged.wav                                              │
│      └── transcript.txt                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Start**: popup sends `START_RECORDING` → service worker generates UUID → calls `POST /session/start` → gets `session_id` back
2. **Capture**: offscreen document creates Web Audio graph (tab audio + optional mic) → AudioWorklet buffers 4-second PCM chunks
3. **Send**: each chunk is WAV-encoded and placed in an in-memory retry queue with a sequence number
4. **Transcribe**: queue processor sends chunks to `POST /transcribe` with `session_id`, `seq`, and `model` → server saves chunk WAV and returns transcript text
5. **Stop**: worklet flushes partial chunk → queue drains to empty → service worker calls `POST /session/stop` → server merges all WAVs and writes `transcript.txt`

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
    ├── popup.js                     # UI logic: model selector, conn status
    ├── style.css
    ├── service_worker.js            # Background: session lifecycle, message relay
    ├── offscreen.html               # Offscreen document host
    ├── offscreen.js                 # Audio capture + retry send queue
    └── pcm-worklet.js               # AudioWorklet: real-time PCM buffering
```

---

## Requirements

- **Python 3.13**
- **macOS** (Voxtral uses Apple MLX framework; Whisper works on any OS)
- **Google Chrome** 116+ (Manifest V3, Offscreen Documents API)
- ~4 GB free memory for Voxtral (4-bit quantized); ~500 MB for Whisper medium

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

`requirements.txt` installs: `fastapi`, `uvicorn`, `faster-whisper`, `mlx-audio`, `mcp[cli]`, `python-multipart`.

---

## Running the Server

### Both models (recommended)

```bash
.venv/bin/python3 server.py
```

### Whisper only (faster startup, ~500 MB RAM)

```bash
LOAD_VOXTRAL=false .venv/bin/python3 server.py
```

### Voxtral only (~4 GB RAM, Apple Silicon only)

```bash
LOAD_WHISPER=false .venv/bin/python3 server.py
```

### With audio merge interval

```bash
MERGE_INTERVAL_SECS=30 .venv/bin/python3 server.py
```

### Verify server is ready

```bash
curl http://localhost:8000/health
# {"ok":true,"models":{"whisper":true,"voxtral":true}}
```

> **Note:** Voxtral loads in ~5 seconds after the first run (model is cached by HuggingFace). Initial download is ~2.5 GB.

---

## Configuration

All settings are controlled via environment variables. No config files needed.

| Variable | Default | Description |
|---|---|---|
| `LOAD_WHISPER` | `true` | Load Whisper model on startup |
| `LOAD_VOXTRAL` | `true` | Load Voxtral model on startup |
| `WHISPER_MODEL` | `medium` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large` |
| `COMPUTE_TYPE` | `float32` | Whisper compute type: `float32` / `int8` |
| `BEAM_SIZE` | `1` | Whisper beam search width (higher = more accurate, slower) |
| `VOXTRAL_MODEL` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | Voxtral model path (HuggingFace ID or local path) |
| `MAX_TOKENS` | `4096` | Maximum tokens for Voxtral output |
| `TEMPERATURE` | `0.0` | Voxtral sampling temperature (0 = deterministic) |
| `RECORDINGS_DIR` | `./recordings` | Root directory for session audio and transcripts |
| `MERGE_INTERVAL_SECS` | `0` | Auto-merge chunks every N seconds; `0` = merge on stop only |
| `MAX_UPLOAD_MB` | `100` | Maximum audio chunk upload size in MB |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | FastAPI HTTP port |
| `MCP_PORT` | `8001` | MCP SSE server port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `LOG_DIR` | `.` | Directory for rotating log files (daily rotation, 30 days) |

**Example with multiple overrides:**

```bash
WHISPER_MODEL=large \
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
5. The **Waveline** icon appears in your toolbar (pin it via the puzzle piece menu)

### Usage

1. Make sure the backend server is running (`curl localhost:8000/health`)
2. Open any tab with audio — YouTube, a meeting, a podcast
3. Click the **Waveline** icon in the toolbar
4. Select your model: **Whisper** (general purpose) or **Voxtral** (optimized for Apple Silicon)
5. Optionally enable **Mic** to also capture your microphone
6. Click **Record** — transcript lines appear in real time
7. Click **Stop** — audio is merged and saved; transcript written to disk
8. Use **Export** to download the transcript as a `.txt` file

### Popup UI

| Element | Description |
|---|---|
| **Record / Stop** | Start or stop a recording session |
| **Mic toggle** | Mix microphone audio with tab audio |
| **Model selector** | Choose Whisper or Voxtral (disabled during recording) |
| **Connection status** | Live indicator: Connected / Unreachable + loaded models |
| **Transcript area** | Live transcript; editable; persists across popup close/reopen |
| **Export** | Download transcript as `.txt` |
| **Clear** | Clear transcript from UI and storage |
| **Settings** | Configure backend URL (default: `http://localhost:8000`) |

---

## API Reference

### `POST /session/start`

Start a new recording session.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | `whisper` | Model to use: `whisper` or `voxtral` |
| `user_tag` | string | `""` | Optional identifier for multi-user setups |

**Response:**
```json
{ "session_id": "uuid", "model": "whisper" }
```

---

### `POST /session/stop`

Finalize a session: merges all audio chunks into `merged.wav` and writes `transcript.txt`.

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

Upload a single audio chunk for transcription. **Idempotent** — sending the same `seq` twice returns the cached result.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Session UUID from `/session/start` |
| `seq` | integer | Chunk sequence number (0-indexed, monotonic) |
| `model` | string | `whisper` or `voxtral` |
| `audio` | file | WAV audio chunk |

**Response:**
```json
{ "ok": true, "seq": 0, "text": "transcribed text here" }
```

**Error codes:** `404` session not found · `409` session closed · `503` model not loaded · `413` file too large

---

### `GET /transcript/{session_id}`

Retrieve the full transcript in sequence order.

**Response:**
```json
{ "session_id": "uuid", "text": "line 1\nline 2\n...", "chunk_count": 12 }
```

---

### `GET /audio/{session_id}`

Download the merged WAV file for a session. Triggers an on-demand merge if not yet merged.

**Response:** `audio/wav` file download

---

### `GET /health`

Check server status and loaded models.

**Response:**
```json
{ "ok": true, "models": { "whisper": true, "voxtral": true } }
```

Returns `503` if no models are loaded.

---

## MCP Server

The MCP server runs on port 8001 alongside FastAPI. It exposes session operations as tools for use with Claude Code and other MCP clients.

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

Each recording session creates an isolated directory:

```
recordings/
└── {session_id}/              # UUID, e.g. c7ea4cdf-dd2a-499b-bff2-1c7b2a27e8b7
    ├── chunks/
    │   ├── chunk_00000.wav    # 4-second PCM WAV, 16kHz mono 16-bit
    │   ├── chunk_00001.wav
    │   └── ...
    ├── merged.wav             # All chunks merged in sequence order
    └── transcript.txt         # Full transcript, one chunk per line
```

- Chunks are saved immediately on receipt — audio is preserved even if transcription fails
- `merged.wav` is created using Python's stdlib `wave` module — no ffmpeg dependency
- Multiple concurrent sessions never share directories — safe for team use

---

## Zero-Loss Design

The original system silently dropped audio chunks on any network or server error. The redesigned system guarantees no silent data loss:

### Plugin-side retry queue

```
chunk ready
    │
    ▼
enqueue(seq, blob)  ←──────────────────────┐
    │                                       │
    ▼                                       │
processQueue() [every 500ms]                │
    │                                       │
    ├─ send to /transcribe ──► success ──► remove from queue
    │                                       │
    └─► failure ──► retries < 10 ──────────┘
                        │
                    retries = 10
                        │
                        ▼
                  mark DEAD, notify popup
                  (never silently dropped)
```

**Retry schedule:** 1s → 2s → 4s → 8s → … → 30s (max), exponential backoff

**Drain before close:** when Stop is clicked, the audio worklet flushes its partial buffer, then the queue fully drains before `POST /session/stop` is called. The brittle 800ms timeout from the original design is gone.

**Idempotent server:** each chunk carries a `seq` number. If a chunk is retried after a successful server write (e.g. the network response was lost), the server returns the cached result without re-transcribing.

**Crash recovery:** on unexpected page close, the pending queue metadata is serialized to `chrome.storage.local` for inspection on next open.

---

## Tests Performed

All tests were run against `server.py` using the `.venv` Python environment.

### 1. Health check

```bash
curl http://localhost:8000/health
# {"ok":true,"models":{"whisper":true,"voxtral":true}}
```
✅ Both models loaded and reported correctly.

### 2. Whisper — end-to-end session

- Created session with `model=whisper`
- Sent a 2-second 440Hz sine wave WAV as a dummy audio chunk (`seq=0`)
- Retrieved transcript (empty — no speech, correct)
- Called `/session/stop`
- Verified `merged.wav` and `transcript.txt` written to `recordings/{session_id}/`

✅ Session lifecycle, audio saving, and file merge all working.

### 3. Voxtral — end-to-end session

Same test as above with `model=voxtral`.

✅ Voxtral loaded and processed chunk correctly (~5s startup from cache).

### 4. Idempotency (retry simulation)

- Sent the same chunk twice with identical `seq=0`
- Verified second response returned cached text without re-transcribing

✅ Idempotent — safe for plugin retry queue.

### 5. Audio download endpoint

- Called `GET /audio/{session_id}` after session stop
- Verified HTTP 200, `Content-Type: audio/wav`, correct byte size (64 044 bytes for 2s mono 16kHz)

✅ Audio download working.

### 6. MCP server startup

- Started server with both models
- Confirmed MCP server started on port 8001
- Confirmed FastAPI running on port 8000 simultaneously

✅ Both servers co-exist in the same process.

### 7. Multi-session isolation

- Started two sessions back-to-back
- Verified each got a unique UUID and separate `recordings/` directory

✅ Sessions fully isolated.

---

## Known Limitations

| Limitation | Notes |
|---|---|
| **Voxtral: Apple Silicon only** | MLX framework requires Apple M-series hardware |
| **Tab audio requires user gesture** | Chrome's `tabCapture` API requires the extension popup to be opened by a real click — cannot be automated via CDP |
| **No authentication** | Server accepts all requests; intended for local/team use behind a firewall |
| **In-memory session store** | Sessions are lost on server restart; recordings on disk remain intact |
| **English only (Whisper)** | `language="en"` is hardcoded for Whisper; Voxtral is multilingual |
| **MCP port conflict** | If port 8001 is in use, MCP server fails silently; FastAPI still runs |

---

## Legacy Servers

`app.py` and `server_voxtral_sst_v2.py` are kept for backward compatibility but are superseded by `server.py`.

| File | Model | Sessions | Audio saving | MCP |
|---|---|---|---|---|
| `server.py` | Whisper + Voxtral | ✅ | ✅ | ✅ |
| `app.py` | Whisper only | ✗ | ✗ | ✗ |
| `server_voxtral_sst_v2.py` | Voxtral only | ✗ | ✗ | ✗ |
