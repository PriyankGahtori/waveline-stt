"""
server.py — unified STT backend supporting Whisper and Voxtral models.

Environment variables:
  LOAD_WHISPER       true|false  (default: true)
  LOAD_VOXTRAL       true|false  (default: true)
  WHISPER_MODEL      model size  (default: medium)
  COMPUTE_TYPE       float32     (default: float32)
  BEAM_SIZE          int         (default: 1)
  VOXTRAL_MODEL      model path  (default: mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit)
  MAX_TOKENS         int         (default: 4096)
  TEMPERATURE        float       (default: 0.0)
  RECORDINGS_DIR     path        (default: ./recordings)
  MERGE_INTERVAL_SECS int        (default: 0 = disabled, merge on stop only)
  MAX_UPLOAD_MB      int         (default: 100)
  HOST               str         (default: 0.0.0.0)
  PORT               int         (default: 8000)
  MCP_PORT           int         (default: 8001)
  CORS_ORIGINS       str         (default: *)
  LOG_DIR            path        (default: .)
"""

import asyncio
import concurrent.futures
import logging
import logging.handlers
import os
import time
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR = os.getenv("LOG_DIR", ".")
LOG_PATH = os.path.join(LOG_DIR, "transcriptions.log")

_fh = logging.handlers.TimedRotatingFileHandler(
    LOG_PATH, when="midnight", interval=1, backupCount=30, encoding="utf-8", utc=True
)
_fh.suffix = "%Y-%m-%d"
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"))
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"))

logger = logging.getLogger("stt_server")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.addHandler(_ch)

# ── Config ────────────────────────────────────────────────────────────────────

_LOAD_WHISPER = os.getenv("LOAD_WHISPER", "true").lower() == "true"
_LOAD_VOXTRAL = os.getenv("LOAD_VOXTRAL", "true").lower() == "true"
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
_COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")
_BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))
_VOXTRAL_MODEL = os.getenv("VOXTRAL_MODEL", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
_RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", "./recordings"))
_MERGE_INTERVAL_SECS = int(os.getenv("MERGE_INTERVAL_SECS", "0"))
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
_HOST = os.getenv("HOST", "0.0.0.0")
_PORT = int(os.getenv("PORT", "8000"))
_MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Model registry ────────────────────────────────────────────────────────────

models: dict = {"whisper": None, "voxtral": None}


def _load_models():
    if _LOAD_WHISPER:
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model: %s", _WHISPER_MODEL)
            models["whisper"] = WhisperModel(_WHISPER_MODEL, compute_type=_COMPUTE_TYPE)
            logger.info("Whisper ready")
        except Exception as e:
            logger.error("Failed to load Whisper: %s", e)

    if _LOAD_VOXTRAL:
        try:
            from mlx_audio.stt.utils import load as voxtral_load
            logger.info("Loading Voxtral model: %s", _VOXTRAL_MODEL)
            models["voxtral"] = voxtral_load(_VOXTRAL_MODEL)
            logger.info("Voxtral ready")
        except Exception as e:
            logger.error("Failed to load Voxtral: %s", e)


def _transcribe_whisper(audio_path: str) -> str:
    m = models["whisper"]
    segments, _ = m.transcribe(audio_path, beam_size=_BEAM_SIZE, vad_filter=True, language="en")
    return " ".join(s.text.strip() for s in segments if s.text.strip())


def _transcribe_voxtral(audio_path: str) -> str:
    m = models["voxtral"]
    kwargs = {"max_tokens": _MAX_TOKENS, "temperature": _TEMPERATURE}
    result = m.generate(audio_path, **kwargs)
    return result.text.strip() if hasattr(result, "text") else str(result).strip()


# Top-level function for subprocess execution (MLX is not thread-safe; must run
# in a separate process to avoid segfaults when called from a thread pool).
def _voxtral_subprocess(audio_path: str, voxtral_model_id: str, max_tokens: int, temperature: float) -> str:
    from mlx_audio.stt.utils import load as voxtral_load
    m = voxtral_load(voxtral_model_id)
    result = m.generate(audio_path, max_tokens=max_tokens, temperature=temperature)
    return result.text.strip() if hasattr(result, "text") else str(result).strip()


_voxtral_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)


def _do_transcribe(audio_path: str, model_name: str) -> str:
    if model_name == "whisper":
        return _transcribe_whisper(audio_path)
    elif model_name == "voxtral":
        # Run in dedicated process — MLX segfaults when called from threads
        future = _voxtral_executor.submit(
            _voxtral_subprocess, audio_path, _VOXTRAL_MODEL, _MAX_TOKENS, _TEMPERATURE
        )
        return future.result()
    raise ValueError(f"Unknown model: {model_name}")


# ── Session store ─────────────────────────────────────────────────────────────

sessions: dict = {}
# sessions[session_id] = {
#   "model": str,
#   "chunks": {seq: text},
#   "audio_dir": Path,
#   "merged_path": Path | None,
#   "transcript_path": Path | None,
#   "started_at": datetime,
#   "closed": bool,
#   "last_merged_seq": int,
#   "user_tag": str,
# }


def _get_session(session_id: str) -> dict:
    s = sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return s


def _create_session(model_name: str, user_tag: str = "") -> str:
    session_id = str(uuid.uuid4())
    audio_dir = _RECORDINGS_DIR / session_id / "chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sessions[session_id] = {
        "model": model_name,
        "chunks": {},
        "audio_dir": audio_dir,
        "merged_path": None,
        "transcript_path": None,
        "started_at": datetime.now(timezone.utc),
        "closed": False,
        "last_merged_seq": -1,
        "user_tag": user_tag,
    }
    logger.info("Session started: %s model=%s user=%s", session_id, model_name, user_tag)
    return session_id


def _merge_audio(session_id: str, force_all: bool = False) -> Optional[Path]:
    """Merge chunk WAVs in seq order into merged.wav. Returns merged path."""
    s = sessions[session_id]
    chunks_dir = s["audio_dir"]
    session_dir = chunks_dir.parent

    chunk_files = sorted(chunks_dir.glob("chunk_*.wav"), key=lambda p: int(p.stem.split("_")[1]))
    if not chunk_files:
        return None

    merged_path = session_dir / "merged.wav"

    with wave.open(str(merged_path), "wb") as out_wav:
        params_set = False
        for cf in chunk_files:
            try:
                with wave.open(str(cf), "rb") as in_wav:
                    if not params_set:
                        out_wav.setparams(in_wav.getparams())
                        params_set = True
                    out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
            except Exception as e:
                logger.warning("Skipping corrupt chunk %s: %s", cf.name, e)

    s["merged_path"] = merged_path
    logger.info("Merged audio for session %s → %s", session_id, merged_path)
    return merged_path


def _write_transcript(session_id: str) -> Path:
    s = sessions[session_id]
    session_dir = s["audio_dir"].parent
    transcript_path = session_dir / "transcript.txt"

    # Write chunks in seq order
    ordered = sorted(s["chunks"].items())
    text = "\n".join(t for _, t in ordered if t)
    transcript_path.write_text(text, encoding="utf-8")

    s["transcript_path"] = transcript_path
    return transcript_path


# ── Lifespan ──────────────────────────────────────────────────────────────────

_merge_task: Optional[asyncio.Task] = None


async def _periodic_merge():
    while True:
        await asyncio.sleep(_MERGE_INTERVAL_SECS)
        for sid, s in list(sessions.items()):
            if not s["closed"]:
                try:
                    _merge_audio(sid)
                except Exception as e:
                    logger.warning("Periodic merge failed for %s: %s", sid, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _merge_task
    _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    _load_models()
    if _MERGE_INTERVAL_SECS > 0:
        _merge_task = asyncio.create_task(_periodic_merge())
    yield
    if _merge_task:
        _merge_task.cancel()
    logger.info("Server shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="STT Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed)
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

class SessionStartOut(BaseModel):
    session_id: str
    model: str


@app.post("/session/start", response_model=SessionStartOut)
async def session_start(model: str = Form("whisper"), user_tag: str = Form("")):
    if model not in ("whisper", "voxtral"):
        raise HTTPException(status_code=400, detail="model must be 'whisper' or 'voxtral'")
    if models[model] is None:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not loaded")
    session_id = _create_session(model, user_tag)
    return {"session_id": session_id, "model": model}


class SessionStopOut(BaseModel):
    session_id: str
    merged_path: Optional[str]
    transcript_path: Optional[str]
    chunk_count: int


@app.post("/session/stop", response_model=SessionStopOut)
async def session_stop(session_id: str = Form(...)):
    s = _get_session(session_id)
    if s["closed"]:
        return {
            "session_id": session_id,
            "merged_path": str(s["merged_path"]) if s["merged_path"] else None,
            "transcript_path": str(s["transcript_path"]) if s["transcript_path"] else None,
            "chunk_count": len(s["chunks"]),
        }

    merged = _merge_audio(session_id, force_all=True)
    transcript = _write_transcript(session_id)
    s["closed"] = True
    logger.info("Session stopped: %s chunks=%d", session_id, len(s["chunks"]))

    return {
        "session_id": session_id,
        "merged_path": str(merged) if merged else None,
        "transcript_path": str(transcript),
        "chunk_count": len(s["chunks"]),
    }


class TranscribeOut(BaseModel):
    ok: bool
    seq: int
    text: str


@app.post("/transcribe", response_model=TranscribeOut)
async def transcribe(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    seq: int = Form(...),
    model: str = Form("whisper"),
):
    s = _get_session(session_id)
    if s["closed"]:
        raise HTTPException(status_code=409, detail="Session already closed")
    if model not in ("whisper", "voxtral"):
        raise HTTPException(status_code=400, detail="model must be 'whisper' or 'voxtral'")
    if models[model] is None:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not loaded")

    # Idempotent: return cached result for duplicate seq
    if seq in s["chunks"]:
        return {"ok": True, "seq": seq, "text": s["chunks"][seq]}

    raw = await audio.read()
    if len(raw) <= 44:  # 44-byte WAV header only = no audio samples
        return {"ok": True, "seq": seq, "text": ""}
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")

    # Save chunk to disk
    chunk_path = s["audio_dir"] / f"chunk_{seq:05d}.wav"
    chunk_path.write_bytes(raw)

    # Transcribe — Voxtral runs in a ProcessPoolExecutor (MLX not thread-safe)
    try:
        loop = asyncio.get_event_loop()
        if model == "voxtral":
            future = _voxtral_executor.submit(
                _voxtral_subprocess, str(chunk_path), _VOXTRAL_MODEL, _MAX_TOKENS, _TEMPERATURE
            )
            text = await loop.run_in_executor(None, future.result)
        else:
            text = await loop.run_in_executor(None, _do_transcribe, str(chunk_path), model)
    except Exception as exc:
        logger.exception("Transcription failed session=%s seq=%d: %s", session_id, seq, exc)
        # Remove saved chunk so retry can re-save cleanly
        chunk_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Transcription failed") from exc

    s["chunks"][seq] = text
    logger.info("Transcribed session=%s seq=%d model=%s chars=%d", session_id, seq, model, len(text))
    return {"ok": True, "seq": seq, "text": text}


@app.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    s = _get_session(session_id)
    ordered = sorted(s["chunks"].items())
    text = "\n".join(t for _, t in ordered if t)
    return {"session_id": session_id, "text": text, "chunk_count": len(s["chunks"])}


@app.get("/audio/{session_id}")
async def get_audio(session_id: str):
    s = _get_session(session_id)
    if not s["merged_path"] or not Path(s["merged_path"]).exists():
        # Try merging on demand
        merged = _merge_audio(session_id)
        if not merged:
            raise HTTPException(status_code=404, detail="No audio available for this session")
    return FileResponse(str(s["merged_path"]), media_type="audio/wav", filename=f"{session_id}.wav")


@app.get("/health")
async def health():
    loaded = {k: v is not None for k, v in models.items()}
    if not any(loaded.values()):
        return JSONResponse(status_code=503, content={"ok": False, "models": loaded})
    return {"ok": True, "models": loaded}


# ── MCP server ────────────────────────────────────────────────────────────────

def _setup_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.warning("fastmcp not installed — MCP server disabled. Install with: pip install mcp[cli]")
        return None

    mcp = FastMCP("stt-server", port=_MCP_PORT)

    @mcp.tool()
    def start_session(model: str = "whisper", user_tag: str = "") -> dict:
        """Start a new STT recording session. Returns session_id."""
        if model not in ("whisper", "voxtral"):
            return {"error": "model must be 'whisper' or 'voxtral'"}
        if models[model] is None:
            return {"error": f"Model '{model}' not loaded"}
        session_id = _create_session(model, user_tag)
        return {"session_id": session_id, "model": model}

    @mcp.tool()
    def stop_session(session_id: str) -> dict:
        """Stop a session, merge audio, and write transcript. Returns file paths."""
        try:
            s = _get_session(session_id)
        except HTTPException as e:
            return {"error": e.detail}
        if s["closed"]:
            return {"session_id": session_id, "already_closed": True}
        merged = _merge_audio(session_id, force_all=True)
        transcript = _write_transcript(session_id)
        s["closed"] = True
        return {
            "session_id": session_id,
            "merged_path": str(merged) if merged else None,
            "transcript_path": str(transcript),
            "chunk_count": len(s["chunks"]),
        }

    @mcp.tool()
    def get_transcript(session_id: str) -> str:
        """Get the full transcript text for a session in chronological order."""
        try:
            s = _get_session(session_id)
        except HTTPException as e:
            return f"Error: {e.detail}"
        ordered = sorted(s["chunks"].items())
        return "\n".join(t for _, t in ordered if t) or "(empty)"

    @mcp.tool()
    def get_audio_path(session_id: str) -> str:
        """Get the path to the merged audio file for a session."""
        try:
            s = _get_session(session_id)
        except HTTPException as e:
            return f"Error: {e.detail}"
        merged = _merge_audio(session_id)
        return str(merged) if merged else "No audio available"

    @mcp.tool()
    def set_model(session_id: str, model: str) -> dict:
        """Switch the STT model for an active session (whisper or voxtral)."""
        if model not in ("whisper", "voxtral"):
            return {"error": "model must be 'whisper' or 'voxtral'"}
        try:
            s = _get_session(session_id)
        except HTTPException as e:
            return {"error": e.detail}
        if models[model] is None:
            return {"error": f"Model '{model}' not loaded on this server"}
        s["model"] = model
        return {"session_id": session_id, "model": model}

    return mcp


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading
    import uvicorn

    mcp = _setup_mcp()

    if mcp is not None:
        def run_mcp():
            mcp.run(transport="sse")
        t = threading.Thread(target=run_mcp, daemon=True)
        t.start()
        logger.info("MCP server starting on port %d", _MCP_PORT)

    uvicorn.run(
        "server:app",
        host=_HOST,
        port=_PORT,
        log_level="info",
        access_log=False,
    )
