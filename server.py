"""
server.py — unified STT backend supporting Whisper and Voxtral models.

Environment variables:
  LOAD_WHISPER       true|false  (default: true)
  LOAD_VOXTRAL       true|false  (default: true)
  LOAD_VAANI         true|false  (default: true)
  WHISPER_MODEL      model id/size (default: collabora/faster-whisper-medium-hindi)
  WHISPER_LANGUAGE   language code; empty = auto-detect (default: hi)
  COMPUTE_TYPE       float32     (default: float32)
  BEAM_SIZE          int         (default: 1)
  VOXTRAL_MODEL      model path  (default: mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit)
  VAANI_MODEL        model path  (default: ARTPARK-IISc/whisper-medium-vaani-hindi)
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
import importlib
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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


def _load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding shell environment."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv(Path(__file__).with_name(".env"))

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
# Guard against duplicate handlers when uvicorn forks/reloads the module
if not logger.handlers:
    logger.addHandler(_fh)
    logger.addHandler(_ch)

# ── Config ────────────────────────────────────────────────────────────────────

_LOAD_WHISPER = os.getenv("LOAD_WHISPER", "true").lower() == "true"
_LOAD_VOXTRAL = os.getenv("LOAD_VOXTRAL", "true").lower() == "true"
_LOAD_VAANI = os.getenv("LOAD_VAANI", "true").lower() == "true"
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "collabora/faster-whisper-medium-hindi")
_WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "hi").strip() or None
_COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")
_BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))
_VOXTRAL_MODEL = os.getenv("VOXTRAL_MODEL", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
_VAANI_MODEL = os.getenv("VAANI_MODEL", "ARTPARK-IISc/whisper-medium-vaani-hindi")
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

VALID_MODELS = ("whisper", "voxtral", "vaani")
MODEL_ERROR = "model must be 'whisper', 'voxtral', or 'vaani'"
models: dict = {"whisper": None, "voxtral": None, "vaani": None}


def _is_valid_model(model_name: str) -> bool:
    return model_name in VALID_MODELS


def _validate_vaani_model_path(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    model_path = Path(model_id)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=model_id))

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {model_path}")

    for filename in ("weights.safetensors", "model.safetensors", "weights.npz"):
        if (model_path / filename).exists():
            return model_path

    raise FileNotFoundError(f"Could not find Vaani weights in {model_path}")


_voxtral_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
_vaani_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
_io_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_voxtral_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _voxtral_executor
    if _voxtral_executor is None:
        _voxtral_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    return _voxtral_executor


def _get_vaani_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _vaani_executor
    if _vaani_executor is None:
        _vaani_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    return _vaani_executor


def _get_io_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _io_executor
    if _io_executor is None:
        _io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="stt-io")
    return _io_executor


def _load_models():
    if _LOAD_WHISPER:
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model: %s language=%s", _WHISPER_MODEL, _WHISPER_LANGUAGE or "auto")
            cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "8"))
            models["whisper"] = WhisperModel(_WHISPER_MODEL, compute_type=_COMPUTE_TYPE, cpu_threads=cpu_threads, num_workers=1)
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

    if _LOAD_VAANI:
        try:
            import mlx_whisper
            logger.info("Loading Vaani model: %s", _VAANI_MODEL)
            _validate_vaani_model_path(_VAANI_MODEL)
            models["vaani"] = True
            logger.info("Vaani ready")
        except Exception as e:
            logger.error("Failed to load Vaani: %s", e)


def _transcribe_whisper(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe with Whisper. language=None → auto-detect, language='' → use server default."""
    m = models["whisper"]
    transcribe_kwargs = {
        "beam_size": _BEAM_SIZE,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 300},
        "condition_on_previous_text": False,
        "without_timestamps": True,
        "word_timestamps": False,
    }
    # Priority: explicit per-request language > server default > auto-detect
    effective_lang = language if language is not None else _WHISPER_LANGUAGE
    if effective_lang:
        transcribe_kwargs["language"] = effective_lang
    segments, _ = m.transcribe(audio_path, **transcribe_kwargs)
    return " ".join(s.text.strip() for s in segments if s.text.strip())


def _transcribe_voxtral(audio_path: str) -> str:
    m = models["voxtral"]
    kwargs = {"max_tokens": _MAX_TOKENS, "temperature": _TEMPERATURE}
    result = m.generate(audio_path, **kwargs)
    return result.text.strip() if hasattr(result, "text") else str(result).strip()


def _transcribe_vaani(audio_path: str, language: Optional[str] = None) -> str:
    # We use the subprocess for vaani too as MLX can be picky about threads
    effective_language = language if language is not None else _WHISPER_LANGUAGE
    if effective_language == "":
        effective_language = None
    future = _get_vaani_executor().submit(
        _vaani_subprocess, audio_path, _VAANI_MODEL, effective_language
    )
    return future.result()


# Top-level function for subprocess execution (MLX is not thread-safe; must run
# in a separate process to avoid segfaults when called from a thread pool).
def _voxtral_subprocess(audio_path: str, voxtral_model_id: str, max_tokens: int, temperature: float) -> str:
    from mlx_audio.stt.utils import load as voxtral_load
    m = voxtral_load(voxtral_model_id)
    result = m.generate(audio_path, max_tokens=max_tokens, temperature=temperature)
    return result.text.strip() if hasattr(result, "text") else str(result).strip()


def _vaani_subprocess(audio_path: str, model_id: str, language: Optional[str]) -> str:
    import mlx_whisper.load_models as load_models
    import mlx_whisper.whisper as whisper
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download

    def to_mlx_key(key: str) -> Optional[str]:
        """Map Hugging Face Whisper checkpoint names to mlx-whisper names."""
        if not key.startswith("model."):
            return key

        key = key.removeprefix("model.")
        if key == "encoder.embed_positions.weight":
            return None
        if key.startswith("encoder.layer_norm."):
            return key.replace("encoder.layer_norm.", "encoder.ln_post.", 1)
        if key == "decoder.embed_tokens.weight":
            return "decoder.token_embedding.weight"
        if key == "decoder.embed_positions.weight":
            return "decoder.positional_embedding"
        if key.startswith("decoder.layer_norm."):
            return key.replace("decoder.layer_norm.", "decoder.ln.", 1)

        key = key.replace(".layers.", ".blocks.")
        key = key.replace(".self_attn_layer_norm.", ".attn_ln.")
        key = key.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")
        key = key.replace(".final_layer_norm.", ".mlp_ln.")
        key = key.replace(".self_attn.", ".attn.")
        key = key.replace(".encoder_attn.", ".cross_attn.")
        key = key.replace(".q_proj.", ".query.")
        key = key.replace(".k_proj.", ".key.")
        key = key.replace(".v_proj.", ".value.")
        key = key.replace(".out_proj.", ".out.")
        key = key.replace(".fc1.", ".mlp1.")
        key = key.replace(".fc2.", ".mlp2.")
        return key

    def normalize_weights(raw_weights: dict) -> dict:
        normalized = {}
        for key, value in raw_weights.items():
            new_key = to_mlx_key(key)
            if new_key is None:
                continue
            if new_key in ("encoder.conv1.weight", "encoder.conv2.weight") and value.ndim == 3:
                value = value.transpose(0, 2, 1)
            normalized[new_key] = value
        return normalized

    def robust_load_model(path_or_hf_repo: str, dtype: mx.Dtype = mx.float32) -> whisper.Whisper:
        model_path = Path(path_or_hf_repo)
        if not model_path.exists():
            model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

        with open(str(model_path / "config.json"), "r") as f:
            config = json.loads(f.read())

        # Mapping for standard HF config to MLX ModelDimensions
        mapping = {
            "n_mels": "num_mel_bins",
            "n_audio_ctx": "max_source_positions",
            "n_audio_state": "d_model",
            "n_audio_head": "encoder_attention_heads",
            "n_audio_layer": "encoder_layers",
            "n_vocab": "vocab_size",
            "n_text_ctx": "max_target_positions",
            "n_text_state": "d_model",
            "n_text_head": "decoder_attention_heads",
            "n_text_layer": "decoder_layers",
        }

        dims_args = {}
        if "n_mels" in config:  # Already in MLX format
            dims_args = {k: v for k, v in config.items() if k != "model_type" and k != "quantization"}
        else:
            for mlx_key, hf_key in mapping.items():
                val = config.get(hf_key) if hf_key in config else config.get(mlx_key)
                if val is not None:
                    dims_args[mlx_key] = val

        model_args = whisper.ModelDimensions(**dims_args)

        # Try multiple weight filenames
        wf = model_path / "weights.safetensors"
        if not wf.exists():
            wf = model_path / "model.safetensors"
        if not wf.exists():
            wf = model_path / "weights.npz"

        if not wf.exists():
            raise FileNotFoundError(f"Could not find weights in {model_path}")

        weights = normalize_weights(mx.load(str(wf)))
        model = whisper.Whisper(model_args, dtype)

        quantization = config.get("quantization")
        if quantization is not None:
            class_predicate = (
                lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                and f"{p}.scales" in weights
            )
            nn.quantize(model, **quantization, class_predicate=class_predicate)

        weights = tree_unflatten(list(weights.items()))
        model.update(weights)
        mx.eval(model.parameters())
        return model

    # Monkeypatch the load_model function used by mlx_whisper.transcribe
    transcribe_mod = importlib.import_module("mlx_whisper.transcribe")
    load_models.load_model = robust_load_model
    transcribe_mod.load_model = robust_load_model

    try:
        import logging
        logging.getLogger("stt_server").info(f"Using robust loader for {model_id}")
        # Call transcribe from the module we just monkeypatched
        transcribe_kwargs = {
            "path_or_hf_repo": model_id,
            "condition_on_previous_text": False,
            "word_timestamps": False,
            "fp16": False,
        }
        if language:
            transcribe_kwargs["language"] = language
        result = transcribe_mod.transcribe(audio_path, **transcribe_kwargs)
        return result["text"].strip()
    except Exception as e:
        import logging
        logging.getLogger("stt_server").error(f"Vaani subprocess error: {e}")
        raise


# Whisper is CPU-bound — one inference at a time prevents contention and thrashing
_whisper_sem: asyncio.Semaphore  # initialised in lifespan (needs running loop)


def _do_transcribe(audio_path: str, model_name: str, language: Optional[str] = None) -> str:
    if model_name == "whisper":
        return _transcribe_whisper(audio_path, language=language)
    elif model_name == "voxtral":
        future = _get_voxtral_executor().submit(
            _voxtral_subprocess, audio_path, _VOXTRAL_MODEL, _MAX_TOKENS, _TEMPERATURE
        )
        return future.result()
    elif model_name == "vaani":
        return _transcribe_vaani(audio_path, language=language)
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


def _create_session(model_name: str, user_tag: str = "", language: Optional[str] = None) -> str:
    session_id = str(uuid.uuid4())
    audio_dir = _RECORDINGS_DIR / session_id / "chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sessions[session_id] = {
        "model": model_name,
        "language": language,  # None = use server default; "" = auto-detect; "hi"/"en"/etc = override
        "chunks": {},
        "audio_dir": audio_dir,
        "merged_path": None,
        "transcript_path": None,
        "started_at": datetime.now(timezone.utc),
        "closed": False,
        "last_merged_chunk": -1,
        "user_tag": user_tag,
        "ws_done": None,  # asyncio.Event set when WS handler finishes draining
    }
    logger.info("Session started: %s model=%s lang=%s user=%s", session_id, model_name, language or "server-default", user_tag)
    return session_id


def _merge_audio(session_id: str, force_all: bool = False) -> Optional[Path]:
    """Incrementally append only new chunks to merged.wav. Returns merged path."""
    s = sessions[session_id]
    chunks_dir = s["audio_dir"]
    session_dir = chunks_dir.parent
    merged_path = session_dir / "merged.wav"

    all_chunks = sorted(chunks_dir.glob("chunk_*.wav"), key=lambda p: int(p.stem.split("_")[1]))
    if not all_chunks:
        return None

    last_merged = s.get("last_merged_chunk", -1)
    new_chunks = [cf for cf in all_chunks if int(cf.stem.split("_")[1]) > last_merged]
    if not new_chunks:
        return merged_path if merged_path.exists() else None

    # Append mode: open existing merged.wav and add only new frames
    if merged_path.exists() and last_merged >= 0:
        with wave.open(str(merged_path), "rb") as existing:
            params = existing.getparams()
            existing_frames = existing.readframes(existing.getnframes())
    else:
        params = None
        existing_frames = None

    try:
        with wave.open(str(merged_path), "wb") as out_wav:
            first_new = True
            if params is not None:
                out_wav.setparams(params)
                out_wav.writeframes(existing_frames)
            for cf in new_chunks:
                try:
                    with wave.open(str(cf), "rb") as in_wav:
                        if params is None and first_new:
                            out_wav.setparams(in_wav.getparams())
                            params = in_wav.getparams()
                            first_new = False
                        out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
                except Exception as e:
                    logger.warning("Skipping corrupt chunk %s: %s", cf.name, e)
    except Exception as e:
        logger.error("Merge failed for session %s: %s", session_id, e)
        return None

    s["last_merged_chunk"] = int(new_chunks[-1].stem.split("_")[1])
    s["merged_path"] = merged_path
    logger.info("Merged +%d chunks for session %s → %s", len(new_chunks), session_id, merged_path)
    return merged_path


def _write_transcript(session_id: str) -> Path:
    s = sessions[session_id]
    session_dir = s["audio_dir"].parent
    transcript_path = session_dir / "transcript.txt"

    ordered = sorted(s["chunks"].items())
    with transcript_path.open("w", encoding="utf-8") as f:
        for _, text in ordered:
            if text:
                f.write(text)
                f.write("\n")

    s["transcript_path"] = transcript_path
    return transcript_path


async def _merge_and_write_async(session_id: str) -> tuple:
    """Run merge + transcript write concurrently in the IO thread pool."""
    loop = asyncio.get_event_loop()
    io_executor = _get_io_executor()
    merge_fut = loop.run_in_executor(io_executor, _merge_audio, session_id, True)
    transcript_fut = loop.run_in_executor(io_executor, _write_transcript, session_id)
    merged, transcript = await asyncio.gather(merge_fut, transcript_fut)
    return merged, transcript


# ── Lifespan ──────────────────────────────────────────────────────────────────

_merge_task: Optional[asyncio.Task] = None


async def _periodic_merge():
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(_MERGE_INTERVAL_SECS)
        tasks = [
            loop.run_in_executor(_get_io_executor(), _merge_audio, sid, False)
            for sid, s in list(sessions.items())
            if not s["closed"]
        ]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sid, result in zip([sid for sid, s in list(sessions.items()) if not s["closed"]], results):
                if isinstance(result, Exception):
                    logger.warning("Periodic merge failed for %s: %s", sid, result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _merge_task, _whisper_sem, _voxtral_executor, _vaani_executor, _io_executor
    _whisper_sem = asyncio.Semaphore(1)  # one Whisper inference at a time
    _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    _load_models()
    if _MERGE_INTERVAL_SECS > 0:
        _merge_task = asyncio.create_task(_periodic_merge())
    yield
    if _merge_task:
        _merge_task.cancel()
    if _voxtral_executor is not None:
        _voxtral_executor.shutdown(wait=False)
        _voxtral_executor = None
    if _vaani_executor is not None:
        _vaani_executor.shutdown(wait=False)
        _vaani_executor = None
    if _io_executor is not None:
        _io_executor.shutdown(wait=False)
        _io_executor = None
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
    language: Optional[str]


@app.post("/session/start", response_model=SessionStartOut)
async def session_start(
    model: str = Form("whisper"),
    user_tag: str = Form(""),
    language: str = Form(""),  # "" = use server default; "auto" or specific code like "en"/"hi"
):
    if not _is_valid_model(model):
        raise HTTPException(status_code=400, detail=MODEL_ERROR)
    if models[model] is None:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not loaded")
    # Normalise: "auto" → None (faster-whisper auto-detect), "" → None (use server default)
    lang = language.strip().lower() or None
    if lang == "auto":
        lang = ""  # empty string signals explicit auto-detect, overriding server default
    session_id = _create_session(model, user_tag, language=lang)
    return {"session_id": session_id, "model": model, "language": lang}


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

    s["closed"] = True

    # Wait for the WS handler to finish draining all in-flight transcriptions
    # before writing the transcript so it's complete, not partial.
    ws_done: Optional[asyncio.Event] = s.get("ws_done")
    if ws_done is not None:
        try:
            await asyncio.wait_for(ws_done.wait(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning("Session %s WS drain timed out after 60s — writing partial transcript", session_id)

    chunk_count = len(s["chunks"])
    logger.info("Session stopped: %s chunks=%d — running merge+transcript in background", session_id, chunk_count)

    # Run merge + transcript write concurrently in IO thread pool
    merged, transcript = await _merge_and_write_async(session_id)

    return {
        "session_id": session_id,
        "merged_path": str(merged) if merged else None,
        "transcript_path": str(transcript),
        "chunk_count": chunk_count,
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
    if not _is_valid_model(model):
        raise HTTPException(status_code=400, detail=MODEL_ERROR)
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

    try:
        loop = asyncio.get_event_loop()
        lang = s.get("language")  # None = server default, "" = auto-detect, "hi"/"en"/etc = override
        if model == "voxtral":
            future = _get_voxtral_executor().submit(
                _voxtral_subprocess, str(chunk_path), _VOXTRAL_MODEL, _MAX_TOKENS, _TEMPERATURE
            )
            text = await loop.run_in_executor(None, future.result)
        else:
            text = await loop.run_in_executor(None, _do_transcribe, str(chunk_path), model, lang)
    except Exception as exc:
        logger.exception("Transcription failed session=%s seq=%d: %s", session_id, seq, exc)
        chunk_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Transcription failed") from exc

    s["chunks"][seq] = text
    logger.info("Transcribed session=%s seq=%d model=%s lang=%s chars=%d", session_id, seq, model, s.get("language") or "default", len(text))
    return {"ok": True, "seq": seq, "text": text}


@app.websocket("/ws/transcribe/{session_id}")
async def ws_transcribe(websocket: WebSocket, session_id: str, model: str = "whisper"):
    """
    WebSocket streaming transcription endpoint.
    Client sends binary frames: 4-byte little-endian seq_number + WAV bytes.
    Server responds with JSON: {"seq": N, "text": "..."}
    """
    await websocket.accept()
    try:
        s = _get_session(session_id)
    except HTTPException as exc:
        await websocket.close(code=1008, reason=exc.detail)
        return

    if not _is_valid_model(model):
        await websocket.close(code=1008, reason=MODEL_ERROR)
        return

    if models[model] is None:
        await websocket.close(code=1008, reason=f"Model '{model}' not loaded")
        return

    loop = asyncio.get_event_loop()
    tmp_dir = s["audio_dir"]

    # Register a done-event so session/stop can wait for full drain
    ws_done = asyncio.Event()
    s["ws_done"] = ws_done

    # One Whisper inference at a time — CPU-bound, concurrency causes thrashing
    # Chunks queue up behind the semaphore and are processed in arrival order
    async def _transcribe_chunk(seq: int, wav_bytes: bytes):
        chunk_path = tmp_dir / f"chunk_{seq:05d}.wav"
        chunk_path.write_bytes(wav_bytes)
        try:
            lang = s.get("language")
            if model == "voxtral":
                future = _get_voxtral_executor().submit(
                    _voxtral_subprocess, str(chunk_path), _VOXTRAL_MODEL, _MAX_TOKENS, _TEMPERATURE
                )
                text = await loop.run_in_executor(None, future.result)
            elif model == "vaani":
                text = await loop.run_in_executor(None, _transcribe_vaani, str(chunk_path), lang)
            else:
                async with _whisper_sem:
                    text = await loop.run_in_executor(None, _do_transcribe, str(chunk_path), model, lang)
        except Exception as exc:
            logger.exception("WS transcription failed session=%s seq=%d: %s", session_id, seq, exc)
            chunk_path.unlink(missing_ok=True)
            try:
                await websocket.send_json({"seq": seq, "text": "", "error": "transcription failed"})
            except Exception:
                pass
            return
        s["chunks"][seq] = text
        logger.info("WS transcribed session=%s seq=%d chars=%d", session_id, seq, len(text))
        try:
            await websocket.send_json({"seq": seq, "text": text})
        except Exception:
            pass  # WS may already be closed — result is still stored in s["chunks"]

    # Max chunks queued for transcription at once. Beyond this, apply
    # backpressure by waiting for at least one transcription to finish.
    MAX_PENDING = 10
    pending: set[asyncio.Task] = set()

    try:
        while True:
            raw = await websocket.receive_bytes()
            if len(raw) == 0:
                # Zero-byte sentinel: client is done sending, break to drain
                break
            if len(raw) < 5:
                continue

            seq = int.from_bytes(raw[:4], "little")
            wav_bytes = raw[4:]

            if len(wav_bytes) <= 44:
                await websocket.send_json({"seq": seq, "text": ""})
                continue

            if seq in s["chunks"]:
                await websocket.send_json({"seq": seq, "text": s["chunks"][seq]})
                continue

            while len(pending) >= MAX_PENDING:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = None if task.cancelled() else task.exception()
                    if exc:
                        logger.warning("WS session=%s pending transcription task failed: %s", session_id, exc)

            task = asyncio.create_task(_transcribe_chunk(seq, wav_bytes))
            pending.add(task)
            task.add_done_callback(pending.discard)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS session=%s closed with error: %s", session_id, exc)
    finally:
        if pending:
            logger.info("WS session=%s draining %d queued chunks…", session_id, len(pending))
            await asyncio.gather(*pending, return_exceptions=True)
        try:
            await websocket.send_json({"type": "drained"})
        except Exception:
            pass
        ws_done.set()  # unblock session/stop so transcript is written after full drain
        logger.info("WS session=%s fully drained", session_id)


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
        loop = asyncio.get_event_loop()
        merged = await loop.run_in_executor(_get_io_executor(), _merge_audio, session_id, False)
        if not merged:
            raise HTTPException(status_code=404, detail="No audio available for this session")
    return FileResponse(str(s["merged_path"]), media_type="audio/wav", filename=f"{session_id}.wav")


@app.get("/health")
async def health():
    loaded = {k: v is not None for k, v in models.items()}
    if not any(loaded.values()):
        return JSONResponse(status_code=503, content={"ok": False, "models": loaded})
    return {
        "ok": True,
        "models": loaded,
        "defaults": {
            "whisper_language": _WHISPER_LANGUAGE or "auto",
            "whisper_model": _WHISPER_MODEL,
            "voxtral_model": _VOXTRAL_MODEL,
            "vaani_model": _VAANI_MODEL,
        },
    }


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
        if not _is_valid_model(model):
            return {"error": MODEL_ERROR}
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
        """Switch the STT model for an active session (whisper, voxtral, or vaani)."""
        if not _is_valid_model(model):
            return {"error": MODEL_ERROR}
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
