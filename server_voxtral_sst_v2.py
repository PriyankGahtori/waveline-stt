import os
import tempfile
import logging
import logging.handlers
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Logging setup: midnight rollover, keep 30 days, always append ---
LOG_DIR = os.getenv("LOG_DIR", ".")
LOG_PATH = os.path.join(LOG_DIR, "transcriptions.log")

_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_PATH,
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
    delay=False,
    utc=True,
)
_handler.suffix = "%Y-%m-%d"
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"))

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"))

logger = logging.getLogger("voxtral_sst")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.addHandler(_console)

# --- Configuration ---
MODEL_PATH = os.getenv("VOXTRAL_MODEL", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
TRANSCRIPTION_DELAY_MS = os.getenv("TRANSCRIPTION_DELAY_MS")
if TRANSCRIPTION_DELAY_MS is not None:
    TRANSCRIPTION_DELAY_MS = int(TRANSCRIPTION_DELAY_MS)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from mlx_audio.stt.utils import load
    logger.info("Loading Voxtral model: %s", MODEL_PATH)
    model = load(MODEL_PATH)
    logger.info("Model loaded and ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Voxtral STT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed)
    return response


class TranscriptionOut(BaseModel):
    text: str


@app.post("/transcribe", response_model=TranscriptionOut)
async def transcribe(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = ".wav"
    if audio.filename:
        _, ext = os.path.splitext(audio.filename)
        if ext:
            suffix = ext.lower()

    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            audio_path = tmp.name

        generate_kwargs = {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        if TRANSCRIPTION_DELAY_MS is not None:
            generate_kwargs["transcription_delay_ms"] = TRANSCRIPTION_DELAY_MS

        if suffix == ".mp3":
            from mlx_audio.stt.utils import load_audio
            input_audio = load_audio(audio_path)
        else:
            input_audio = audio_path

        result = model.generate(input_audio, **generate_kwargs)
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        logger.info("transcription filename=%s chars=%d text=%r", audio.filename, len(text), text[:120])
        return {"text": text}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed for filename=%s: %s", audio.filename, exc)
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "model not loaded"})
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_voxtral_sst_v2:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,  # handled by our middleware
    )
