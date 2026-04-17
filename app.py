import io
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")  
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")  
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)

class TranscriptionOut(BaseModel):
    text: str

@app.post("/transcribe", response_model=TranscriptionOut)
async def transcribe(audio: UploadFile = File(...)):
    
    raw = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
        wav_path = tmp.name

    try:
        segments, info = model.transcribe(
            wav_path,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            language="en",  
        )
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        return {"text": text}
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

@app.get("/health")
async def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
