"""
serve.py
========
Modal deployment for TikTalk ASR inference using faster-whisper.

Deployment:
    modal deploy serve.py

Upload model to Modal Volume:
    modal volume put tiktalk-asr-model ct2_model/ /

Test locally (dev mode):
    modal serve serve.py
"""

import tempfile
from pathlib import Path

import modal
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ─── Modal App Configuration ────────────────────────────────────────────────

app = modal.App("tiktalk-asr")

# Volume to persist the CTranslate2 model
MODEL_VOLUME_NAME = "tiktalk-asr-model"
MODEL_MOUNT_PATH = "/model"
volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# ─── Container Image ────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper>=1.1.0",
        "fastapi[standard]",
        "python-multipart",
        "nvidia-cublas-cu12", # Explicitly install required libraries
        "nvidia-cudnn-cu12"   # Explicitly install required libraries
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib"
    })
)

# ─── Inference Hyperparameters ───────────────────────────────────────────────
# These are used during transcription by faster-whisper.

BEAM_SIZE = 5               # Beam search width (higher = more accurate, slower)
LANGUAGE = "en"             # Target language (English for child speech)
VAD_FILTER = True           # Voice Activity Detection to skip silence
VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,  # Minimum silence duration to split (ms)
}
COMPUTE_TYPE = "float16"    # Inference precision on GPU (float16 for T4)


# ─── Whisper ASR Model Class ────────────────────────────────────────────────


@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_MOUNT_PATH: volume},
    timeout=300,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=5)
class WhisperASR:
    """Faster-Whisper ASR inference service on Modal."""

    @modal.enter()
    def load_model(self):
        """Load the faster-whisper model on container startup."""
        from faster_whisper import WhisperModel

        print(f"Loading faster-whisper model from {MODEL_MOUNT_PATH} ...")
        self.model = WhisperModel(
            MODEL_MOUNT_PATH,
            device="cuda",
            compute_type=COMPUTE_TYPE,
        )
        print("✓ Model loaded successfully.")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        """Transcribe audio bytes and return results.

        Args:
            audio_bytes: Raw audio file bytes (wav or mp3).

        Returns:
            dict with 'text', 'segments', and 'language' keys.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            segments, info = self.model.transcribe(
                tmp.name,
                beam_size=BEAM_SIZE,
                language=LANGUAGE,
                vad_filter=VAD_FILTER,
                vad_parameters=VAD_PARAMETERS,
            )

            segment_list = []
            full_text_parts = []
            for seg in segments:
                segment_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
                full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "segments": segment_list,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
        }


# ─── FastAPI Web App ─────────────────────────────────────────────────────────

web_app = FastAPI(title="TikTalk ASR")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


@web_app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file.

    - **audio**: Audio file (wav, mp3, m4a, flac, ogg, webm)
    """
    # Validate file type
    filename = audio.filename or "audio.wav"
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported file format: '{file_ext}'. "
                f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            },
        )

    # Read the uploaded audio bytes
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty audio file."},
        )

    # Run transcription via the Modal class
    model = WhisperASR()
    result = model.transcribe.remote(audio_bytes)

    return result


@app.function(image=image, timeout=300)
@modal.asgi_app()
def serve_app():
    """Serve the FastAPI app as a Modal ASGI app."""
    return web_app
