"""
ASR service — calls the deployed Modal faster-whisper endpoint.

Endpoint returns:
{
    "text": "full transcription",
    "segments": [{"start": 0.0, "end": 3.1, "text": "..."}],
    "language": "en",
    "language_probability": 0.98,
    "duration": 26.71
}
"""
from __future__ import annotations

import os

import httpx

ASR_ENDPOINT = os.getenv(
    "ASR_ENDPOINT",
    "https://howdybunny--tiktalk-asr-serve-app.modal.run/transcribe",
)

# Modal cold-start can take ~30s; give generous timeout
_TIMEOUT = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=5.0)


async def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """Upload audio bytes to the ASR endpoint and return the transcript dict."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            ASR_ENDPOINT,
            files={"audio": (filename, audio_bytes, _guess_content_type(filename))},
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"ASR endpoint returned {response.status_code}: {response.text}"
        )

    return response.json()


def _guess_content_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
    }.get(ext, "application/octet-stream")
