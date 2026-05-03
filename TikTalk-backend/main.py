"""
TikTalk Backend — unified FastAPI gateway.

Start:
    uv run uvicorn main:app --reload --port 8080

Docs:
    http://localhost:8080/docs
"""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from routers import session  # noqa: E402 (load_dotenv must run first)

app = FastAPI(
    title="TikTalk Backend",
    version="0.1.0",
    description=(
        "Unified backend for TikTalk — AI-powered English speaking coach "
        "for Singapore primary school students.\n\n"
        "**Flow:**\n"
        "1. `POST /session/start` → generate practice image\n"
        "2. `POST /session/vlm` → analyse image, get ground truth *(call while student views image)*\n"
        "3. `POST /session/evaluate` → transcribe audio + score response\n"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router, prefix="/session", tags=["Session"])


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "modules": {
            "diffusion": "dall-e-3",
            "vlm": "qwen + openai + gemini-judge",
            "asr": "whisper-large-v3 (modal)",
            "evaluate": "gpt-4o-mini (llm-based)",
        },
    }
