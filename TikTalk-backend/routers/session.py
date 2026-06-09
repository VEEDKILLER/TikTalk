"""
Session router — three endpoints that form the full TikTalk practice flow.

Flow
────
  1. POST /session/start
     Generate image with DALL-E 3 → returns image_base64 + image_id.

  2. POST /session/vlm
     Run VLM pipeline on the image → persists ground_truth server-side.

  3. POST /session/evaluate
     Send audio + image_id → backend loads ground_truth, runs ASR + scoring.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import (
    CategoriesResponse,
    SessionStartRequest,
    SessionStartResponse,
    VLMRequest,
    VLMResponse,
)
from services import asr_service, diffusion_service, evaluate_service, vlm_service

router = APIRouter()


# ── 1. Generate image ─────────────────────────────────────────────────────────

@router.post("/start", response_model=SessionStartResponse)
async def session_start(req: SessionStartRequest):
    """Generate a practice image with DALL-E 3."""
    try:
        base_prompt = diffusion_service.resolve_prompt(
            req.prompt, req.category, req.character, req.action
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await diffusion_service.generate_image(base_prompt, req.apply_style)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {e}")

    return SessionStartResponse(
        image_id=result["image_id"],
        image_base64=result["image_base64"],
        prompt_used=result["prompt_used"],
    )


# ── 2. VLM analysis ───────────────────────────────────────────────────────────

@router.post("/vlm", response_model=VLMResponse)
async def session_vlm(req: VLMRequest):
    """Analyse image and persist ground truth (call while student views image)."""
    try:
        image_path = diffusion_service.get_image_path(req.image_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = await vlm_service.analyze_image(image_path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"VLM analysis failed: {e}")

    return VLMResponse(
        ground_truth=result["ground_truth"],
        vlm_provider=result["vlm_provider"],
    )


# ── 3. Evaluate student response ──────────────────────────────────────────────

@router.post("/evaluate")
async def session_evaluate(
    audio: UploadFile = File(..., description="Student audio (.wav / .mp3 / .webm …)"),
    image_id: str = Form(..., description="image_id from /session/start"),
    age_group: Optional[str] = Form(None, description="e.g. '6-8', '9-10', '11-12'"),
    grade_level: Optional[str] = Form(None, description="e.g. 'P1' … 'P6'"),
):
    """
    Evaluate the student's spoken response.

    Accepts multipart/form-data:
      - audio       : audio file
      - image_id    : returned by /session/start  (backend loads GT from file)
      - age_group   : optional
      - grade_level : optional
    """
    # ── Load ground truth persisted by /session/vlm ───────────────────────────
    try:
        ground_truth = vlm_service.load_ground_truth(image_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # ── Read audio ────────────────────────────────────────────────────────────
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    # ── ASR ───────────────────────────────────────────────────────────────────
    try:
        asr_result = await asr_service.transcribe(audio_bytes, audio.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ASR transcription failed: {e}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    try:
        result = await evaluate_service.evaluate(
            asr_result=asr_result,
            ground_truth=ground_truth,
            age_group=age_group,
            grade_level=grade_level,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {e}")

    return result


# ── Utility ───────────────────────────────────────────────────────────────────

@router.get("/categories", response_model=CategoriesResponse)
def list_categories():
    return CategoriesResponse(**diffusion_service.list_options())
