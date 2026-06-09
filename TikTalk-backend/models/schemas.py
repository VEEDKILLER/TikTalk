from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── Session Start ───────────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    prompt: str | None = Field(None, description="Free-form image description prompt")
    category: str | None = Field(None, description="Predefined category (e.g. 'outdoor', 'school')")
    character: str = Field("boy", description="Character to use in template")
    action: str | None = Field(
        None,
        description=(
            "Specific action for a deterministic, reproducible scene "
            "(e.g. 'cooking in the kitchen'). Use None or 'random' for a "
            "randomly sampled scene."
        ),
    )
    apply_style: bool = Field(True, description="Append child-friendly style suffix to prompt")


class SessionStartResponse(BaseModel):
    image_id: str = Field(..., description="Opaque ID to reference this image in subsequent calls")
    image_base64: str = Field(..., description="Base64-encoded PNG image")
    prompt_used: str = Field(..., description="Full prompt sent to DALL-E 3")


# ─── VLM Analysis ────────────────────────────────────────────────────────────

class VLMRequest(BaseModel):
    image_id: str = Field(..., description="image_id returned by /session/start")


class MainObject(BaseModel):
    name: str
    count: int
    attributes: list[str]


class GroundTruth(BaseModel):
    scene_summary: str
    detailed_description: str
    main_objects: list[MainObject]
    actions: list[str]
    setting: str
    visible_text: list[str]
    teaching_focus_words: list[str]
    age_level: str


class VLMResponse(BaseModel):
    ground_truth: GroundTruth
    vlm_provider: str = Field(..., description="Which VLM won: 'openai' or 'qwen'")


# ─── Evaluate ────────────────────────────────────────────────────────────────

class TranscriptResult(BaseModel):
    raw: str = Field(..., description="Original ASR output")
    clean: str = Field(..., description="Normalized transcript (fillers removed)")


class Scores(BaseModel):
    semantic: float = Field(..., ge=0, le=100, description="Content relevance to image")
    grammar: float = Field(..., ge=0, le=100, description="Grammatical correctness")
    pronunciation: float = Field(..., ge=0, le=100, description="Pronunciation clarity (ASR-based proxy)")
    fluency: float = Field(..., ge=0, le=100, description="Speech rate and pause pattern score")
    total: float = Field(..., ge=0, le=100, description="Weighted final score")


class GrammarError(BaseModel):
    text: str = Field(..., description="Original text with error")
    correction: str = Field(..., description="Suggested correction")
    error_type: str = Field(..., description="Error category (tense / plural / article / etc.)")


class FeedbackSlots(BaseModel):
    positive_points: list[str]
    improvement_points: list[str]
    grammar_focus: list[str]
    pronunciation_focus: list[str]
    encouragement_level: str = Field(..., description="'high' | 'medium' | 'low'")


class ScoreDetails(BaseModel):
    mentioned_concepts: list[str]
    missing_concepts: list[str]
    grammar_errors: list[GrammarError]
    speech_rate_wpm: float = Field(..., description="Words per minute")
    pause_count: int = Field(..., description="Number of pauses > 0.5s")
    pause_ratio: float = Field(..., description="Fraction of total time spent silent")


class ASRDetails(BaseModel):
    segments: list[dict[str, Any]]
    duration: float
    language_probability: float


class EvaluateResponse(BaseModel):
    transcript: TranscriptResult
    scores: Scores
    risk_flags: list[str] = Field(..., description="['silence', 'off_topic', 'low_asr_confidence']")
    feedback: FeedbackSlots
    asr_details: ASRDetails
    score_details: ScoreDetails


# ─── Utility ─────────────────────────────────────────────────────────────────

class CategoriesResponse(BaseModel):
    categories: list[str]
    characters: list[str]
    actions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-category list of selectable actions for deterministic scenes.",
    )


class HealthResponse(BaseModel):
    status: str
    modules: dict[str, str]
