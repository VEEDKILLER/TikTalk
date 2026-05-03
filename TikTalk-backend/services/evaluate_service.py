"""
Evaluate service — orchestrator.

Calls each scorer, gathers traces, runs fusion, and returns the full
EvaluateResponse payload.

Scorer execution plan:
  Parallel:  semantic, grammar, pronunciation, fluency
  Serial:    risk detection → fusion → feedback generation (LLM)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from openai import AsyncOpenAI

from .evaluate import (
    fluency_scorer,
    grammar_scorer,
    pronunciation_scorer,
    semantic_scorer,
    transcript_normalizer,
    fusion,
)

# ── Risk flag detection ───────────────────────────────────────────────────────

def _detect_risk_flags(
    asr_result: dict[str, Any],
    transcript_views: dict[str, Any],
    semantic_result: dict[str, Any],
) -> list[str]:
    flags: set[str] = set()

    duration = asr_result.get("duration", 0.0)
    lang_prob = asr_result.get("language_probability", 1.0)
    segments = asr_result.get("segments", [])
    clean = transcript_views.get("clean_transcript", "")
    word_count = len(clean.split()) if clean.strip() else 0

    # Silence: very short or empty speech
    voiced_time = sum(s["end"] - s["start"] for s in segments)
    if word_count < 3 or voiced_time < 1.0:
        flags.add("silence")

    # Noise: abnormally many empty segments or very low speech ratio
    if duration > 0 and voiced_time / duration < 0.15 and duration > 3.0:
        flags.add("noise")

    # Low ASR confidence
    if lang_prob < 0.60:
        flags.add("low_asr_confidence")

    # Off-topic: only when BOTH semantic signals are very low
    if semantic_result.get("off_topic", False):
        flags.add("off_topic")

    return sorted(flags)


# ── LLM feedback generation ───────────────────────────────────────────────────

_FEEDBACK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["positive_points", "improvement_points", "grammar_focus", "pronunciation_focus"],
    "properties": {
        "positive_points":    {"type": "array", "items": {"type": "string"}},
        "improvement_points": {"type": "array", "items": {"type": "string"}},
        "grammar_focus":      {"type": "array", "items": {"type": "string"}},
        "pronunciation_focus": {"type": "array", "items": {"type": "string"}},
    },
}

_FEEDBACK_SYSTEM = """You are a warm English teacher giving feedback to a Singapore primary school child.

Your job is to generate feedback slots ONLY — do not score, do not judge severity.
Keep language simple, encouraging, and age-appropriate (6–12 years old).

Rules:
- positive_points: 1–3 specific things the child did well (mention what they said)
- improvement_points: 1–2 concrete, actionable suggestions (show an example sentence)
- grammar_focus: one tip per grammar error type found (keep it simple)
- pronunciation_focus: flag at most 2 difficult words; can be empty if speech was clear"""


async def _generate_feedback(
    clean_transcript: str,
    ground_truth: dict[str, Any],
    semantic_result: dict[str, Any],
    grammar_result: dict[str, Any],
    pronunciation_result: dict[str, Any],
) -> dict[str, Any]:
    if not clean_transcript.strip():
        return {
            "positive_points": [],
            "improvement_points": ["Please speak clearly into the microphone and try again."],
            "grammar_focus": [],
            "pronunciation_focus": [],
        }

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("EVALUATE_MODEL", "gpt-4o-mini")
    client = AsyncOpenAI(api_key=api_key)

    context = {
        "scene": ground_truth.get("scene_summary", ""),
        "key_objects": [o["name"] for o in ground_truth.get("main_objects", [])],
        "actions": ground_truth.get("actions", []),
        "child_said": clean_transcript,
        "mentioned_concepts": semantic_result.get("mentioned_concepts", []),
        "missing_concepts": semantic_result.get("missing_concepts", [])[:4],
        "grammar_errors": grammar_result.get("grammar_errors", [])[:4],
        "difficult_words": pronunciation_result.get("weak_words", [])[:3],
    }

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _FEEDBACK_SYSTEM},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "feedback_slots",
                "strict": True,
                "schema": _FEEDBACK_SCHEMA,
            },
        },
        temperature=0.3,
    )

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def evaluate(
    asr_result: dict[str, Any],
    ground_truth: dict[str, Any],
    age_group: str | None = None,
    grade_level: str | None = None,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())[:8]
    t_start = time.monotonic()
    timing: dict[str, float] = {}

    # ── Transcript normalisation ─────────────────────────────────────────────
    t0 = time.monotonic()
    transcript_views = transcript_normalizer.normalize(asr_result.get("text", ""))
    timing["transcript_normalizer_ms"] = round((time.monotonic() - t0) * 1000, 1)

    clean = transcript_views["clean_transcript"]
    segments = asr_result.get("segments", [])
    duration = asr_result.get("duration", 0.0)
    lang_prob = asr_result.get("language_probability", 0.9)

    # ── Parallel scoring: semantic + grammar + fluency ────────────────────────
    t0 = time.monotonic()

    fluency_result = fluency_scorer.compute(segments, duration, clean)
    speech_rate = fluency_result["speech_rate_wpm"]

    async def _run_semantic():
        t = time.monotonic()
        r = semantic_scorer.compute_semantic(clean, ground_truth)
        timing["semantic_ms"] = round((time.monotonic() - t) * 1000, 1)
        return r

    async def _run_grammar():
        t = time.monotonic()
        r = await grammar_scorer.score(clean)
        timing["grammar_ms"] = round((time.monotonic() - t) * 1000, 1)
        return r

    semantic_result, grammar_result = await asyncio.gather(
        _run_semantic(),
        _run_grammar(),
    )
    timing["parallel_scorers_ms"] = round((time.monotonic() - t0) * 1000, 1)

    pronunciation_result = pronunciation_scorer.compute(lang_prob, speech_rate, clean)
    timing["fluency_ms"] = round((time.monotonic() - t0) * 1000, 1)

    # ── Risk flags ────────────────────────────────────────────────────────────
    risk_flags = _detect_risk_flags(asr_result, transcript_views, semantic_result)

    # ── Fusion ────────────────────────────────────────────────────────────────
    fusion_result = fusion.fuse(
        semantic=semantic_result["semantic_score"],
        grammar=grammar_result["grammar_score"],
        pronunciation=pronunciation_result["pronunciation_score"],
        fluency=fluency_result["fluency_score"],
        risk_flags=risk_flags,
    )

    # ── Feedback (LLM, after fusion so context is rich) ───────────────────────
    t0 = time.monotonic()
    feedback = await _generate_feedback(
        clean, ground_truth, semantic_result, grammar_result, pronunciation_result
    )
    timing["feedback_ms"] = round((time.monotonic() - t0) * 1000, 1)
    timing["total_ms"] = round((time.monotonic() - t_start) * 1000, 1)

    # ── Assemble response ─────────────────────────────────────────────────────
    return {
        "request_id": request_id,
        "transcript": {
            "raw": transcript_views["raw_transcript"],
            "clean": transcript_views["clean_transcript"],
            "disfluency_spans": transcript_views["disfluency_spans"],
            "low_confidence_spans": transcript_views["low_confidence_spans"],
        },
        "scores": {
            "semantic":      semantic_result["semantic_score"],
            "grammar":       grammar_result["grammar_score"],
            "pronunciation": pronunciation_result["pronunciation_score"],
            "fluency":       fluency_result["fluency_score"],
            "total":         fusion_result["final_score"],
        },
        "score_confidence": fusion_result["score_confidence"],
        "risk_flags": risk_flags,
        "feedback": {
            **feedback,
            "encouragement_level": fusion_result["encouragement_level"],
        },
        "asr_details": {
            "segments": segments,
            "duration": duration,
            "language_probability": lang_prob,
        },
        "score_details": {
            # Semantic
            "concept_recall":      semantic_result["concept_recall"],
            "reference_similarity": semantic_result["reference_similarity"],
            "mentioned_concepts":  semantic_result["mentioned_concepts"],
            "missing_concepts":    semantic_result["missing_concepts"],
            # Grammar
            "grammar_errors":      grammar_result["grammar_errors"],
            # Pronunciation
            "confidence_basis":    pronunciation_result["confidence_basis"],
            "weak_words":          pronunciation_result["weak_words"],
            "phoneme_scores":      pronunciation_result["phoneme_scores"],
            "weak_phonemes":       pronunciation_result["weak_phonemes"],
            # Fluency
            "speech_rate_wpm":     fluency_result["speech_rate_wpm"],
            "articulation_rate_wpm": fluency_result["articulation_rate_wpm"],
            "pause_count":         fluency_result["pause_count"],
            "pause_ratio":         fluency_result["pause_ratio"],
            "avg_pause_duration":  fluency_result["avg_pause_duration"],
            "prosody_alerts":      fluency_result["prosody_alerts"],
        },
        "fusion_trace": fusion_result["fusion_trace"],
        "module_traces": {
            "semantic": {
                "concept_recall": semantic_result["concept_recall"],
                "reference_similarity": semantic_result["reference_similarity"],
                "off_topic": semantic_result["off_topic"],
            },
            "grammar": {
                "error_count": grammar_result["error_count"],
            },
            "pronunciation": {
                "confidence_basis": pronunciation_result["confidence_basis"],
                "language_probability": lang_prob,
            },
            "fluency": {
                "speech_rate_wpm": fluency_result["speech_rate_wpm"],
                "pause_ratio": fluency_result["pause_ratio"],
                "prosody_alerts": fluency_result["prosody_alerts"],
            },
        },
        "meta": {
            "age_group":   age_group,
            "grade_level": grade_level,
            "timing_ms":   timing,
        },
    }
