"""
Fusion layer — combines four dimension scores into a final score.

Design goals:
  - All weights, gates, and adjustments are explicit and logged in fusion_trace.
  - encouragement_level is derived from final_score + risk_flags here,
    NOT by the LLM.
  - score_confidence reflects how trustworthy the score is.
"""
from __future__ import annotations

from typing import Any

# ── Dimension weights (must sum to 1.0) ───────────────────────────────────────
WEIGHTS = {
    "semantic":      0.40,
    "grammar":       0.25,
    "pronunciation": 0.20,
    "fluency":       0.15,
}

# ── Risk flag behaviour ───────────────────────────────────────────────────────
_SILENCE_SCORE = 0.0         # override: silence → score = 0
_OFF_TOPIC_CAP = 50.0        # cap final score if off_topic
_LOW_ASR_DISCOUNT = 0.88     # multiply if low ASR confidence
_NOISE_DISCOUNT = 0.92       # multiply if noise detected

# ── Confidence thresholds ─────────────────────────────────────────────────────
_HIGH_CONF_FLOOR = 75.0
_MED_CONF_FLOOR = 50.0

# ── Encouragement thresholds ─────────────────────────────────────────────────
_ENC_HIGH = 75.0
_ENC_MED = 50.0


def fuse(
    semantic: float,
    grammar: float,
    pronunciation: float,
    fluency: float,
    risk_flags: list[str],
) -> dict[str, Any]:
    """
    Combine dimension scores, apply gating rules, and produce:
      final_score, score_confidence, encouragement_level, fusion_trace
    """
    trace: dict[str, Any] = {
        "weights": dict(WEIGHTS),
        "input_scores": {
            "semantic": semantic,
            "grammar": grammar,
            "pronunciation": pronunciation,
            "fluency": fluency,
        },
        "risk_flags": risk_flags,
        "gates_applied": [],
        "adjustments": [],
    }

    # ── Gate: silence ─────────────────────────────────────────────────────────
    if "silence" in risk_flags:
        trace["gates_applied"].append("silence → score forced to 0")
        trace["final_score"] = _SILENCE_SCORE
        trace["score_confidence"] = 0.1
        trace["encouragement_level"] = "low"
        return _build_result(_SILENCE_SCORE, 0.1, "low", trace)

    # ── Weighted base score ────────────────────────────────────────────────────
    base = (
        WEIGHTS["semantic"]      * semantic
        + WEIGHTS["grammar"]     * grammar
        + WEIGHTS["pronunciation"] * pronunciation
        + WEIGHTS["fluency"]     * fluency
    )
    trace["base_score"] = round(base, 2)

    adjusted = base

    # ── Gate: off_topic cap ────────────────────────────────────────────────────
    if "off_topic" in risk_flags:
        prev = adjusted
        adjusted = min(adjusted, _OFF_TOPIC_CAP)
        trace["gates_applied"].append(
            f"off_topic → cap {prev:.1f} → {adjusted:.1f}"
        )

    # ── Discount: low ASR confidence ──────────────────────────────────────────
    if "low_asr_confidence" in risk_flags:
        prev = adjusted
        adjusted *= _LOW_ASR_DISCOUNT
        trace["adjustments"].append(
            f"low_asr_confidence discount ×{_LOW_ASR_DISCOUNT} → {prev:.1f} → {adjusted:.1f}"
        )

    # ── Discount: noise ───────────────────────────────────────────────────────
    if "noise" in risk_flags:
        prev = adjusted
        adjusted *= _NOISE_DISCOUNT
        trace["adjustments"].append(
            f"noise discount ×{_NOISE_DISCOUNT} → {prev:.1f} → {adjusted:.1f}"
        )

    final = round(adjusted, 1)

    # ── Score confidence ──────────────────────────────────────────────────────
    n_flags = len(risk_flags)
    if n_flags == 0 and final >= _HIGH_CONF_FLOOR:
        confidence = 0.90
    elif n_flags == 0 and final >= _MED_CONF_FLOOR:
        confidence = 0.80
    elif n_flags == 1:
        confidence = 0.65
    elif n_flags >= 2:
        confidence = 0.45
    else:
        confidence = 0.75

    # ── Encouragement level (deterministic from score + flags) ────────────────
    if "silence" in risk_flags or "off_topic" in risk_flags:
        enc = "low"
    elif final >= _ENC_HIGH and n_flags == 0:
        enc = "high"
    elif final >= _ENC_MED:
        enc = "medium"
    else:
        enc = "low"

    trace["final_score"] = final
    trace["score_confidence"] = confidence
    trace["encouragement_level"] = enc

    return _build_result(final, confidence, enc, trace)


def _build_result(
    final_score: float,
    confidence: float,
    encouragement: str,
    trace: dict,
) -> dict[str, Any]:
    return {
        "final_score": final_score,
        "score_confidence": confidence,
        "encouragement_level": encouragement,
        "fusion_trace": trace,
    }
