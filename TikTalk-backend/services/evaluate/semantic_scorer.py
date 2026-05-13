"""
Semantic scorer — three-segment baseline (no LLM, fully rule-based).

Segments
────────
1. concept_f1        Recall of GT concepts (objects / actions / keywords)
                     in the clean transcript.
2. reference_sim     Word-level F1 between transcript and GT reference text.
3. semantic_score    Weighted fusion of the two signals (0–100).

off_topic is flagged here when BOTH signals are very low.
"""
from __future__ import annotations

import re
from typing import Any

# Stop words to exclude from word-overlap calculation
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "this", "that", "there", "here", "can", "see", "some", "has", "have",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP_WORDS and len(w) > 1}


# ── Segment 1: concept recall ─────────────────────────────────────────────────

def _extract_gt_concepts(ground_truth: dict) -> list[str]:
    """Collect all concept strings from the VLM ground truth."""
    concepts: list[str] = []

    for obj in ground_truth.get("main_objects", []):
        name = obj.get("name", "").strip().lower()
        if name:
            concepts.append(name)

    for action in ground_truth.get("actions", []):
        # Keep multi-word actions as phrases and also add individual content words
        phrase = action.strip().lower()
        if phrase:
            concepts.append(phrase)
            for w in _WORD_RE.findall(phrase):
                if w.lower() not in _STOP_WORDS and len(w) > 2:
                    concepts.append(w.lower())

    for kw in ground_truth.get("teaching_focus_words", []):
        kw = kw.strip().lower()
        if kw:
            concepts.append(kw)

    # Deduplicate, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def compute_concept_f1(clean_transcript: str, ground_truth: dict) -> dict[str, Any]:
    transcript_lower = clean_transcript.lower()
    gt_concepts = _extract_gt_concepts(ground_truth)

    if not gt_concepts:
        return {
            "concept_recall": 0.0,
            "mentioned_concepts": [],
            "missing_concepts": [],
            "total_gt_concepts": 0,
        }

    mentioned = [c for c in gt_concepts if c in transcript_lower]
    missing = [c for c in gt_concepts if c not in transcript_lower]
    recall = len(mentioned) / len(gt_concepts)

    return {
        "concept_recall": round(recall, 3),
        "mentioned_concepts": mentioned,
        "missing_concepts": missing,
        "total_gt_concepts": len(gt_concepts),
    }


# ── Segment 2: reference word-level similarity ────────────────────────────────

def compute_reference_similarity(clean_transcript: str, ground_truth: dict) -> float:
    """Word-level F1 between transcript and the GT reference text."""
    ref_parts = [
        ground_truth.get("scene_summary", ""),
        ground_truth.get("detailed_description", ""),
        ground_truth.get("setting", ""),
        " ".join(ground_truth.get("teaching_focus_words", [])),
        " ".join(a for a in ground_truth.get("actions", [])),
    ]
    ref_words = _words(" ".join(ref_parts))
    trans_words = _words(clean_transcript)

    if not ref_words or not trans_words:
        return 0.0

    intersection = ref_words & trans_words
    precision = len(intersection) / len(trans_words)
    recall = len(intersection) / len(ref_words)

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 3)


# ── Segment 3: fusion → semantic_score ───────────────────────────────────────

_CONCEPT_WEIGHT = 0.65
_REF_WEIGHT = 0.35

_OFF_TOPIC_CONCEPT_THRESHOLD = 0.15
_OFF_TOPIC_SIM_THRESHOLD = 0.10


def compute_semantic(clean_transcript: str, ground_truth: dict) -> dict[str, Any]:
    """Run all three segments and return the full semantic result."""
    concept_result = compute_concept_f1(clean_transcript, ground_truth)
    reference_sim = compute_reference_similarity(clean_transcript, ground_truth)

    concept_recall = concept_result["concept_recall"]
    raw_score = _CONCEPT_WEIGHT * concept_recall + _REF_WEIGHT * reference_sim
    semantic_score = round(raw_score * 100, 1)

    off_topic = (
        concept_recall < _OFF_TOPIC_CONCEPT_THRESHOLD
        and reference_sim < _OFF_TOPIC_SIM_THRESHOLD
    )

    return {
        "semantic_score": semantic_score,
        "concept_recall": concept_recall,
        "reference_similarity": reference_sim,
        "mentioned_concepts": concept_result["mentioned_concepts"],
        "missing_concepts": concept_result["missing_concepts"],
        "total_gt_concepts": concept_result["total_gt_concepts"],
        "off_topic": off_topic,
    }
