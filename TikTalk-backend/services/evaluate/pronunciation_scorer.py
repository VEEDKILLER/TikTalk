"""
Pronunciation scorer — speech clarity proxy (MVP).

Naming is intentionally honest: this is NOT a real pronunciation scorer.
It is a proxy based on ASR metadata until a GOP / phoneme-alignment
pipeline is integrated (see evaluateModule.md §10).

Output fields:
  speech_clarity_score   0–100 proxy score (alias: pronunciation_score)
  confidence_basis       What signal was used to compute the score
  weak_words             Words likely to cause pronunciation difficulty
                         (heuristic, based on phoneme complexity)
  phoneme_scores         Empty list — reserved for future GOP integration
  weak_phonemes          Empty list — reserved for future GOP integration
"""
from __future__ import annotations

import re
from typing import Any

# ── Phoneme difficulty heuristics ─────────────────────────────────────────────
# Words / patterns that Singapore primary students often find difficult
_DIFFICULT_PATTERNS = [
    (r"\bth[aeiou]", "th- onset"),
    (r"[aeiou]th\b", "-th coda"),
    (r"\bwh", "wh- onset"),
    (r"[bcdfghjklmnpqrstvwxyz]{3}", "consonant cluster"),
    (r"tion\b", "-tion suffix"),
    (r"ture\b", "-ture suffix"),
    (r"\bl[aeiou]", "l- onset"),
    (r"\br[aeiou]", "r- onset"),
]


def _find_difficult_words(transcript: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    difficult: list[str] = []
    for word in set(words):
        for pattern, _ in _DIFFICULT_PATTERNS:
            if re.search(pattern, word):
                difficult.append(word)
                break
    return difficult


# ── Score computation ─────────────────────────────────────────────────────────

def compute(
    language_probability: float,
    speech_rate_wpm: float,
    clean_transcript: str,
) -> dict[str, Any]:
    """
    Compute speech clarity proxy.

    Signals used:
      language_probability  Whisper's confidence the speech is English (0–1).
                            A low value suggests heavy accent / low intelligibility.
      speech_rate_wpm       Extremely fast / slow speech correlates with
                            reduced clarity.
    """
    # Base score from language_probability
    if language_probability >= 0.95:
        base = 90.0
    elif language_probability >= 0.85:
        base = 80.0
    elif language_probability >= 0.70:
        base = 65.0
    elif language_probability >= 0.50:
        base = 55.0
    else:
        base = 45.0

    # Penalty for extreme speech rates
    rate_penalty = 0.0
    if speech_rate_wpm > 200 or (0 < speech_rate_wpm < 40):
        rate_penalty = 15.0
    elif speech_rate_wpm > 170 or (0 < speech_rate_wpm < 55):
        rate_penalty = 8.0

    score = max(0.0, base - rate_penalty)

    weak_words = _find_difficult_words(clean_transcript) if clean_transcript else []

    return {
        # Primary field used in fusion
        "speech_clarity_score": round(score, 1),
        # Alias kept for API compatibility
        "pronunciation_score": round(score, 1),
        "confidence_basis": "whisper_language_probability + speech_rate",
        "weak_words": weak_words,
        # Reserved for future GOP pipeline
        "phoneme_scores": [],
        "weak_phonemes": [],
    }
