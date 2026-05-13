"""
Transcript normalizer — multi-view output.

Views produced:
  raw_transcript       Original ASR text, untouched.
  clean_transcript     Fillers removed, consecutive duplicates collapsed.
  disfluency_spans     Character-level spans of detected disfluencies.
  low_confidence_spans Reserved for ASR token-confidence data (empty for now;
                       ASR endpoint does not yet return per-token confidence).
"""
from __future__ import annotations

import re
from typing import Any

# ── Filler word list ──────────────────────────────────────────────────────────
_FILLERS = {
    "um", "uh", "er", "hmm", "hm", "ah", "eh",
    "mm", "umm", "uhh", "err", "ohh",
}

_FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _FILLERS) + r"+)\b",
    re.IGNORECASE,
)


def normalize(raw: str) -> dict[str, Any]:
    """Return a dict with all transcript views."""
    raw = raw.strip()

    disfluency_spans: list[dict] = []

    # ── Pass 1: tag filler spans (on raw text, before any modification) ───────
    for m in _FILLER_RE.finditer(raw):
        disfluency_spans.append({
            "start_char": m.start(),
            "end_char": m.end(),
            "text": m.group(),
            "type": "filler",
        })

    # ── Pass 2: remove fillers ────────────────────────────────────────────────
    no_fillers = _FILLER_RE.sub(" ", raw)

    # ── Pass 3: collapse consecutive duplicate words ──────────────────────────
    words = no_fillers.split()
    deduped: list[str] = []
    i = 0
    while i < len(words):
        deduped.append(words[i])
        # Find run of duplicates
        run_start = i
        j = i + 1
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        if j - run_start > 1:
            # Record as a repetition disfluency on the raw text
            # (approximate position; exact char positions would need alignment)
            disfluency_spans.append({
                "start_char": None,
                "end_char": None,
                "text": " ".join(words[run_start:j]),
                "type": "repetition",
            })
        i = j

    clean = " ".join(deduped).strip()
    clean = re.sub(r" {2,}", " ", clean)
    if clean:
        clean = clean[0].upper() + clean[1:]

    return {
        "raw_transcript": raw,
        "clean_transcript": clean,
        "disfluency_spans": disfluency_spans,
        # Placeholder: ASR does not yet return per-token confidence
        "low_confidence_spans": [],
    }
