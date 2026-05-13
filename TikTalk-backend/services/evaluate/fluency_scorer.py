"""
Fluency scorer — computed from ASR segment timing, no LLM.

Features extracted:
  speech_rate_wpm        Words per minute of actual speech time
  articulation_rate_wpm  Words per minute of total recording duration
  pause_count            Inter-segment gaps > 0.5 s
  pause_ratio            Silent time / total duration
  avg_pause_duration     Mean duration of detected pauses (s)
  utterance_duration     Total recording duration (s)
  speech_time            Total voiced time (s)
"""
from __future__ import annotations

from typing import Any

_MIN_PAUSE_GAP = 0.5       # seconds — shorter gaps are fluent micro-pauses
_TARGET_WPM_LOW = 70.0     # comfortable range for children
_TARGET_WPM_HIGH = 150.0


def compute(
    segments: list[dict],
    total_duration: float,
    clean_transcript: str,
) -> dict[str, Any]:
    if not segments or total_duration <= 0:
        return _zero_result(total_duration)

    speech_time = sum(s["end"] - s["start"] for s in segments)
    if speech_time <= 0:
        return _zero_result(total_duration)

    silence_time = max(0.0, total_duration - speech_time)
    pause_ratio = silence_time / total_duration

    # Detect pauses between consecutive segments
    pauses: list[float] = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap > _MIN_PAUSE_GAP:
            pauses.append(gap)

    pause_count = len(pauses)
    avg_pause = (sum(pauses) / pause_count) if pauses else 0.0

    word_count = len(clean_transcript.split()) if clean_transcript.strip() else 0
    speech_rate_wpm = (word_count / speech_time * 60.0) if speech_time > 0 else 0.0
    articulation_rate_wpm = (word_count / total_duration * 60.0) if total_duration > 0 else 0.0

    # ── Score ────────────────────────────────────────────────────────────────
    # Rate component
    if _TARGET_WPM_LOW <= speech_rate_wpm <= _TARGET_WPM_HIGH:
        rate_score = 100.0
    elif 50 <= speech_rate_wpm < _TARGET_WPM_LOW or _TARGET_WPM_HIGH < speech_rate_wpm <= 180:
        rate_score = 80.0
    elif 30 <= speech_rate_wpm < 50 or 180 < speech_rate_wpm <= 220:
        rate_score = 65.0
    else:
        rate_score = 50.0

    # Pause penalty
    if pause_ratio > 0.60:
        pause_penalty = 40.0
    elif pause_ratio > 0.45:
        pause_penalty = 28.0
    elif pause_ratio > 0.30:
        pause_penalty = 15.0
    elif pause_ratio > 0.20:
        pause_penalty = 6.0
    else:
        pause_penalty = 0.0

    fluency_score = max(0.0, rate_score - pause_penalty)

    # Prosody alerts for feedback
    alerts: list[str] = []
    if pause_ratio > 0.40:
        alerts.append("many_pauses")
    if pause_count >= 5:
        alerts.append("frequent_hesitation")
    if speech_rate_wpm < 50:
        alerts.append("very_slow")
    if speech_rate_wpm > 180:
        alerts.append("very_fast")

    return {
        "fluency_score": round(fluency_score, 1),
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "articulation_rate_wpm": round(articulation_rate_wpm, 1),
        "pause_count": pause_count,
        "pause_ratio": round(pause_ratio, 3),
        "avg_pause_duration": round(avg_pause, 2),
        "utterance_duration": round(total_duration, 2),
        "speech_time": round(speech_time, 2),
        "prosody_alerts": alerts,
    }


def _zero_result(duration: float) -> dict[str, Any]:
    return {
        "fluency_score": 0.0,
        "speech_rate_wpm": 0.0,
        "articulation_rate_wpm": 0.0,
        "pause_count": 0,
        "pause_ratio": 1.0,
        "avg_pause_duration": 0.0,
        "utterance_duration": round(duration, 2),
        "speech_time": 0.0,
        "prosody_alerts": ["no_speech"],
    }
