"""
Grammar scorer.

LLM role  : Extract grammar errors from the clean transcript.
Rule role : Map error types + counts to a grammar_score.

Keeping scoring logic separate from LLM means the score won't drift
when prompts are tuned or the model is swapped.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

# ── Error type weights (penalty per occurrence) ───────────────────────────────
_ERROR_WEIGHTS: dict[str, float] = {
    "subject-verb agreement": 9.0,
    "tense":                  8.0,
    "missing word":           7.0,
    "word order":             7.0,
    "plural":                 5.0,
    "article":                4.0,
    "preposition":            4.0,
    "pronoun":                5.0,
    "other":                  3.0,
}

_SEVERITY_MULTIPLIER = {
    "major":  1.5,
    "minor":  0.7,
    "trivial": 0.3,
}

_GRAMMAR_ERROR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["grammar_errors"],
    "properties": {
        "grammar_errors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "correction", "error_type", "severity"],
                "properties": {
                    "text":       {"type": "string"},
                    "correction": {"type": "string"},
                    "error_type": {"type": "string"},
                    "severity":   {"type": "string", "enum": ["major", "minor", "trivial"]},
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """You are a grammar error detector for English learners aged 6–12 in Singapore.

Your ONLY job is to identify grammar errors in the child's response.
Do NOT score or give feedback — only list errors.

Focus on these error types:
- subject-verb agreement  (he go → he goes)
- tense                   (I go yesterday → I went yesterday)
- plural                  (two cat → two cats)
- article                 (a orange → an orange)
- preposition             (go to home → go home)
- missing word            (she eating → she is eating)
- word order              (very is big → is very big)
- pronoun                 (me want → I want)

Severity guide:
- major:   errors that make the sentence hard to understand
- minor:   errors that are noticeable but meaning is still clear
- trivial: very minor stylistic issues

Be lenient and age-appropriate. A 6-year-old saying "me want" is minor,
not major. Only flag clear grammatical mistakes, not vocabulary choices."""


async def extract_errors(clean_transcript: str) -> list[dict[str, Any]]:
    """Call LLM to extract grammar errors. Returns a list of error dicts."""
    if not clean_transcript.strip():
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.getenv("EVALUATE_MODEL", "gpt-4o-mini")
    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f'Child said: "{clean_transcript}"\n\nList all grammar errors.'},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grammar_errors",
                "strict": True,
                "schema": _GRAMMAR_ERROR_SCHEMA,
            },
        },
        temperature=0.1,
    )

    raw = response.choices[0].message.content or '{"grammar_errors": []}'
    data = json.loads(raw)
    return data.get("grammar_errors", [])


# ── Rule-based scoring ────────────────────────────────────────────────────────

def compute_score(grammar_errors: list[dict[str, Any]]) -> float:
    """Map extracted errors → grammar_score using fixed weights."""
    if not grammar_errors:
        return 100.0

    total_penalty = 0.0
    for err in grammar_errors:
        etype = err.get("error_type", "other").lower()
        severity = err.get("severity", "minor").lower()

        # Find base weight
        base = _ERROR_WEIGHTS.get("other", 3.0)
        for key, w in _ERROR_WEIGHTS.items():
            if key in etype:
                base = w
                break

        mult = _SEVERITY_MULTIPLIER.get(severity, 1.0)
        total_penalty += base * mult

    score = max(0.0, 100.0 - total_penalty)
    return round(score, 1)


async def score(clean_transcript: str) -> dict[str, Any]:
    """Full grammar scoring: extract errors then compute score."""
    errors = await extract_errors(clean_transcript)
    grammar_score = compute_score(errors)
    return {
        "grammar_score": grammar_score,
        "grammar_errors": errors,
        "error_count": len(errors),
    }
