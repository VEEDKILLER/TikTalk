from __future__ import annotations

from typing import Any


GROUND_TRUTH_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scene_summary": {
            "type": "string",
            "description": "One short factual sentence in simple English for junior learners.",
        },
        "detailed_description": {
            "type": "string",
            "description": "Two to four short factual sentences. Do not invent hidden details.",
        },
        "main_objects": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Common noun for a clearly visible object or character.",
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many of this object are visible.",
                    },
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short visible attributes such as colors, size, or clothing.",
                    },
                },
                "required": ["name", "count", "attributes"],
            },
        },
        "actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Observable actions only.",
        },
        "setting": {
            "type": "string",
            "description": "Short phrase describing the place or scene.",
        },
        "visible_text": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any readable text present in the image. Use an empty array if none is visible.",
        },
        "teaching_focus_words": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {"type": "string"},
            "description": "Simple vocabulary words drawn from the visible scene.",
        },
        "age_level": {
            "type": "string",
            "enum": ["junior_learners"],
            "description": "Fixed label showing this JSON was written for junior learners.",
        },
    },
    "required": [
        "scene_summary",
        "detailed_description",
        "main_objects",
        "actions",
        "setting",
        "visible_text",
        "teaching_focus_words",
        "age_level",
    ],
}


JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "winner": {
            "type": "string",
            "enum": ["qwen", "openai"],
            "description": "Which candidate JSON should become the ground truth.",
        },
        "reason": {
            "type": "string",
            "description": "Short explanation focused on accuracy and age appropriateness.",
        },
    },
    "required": ["winner", "reason"],
}
