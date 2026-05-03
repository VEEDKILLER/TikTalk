from __future__ import annotations

import base64
import concurrent.futures
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

from .schemas import GROUND_TRUTH_JSON_SCHEMA, JUDGE_JSON_SCHEMA


GENERATION_SYSTEM_PROMPT = """
# Role

You are an experienced children's English teacher. Your task is to observe the image and generate a **simple, natural reference text (Ground Truth) suitable for non-native learners aged 6-12**.



# Task Objective

Generate descriptive text as a standard reference for evaluating a child's spoken response. Focus on the most direct and basic elements visible in the image.



# Writing Rules (Junior Level)

1. **Vocabulary limit**: Use only basic words (CEFR A1-A2 level). For example, use "big" instead of "enormous", and "happy" instead of "cheerful".

2. **Simple sentence patterns**: Prefer "subject + verb + object" and "There is/are" structures. Avoid long complex sentences, subordinate clauses, and passive voice.

3. **Key dimensions**:

    - **Who/What**: Who or what appears in the image (boy, girl, cat, teacher).

    - **Color/Number**: Colors and quantities (two red apples).

    - **Action**: What they are doing (running, eating, sleeping).

    - **Location**: Where it happens (at school, in the park).

4. **Diverse anchor expressions**: Keep language simple, but provide 2-3 equivalent expressions that a child might say at the same level (separated by semicolons) to improve BERTScore tolerance.



# Output Format (JSON for easy parsing)

{

    "scene_summary": "One very simple summary sentence.",

    "detailed_description": "Two to four short factual sentences.",

    "main_objects": [{"name": "girl", "count": 1, "attributes": ["red shirt"]}],

    "actions": ["playing with a ball"],

    "setting": "park",

    "visible_text": [],

    "teaching_focus_words": ["girl", "ball", "park", "play", "red"],

    "age_level": "junior_learners"

}
"""


GENERATION_USER_PROMPT = """Describe this image for junior learners and return one JSON object that follows the schema exactly.
Focus on visible objects, actions, setting, and a simple teaching-oriented description.
"""


JUDGE_PROMPT_TEMPLATE = """You are the final judge for TikTalk image ground truth.

You will receive:
- the original image
- candidate JSON from Qwen
- candidate JSON from OpenAI

Choose the better answer using these rules, in order:
1. Accuracy to the actual image
2. No hallucinated details
3. Age-appropriate, simple English for junior learners
4. Better coverage of visible objects, actions, and setting
5. Better vocabulary choices for English learning

Return JSON only and follow the schema exactly.

Qwen candidate:
{qwen_candidate}

OpenAI candidate:
{openai_candidate}
"""


class PipelineError(RuntimeError):
    """Raised when the VLM pipeline cannot produce a valid ground truth."""


@dataclass(slots=True)
class PipelineConfig:
    openai_api_key: str | None = None
    dashscope_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_model: str = "gpt-4o"
    qwen_model: str = "qwen-vl-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    gemini_model: str = "gemini-1.5-flash"
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        if load_dotenv is not None:
            load_dotenv()

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openai_model=os.getenv("OPENAI_VLM_MODEL", "gpt-4o"),
            qwen_model=os.getenv("QWEN_VLM_MODEL", "qwen-vl-plus"),
            qwen_base_url=os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            gemini_model=os.getenv("GEMINI_JUDGE_MODEL", "gemini-1.5-flash"),
            request_timeout_seconds=float(os.getenv("VLM_REQUEST_TIMEOUT_SECONDS", "60")),
        )


class GroundTruthPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        openai_client: Any | None = None,
        qwen_client: Any | None = None,
        gemini_client: Any | None = None,
    ) -> None:
        self.config = config or PipelineConfig.from_env()
        self._openai_client = openai_client
        self._qwen_client = qwen_client
        self._gemini_client = gemini_client

    def generate_ground_truth(self, image_path: str | Path) -> dict[str, Any]:
        trace = self.generate_ground_truth_with_trace(image_path)
        return trace["final_result"]

    def generate_ground_truth_with_trace(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        image_bytes = path.read_bytes()
        mime_type = _guess_mime_type(path)
        image_data_url = _to_data_url(image_bytes, mime_type)

        candidates = self._generate_candidates(image_data_url)

        if not candidates:
            raise PipelineError("Both VLM providers failed to produce a valid JSON candidate.")

        if len(candidates) == 1:
            chosen_provider = next(iter(candidates.keys()))
            return {
                "chosen_provider": chosen_provider,
                "candidates": candidates,
                "final_result": candidates[chosen_provider],
            }

        winner = self._judge_candidates(
            image_bytes=image_bytes,
            mime_type=mime_type,
            qwen_candidate=candidates["qwen"],
            openai_candidate=candidates["openai"],
        )
        return {
            "chosen_provider": winner,
            "candidates": candidates,
            "final_result": candidates[winner],
        }

    def _generate_candidates(self, image_data_url: str) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_map = {
                executor.submit(self._generate_with_qwen, image_data_url): "qwen",
                executor.submit(self._generate_with_openai, image_data_url): "openai",
            }
            for future, provider in future_map.items():
                try:
                    results[provider] = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures[provider] = str(exc)

        if failures and not results:
            failure_text = "; ".join(f"{provider}: {message}" for provider, message in failures.items())
            raise PipelineError(f"Candidate generation failed for all providers: {failure_text}")

        return results

    def _generate_with_qwen(self, image_data_url: str) -> dict[str, Any]:
        client = self._get_qwen_client()
        content = _call_openai_chat_completion(
            client=client,
            model=self.config.qwen_model,
            image_data_url=image_data_url,
            use_structured_output=True,
        )
        payload = _parse_json_payload(content)
        return _validate_ground_truth_payload(payload)

    def _generate_with_openai(self, image_data_url: str) -> dict[str, Any]:
        client = self._get_openai_client()
        content = _call_openai_chat_completion(
            client=client,
            model=self.config.openai_model,
            image_data_url=image_data_url,
            use_structured_output=True,
        )
        payload = _parse_json_payload(content)
        return _validate_ground_truth_payload(payload)

    def _judge_candidates(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        qwen_candidate: dict[str, Any],
        openai_candidate: dict[str, Any],
    ) -> str:
        client = self._get_gemini_client()

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            qwen_candidate=json.dumps(qwen_candidate, ensure_ascii=True, indent=2),
            openai_candidate=json.dumps(openai_candidate, ensure_ascii=True, indent=2),
        )

        response = client.models.generate_content(
            model=self.config.gemini_model,
            contents=[
                _build_gemini_image_part(image_bytes=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": JUDGE_JSON_SCHEMA,
            },
        )

        decision = _parse_json_payload(response.text)
        winner = decision.get("winner")
        if winner not in {"qwen", "openai"}:
            raise PipelineError(f"Gemini returned an invalid winner: {winner!r}")
        return winner

    def _get_openai_client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client

        if not self.config.openai_api_key:
            raise PipelineError("OPENAI_API_KEY is required for the OpenAI VLM branch.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError("openai package is required.") from exc

        self._openai_client = OpenAI(
            api_key=self.config.openai_api_key,
            timeout=self.config.request_timeout_seconds,
        )
        return self._openai_client

    def _get_qwen_client(self) -> Any:
        if self._qwen_client is not None:
            return self._qwen_client

        if not self.config.dashscope_api_key:
            raise PipelineError("DASHSCOPE_API_KEY is required for the Qwen VLM branch.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError("openai package is required.") from exc

        self._qwen_client = OpenAI(
            api_key=self.config.dashscope_api_key,
            base_url=self.config.qwen_base_url,
            timeout=self.config.request_timeout_seconds,
        )
        return self._qwen_client

    def _get_gemini_client(self) -> Any:
        if self._gemini_client is not None:
            return self._gemini_client

        if not self.config.gemini_api_key:
            raise PipelineError("GEMINI_API_KEY is required for the Gemini judge.")

        try:
            from google import genai
        except ImportError as exc:
            raise PipelineError("google-genai package is required.") from exc

        self._gemini_client = genai.Client(api_key=self.config.gemini_api_key)
        return self._gemini_client


def generate_ground_truth(
    image_path: str | Path,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    return GroundTruthPipeline(config=config).generate_ground_truth(image_path)


def generate_ground_truth_with_trace(
    image_path: str | Path,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    return GroundTruthPipeline(config=config).generate_ground_truth_with_trace(image_path)


def _call_openai_chat_completion(
    *,
    client: Any,
    model: str,
    image_data_url: str,
    use_structured_output: bool,
) -> str:
    base_request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GENERATION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    if use_structured_output:
        structured_request = dict(base_request)
        structured_request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "tiktalk_ground_truth",
                "strict": True,
                "schema": GROUND_TRUTH_JSON_SCHEMA,
            },
        }
        try:
            response = client.chat.completions.create(**structured_request)
            return _extract_message_content(response)
        except Exception:  # noqa: BLE001
            pass

    fallback_request = dict(base_request)
    fallback_request["messages"] = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        GENERATION_USER_PROMPT
                        + "\nReturn raw JSON only. Do not use markdown fences."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]
    response = client.chat.completions.create(**fallback_request)
    return _extract_message_content(response)


def _extract_message_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise PipelineError("Provider response did not contain a usable message content field.") from exc

    if not isinstance(content, str) or not content.strip():
        raise PipelineError("Provider returned empty content.")
    return content


def _parse_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = _strip_markdown_fence(candidate)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = json.loads(_extract_first_json_object(candidate))

    if not isinstance(payload, dict):
        raise PipelineError("Model output must be a JSON object.")
    return payload


def _validate_ground_truth_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_top_level = [
        "scene_summary",
        "detailed_description",
        "main_objects",
        "actions",
        "setting",
        "visible_text",
        "teaching_focus_words",
        "age_level",
    ]
    for key in required_top_level:
        if key not in payload:
            raise PipelineError(f"Ground truth JSON is missing required field: {key}")

    if payload["age_level"] != "junior_learners":
        raise PipelineError("age_level must be 'junior_learners'.")

    for key in ("scene_summary", "detailed_description", "setting"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise PipelineError(f"{key} must be a non-empty string.")
        payload[key] = payload[key].strip()

    for key in ("actions", "visible_text", "teaching_focus_words"):
        payload[key] = _validate_string_list(payload[key], field_name=key)

    main_objects = payload["main_objects"]
    if not isinstance(main_objects, list) or not main_objects:
        raise PipelineError("main_objects must be a non-empty list.")

    normalized_objects: list[dict[str, Any]] = []
    for index, item in enumerate(main_objects):
        if not isinstance(item, dict):
            raise PipelineError(f"main_objects[{index}] must be an object.")
        name = item.get("name")
        count = item.get("count")
        attributes = item.get("attributes")
        if not isinstance(name, str) or not name.strip():
            raise PipelineError(f"main_objects[{index}].name must be a non-empty string.")
        if not isinstance(count, int) or count < 1:
            raise PipelineError(f"main_objects[{index}].count must be an integer >= 1.")
        normalized_objects.append(
            {
                "name": name.strip(),
                "count": count,
                "attributes": _validate_string_list(attributes, field_name=f"main_objects[{index}].attributes"),
            }
        )
    payload["main_objects"] = normalized_objects

    if not 3 <= len(payload["teaching_focus_words"]) <= 8:
        raise PipelineError("teaching_focus_words must contain between 3 and 8 items.")

    return payload


def _validate_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise PipelineError(f"{field_name} must be a list of strings.")

    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise PipelineError(f"{field_name}[{index}] must be a non-empty string.")
        normalized.append(item.strip())
    return normalized


def _strip_markdown_fence(value: str) -> str:
    lines = value.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(value: str) -> str:
    start = value.find("{")
    if start == -1:
        raise PipelineError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]

    raise PipelineError("Unterminated JSON object in model output.")


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/png"


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_gemini_image_part(*, image_bytes: bytes, mime_type: str) -> Any:
    try:
        from google.genai import types
    except ImportError:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
