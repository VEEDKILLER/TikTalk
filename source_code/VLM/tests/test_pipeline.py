from __future__ import annotations

import json
from pathlib import Path

from tiktalk_vlm.pipeline import GroundTruthPipeline, PipelineConfig


VALID_QWEN = {
    "scene_summary": "A girl is reading a book in a park.",
    "detailed_description": "A girl sits on the grass and reads a book. Trees are behind her. The scene looks calm and bright.",
    "main_objects": [
        {"name": "girl", "count": 1, "attributes": ["young", "sitting"]},
        {"name": "book", "count": 1, "attributes": ["open"]},
    ],
    "actions": ["reading", "sitting"],
    "setting": "a park",
    "visible_text": [],
    "teaching_focus_words": ["girl", "book", "park"],
    "age_level": "junior_learners",
}

VALID_OPENAI = {
    "scene_summary": "A young girl reads a book on the grass in a park.",
    "detailed_description": "A young girl is sitting on the grass with an open book. Green trees are in the background. The picture is bright and peaceful.",
    "main_objects": [
        {"name": "girl", "count": 1, "attributes": ["young", "on the grass"]},
        {"name": "book", "count": 1, "attributes": ["open"]},
        {"name": "trees", "count": 2, "attributes": ["green"]},
    ],
    "actions": ["reading", "sitting"],
    "setting": "a park with grass and trees",
    "visible_text": [],
    "teaching_focus_words": ["girl", "book", "grass", "trees"],
    "age_level": "junior_learners",
}


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, responses: list[str], *, fail_first_structured: bool = False) -> None:
        self._responses = list(responses)
        self._fail_first_structured = fail_first_structured
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_first_structured and "response_format" in kwargs:
            self._fail_first_structured = False
            raise RuntimeError("structured output unsupported")
        return FakeChatResponse(self._responses.pop(0))


class FakeOpenAIClient:
    def __init__(self, responses: list[str], *, fail_first_structured: bool = False) -> None:
        self.chat = type(
            "ChatNamespace",
            (),
            {"completions": FakeChatCompletions(responses, fail_first_structured=fail_first_structured)},
        )()


class FakeGeminiModels:
    def __init__(self, winner: str) -> None:
        self.winner = winner
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "FakeGeminiResponse",
            (),
            {"text": json.dumps({"winner": self.winner, "reason": "Better accuracy."})},
        )()


class FakeGeminiClient:
    def __init__(self, winner: str) -> None:
        self.models = FakeGeminiModels(winner)


def test_generate_ground_truth_returns_gemini_selected_candidate(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image-bytes")

    pipeline = GroundTruthPipeline(
        config=PipelineConfig(),
        qwen_client=FakeOpenAIClient([json.dumps(VALID_QWEN)]),
        openai_client=FakeOpenAIClient([json.dumps(VALID_OPENAI)]),
        gemini_client=FakeGeminiClient("openai"),
    )

    result = pipeline.generate_ground_truth(image_path)

    assert result == VALID_OPENAI


def test_generate_ground_truth_with_trace_includes_winner_and_candidates(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image-bytes")

    pipeline = GroundTruthPipeline(
        config=PipelineConfig(),
        qwen_client=FakeOpenAIClient([json.dumps(VALID_QWEN)]),
        openai_client=FakeOpenAIClient([json.dumps(VALID_OPENAI)]),
        gemini_client=FakeGeminiClient("openai"),
    )

    trace = pipeline.generate_ground_truth_with_trace(image_path)

    assert trace["chosen_provider"] == "openai"
    assert trace["final_result"] == VALID_OPENAI
    assert trace["candidates"]["qwen"] == VALID_QWEN
    assert trace["candidates"]["openai"] == VALID_OPENAI


def test_qwen_falls_back_to_prompt_only_json_when_structured_output_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image-bytes")

    qwen_client = FakeOpenAIClient(
        [json.dumps(VALID_QWEN)],
        fail_first_structured=True,
    )
    openai_client = FakeOpenAIClient([json.dumps(VALID_OPENAI)])
    gemini_client = FakeGeminiClient("qwen")

    pipeline = GroundTruthPipeline(
        config=PipelineConfig(),
        qwen_client=qwen_client,
        openai_client=openai_client,
        gemini_client=gemini_client,
    )

    result = pipeline.generate_ground_truth(image_path)

    assert result == VALID_QWEN
    qwen_calls = qwen_client.chat.completions.calls
    assert len(qwen_calls) == 2
    assert "response_format" in qwen_calls[0]
    assert "response_format" not in qwen_calls[1]
