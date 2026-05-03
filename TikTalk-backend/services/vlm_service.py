"""
VLM service — wraps the TikTalk-VLM GroundTruthPipeline.

After analysis, the ground truth is persisted alongside the image so that
/session/evaluate can load it by image_id without the client re-sending it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from tiktalk_vlm.pipeline import GroundTruthPipeline, PipelineConfig
from .diffusion_service import TEMP_DIR

# ── Singleton pipeline (lazy init) ───────────────────────────────────────────
_pipeline: GroundTruthPipeline | None = None


def _get_pipeline() -> GroundTruthPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GroundTruthPipeline(PipelineConfig.from_env())
    return _pipeline


def _gt_path(image_id: str) -> Path:
    return TEMP_DIR / f"{image_id}_gt.json"


def load_ground_truth(image_id: str) -> dict[str, Any]:
    """Load a previously saved ground truth by image_id."""
    path = _gt_path(image_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth for image '{image_id}' not found. "
            "Call /session/vlm first."
        )
    return json.loads(path.read_text())


# ── Public API ────────────────────────────────────────────────────────────────

async def analyze_image(image_path: str | Path) -> dict[str, Any]:
    """Run VLM analysis on a saved image, persist ground truth, return result.

    Returns:
        {
            "ground_truth": { scene_summary, detailed_description, main_objects,
                              actions, setting, visible_text, teaching_focus_words,
                              age_level },
            "vlm_provider": "openai" | "qwen",
        }
    """
    path = Path(image_path)
    loop = asyncio.get_event_loop()
    pipeline = _get_pipeline()

    trace = await loop.run_in_executor(
        None,
        pipeline.generate_ground_truth_with_trace,
        path,
    )

    ground_truth = trace["final_result"]

    # Persist so /session/evaluate can load it by image_id
    image_id = path.stem  # filename without extension = image_id
    _gt_path(image_id).write_text(json.dumps(ground_truth, ensure_ascii=False))

    return {
        "ground_truth": ground_truth,
        "vlm_provider": trace["chosen_provider"],
    }
