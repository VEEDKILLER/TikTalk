"""
Diffusion service — wraps TikTalk-diffusion logic to generate DALL-E 3 images.

Images are saved to a temp directory keyed by image_id so the VLM service
can read them by ID in a subsequent request.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from tiktalk_diffusion.prompt_templates import (
    CHARACTERS,
    SCENARIO_TEMPLATES,
    build_prompt,
    pick_template,
)

# ── Temp image storage ────────────────────────────────────────────────────────
TEMP_DIR = Path(os.getenv("TIKTALK_IMAGE_DIR", "/tmp/tiktalk-images"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_image(prompt: str, apply_style: bool = True) -> dict[str, Any]:
    """Generate one image with DALL-E 3 and save it to the temp directory.

    Returns:
        {
            "image_id": str,       # use this in subsequent /session/vlm calls
            "image_base64": str,   # base64-encoded PNG for the frontend
            "prompt_used": str,
        }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    full_prompt = build_prompt(prompt) if apply_style else prompt

    client = AsyncOpenAI(api_key=api_key)
    response = await client.images.generate(
        model="dall-e-3",
        prompt=full_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    async with httpx.AsyncClient(timeout=60.0) as http:
        img_resp = await http.get(image_url)
        img_resp.raise_for_status()
    img_bytes = img_resp.content

    image_id = f"{int(time.time() * 1000)}"
    image_path = TEMP_DIR / f"{image_id}.png"
    image_path.write_bytes(img_bytes)

    return {
        "image_id": image_id,
        "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
        "prompt_used": full_prompt,
    }


def resolve_prompt(
    prompt: str | None,
    category: str | None,
    character: str,
) -> str:
    """Build a prompt string from either free-form text or a category template.

    When category is given, the action is chosen automatically from
    CATEGORY_ACTIONS so the scene is always semantically coherent.
    """
    if prompt:
        return prompt
    if category:
        return pick_template(category, character)
    raise ValueError("Either 'prompt' or 'category' must be provided.")


def get_image_path(image_id: str) -> Path:
    """Resolve an image_id to its file path. Raises FileNotFoundError if absent."""
    path = TEMP_DIR / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"Image '{image_id}' not found. "
            "It may have been cleaned up or the image_id is incorrect."
        )
    return path


def list_options() -> dict[str, Any]:
    return {
        "categories": list(SCENARIO_TEMPLATES.keys()),
        "characters": CHARACTERS,
    }
