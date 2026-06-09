"""
SDXL Text-to-Image Inference Demo
Generates child-appropriate cartoon-style images for the TikTalk project.
Runs on Apple Silicon (MPS) with memory optimization.
"""

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_pipeline():
    """Load SDXL pipeline with MPS + memory optimization."""
    print("Loading SDXL pipeline (this may take a few minutes on first run)...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Use DPM++ scheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Memory optimization for 16GB Mac
    pipe.enable_attention_slicing()
    # Move to MPS (Apple Silicon GPU)
    pipe = pipe.to("mps")

    print("Pipeline loaded successfully!")
    return pipe


def generate_image(pipe, prompt, negative_prompt=None, num_steps=25, seed=42):
    """Generate a single image from a text prompt."""
    if negative_prompt is None:
        negative_prompt = (
            "realistic, photo, complex background, dark, scary, violent, "
            "blurry, low quality, text, watermark, abstract"
        )

    generator = torch.Generator("mps").manual_seed(seed)

    start = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        generator=generator,
    ).images[0]
    elapsed = time.time() - start
    print(f"  Generated in {elapsed:.1f}s")
    return image


def main():
    pipe = load_pipeline()

    # Test prompts: child-appropriate scenes for oral English practice
    prompts = [
        "A cute cartoon cat sitting on a red chair in a sunny kitchen, simple illustration style, children's book art, bright colors, clean composition",
        "A happy cartoon boy playing with a yellow ball in a green park, simple illustration, flat colors, children's book style",
        "A cartoon family having breakfast at a table, simple and colorful illustration, cute style, children's picture book",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
        image = generate_image(pipe, prompt, seed=42 + i)
        out_path = OUTPUT_DIR / f"sdxl_demo_{i+1}.png"
        image.save(out_path)
        print(f"  Saved to: {out_path}")

    print(f"\nAll images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
