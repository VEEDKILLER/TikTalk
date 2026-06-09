"""
Quick Feasibility Test using SDXL-Turbo
Much faster than full SDXL: only 4 inference steps, smaller model.
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading SDXL-Turbo (lightweight, ~3.5GB)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,  # float32 for MPS compatibility (float16 causes black images)
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to("mps")
    print("Model loaded!\n")

    prompts = [
        "A cute cartoon cat sitting in a kitchen, children's book illustration, bright colors, simple",
        "Happy cartoon children playing in a park, colorful flat illustration, cute style",
        "A cartoon family eating dinner together, warm illustration, children's book style",
        "A cartoon boy riding a bicycle on a sunny day, simple colorful illustration",
        "A cartoon classroom with kids raising hands, cheerful illustration style",
    ]

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")
        gen = torch.Generator("mps").manual_seed(42 + i)
        start = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,      # Turbo only needs 1-4 steps!
            guidance_scale=0.0,          # Turbo works without CFG
            width=512,
            height=512,
            generator=gen,
        ).images[0]
        elapsed = time.time() - start

        out_path = OUTPUT_DIR / f"turbo_demo_{i+1}.png"
        image.save(out_path)
        print(f"  {elapsed:.1f}s -> {out_path}")

    print(f"\nDone! Check {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
