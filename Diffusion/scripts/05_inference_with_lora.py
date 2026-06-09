"""
SDXL Inference with trained LoRA adapter.
Generates images using the fine-tuned cartoon style.
"""

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from pathlib import Path
import time

LORA_DIR = Path(__file__).parent.parent / "lora_output" / "unet_lora"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_pipeline_with_lora():
    """Load SDXL + LoRA adapter."""
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load LoRA adapter
    if LORA_DIR.exists():
        print(f"Loading LoRA adapter from {LORA_DIR}...")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
        pipe.unet = pipe.unet.merge_and_unload()  # Merge for faster inference
        print("LoRA adapter loaded and merged!")
    else:
        print(f"WARNING: LoRA adapter not found at {LORA_DIR}")
        print("Running with base SDXL model only.")

    pipe.enable_attention_slicing()
    pipe = pipe.to("mps")
    return pipe


def main():
    pipe = load_pipeline_with_lora()

    # Prompts matching TikTalk use cases (PSLE oral exam scenarios)
    prompts = [
        "A cute cartoon cat sitting on a red chair in a sunny kitchen, simple illustration style, children's book art, bright colors",
        "Two cartoon children sharing an umbrella in the rain, simple colorful illustration, cute style",
        "A cartoon family at a picnic in the park, simple illustration, bright cheerful colors, children's book style",
        "A cartoon boy helping an old lady cross the street, simple illustration, warm colors, children's picture book",
        "A cartoon classroom with happy children raising their hands, simple colorful illustration style",
    ]

    negative_prompt = (
        "realistic, photo, complex background, dark, scary, violent, "
        "blurry, low quality, text, watermark, abstract, nsfw"
    )

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
        generator = torch.Generator("mps").manual_seed(42 + i)

        start = time.time()
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]
        elapsed = time.time() - start

        out_path = OUTPUT_DIR / f"lora_result_{i+1}.png"
        image.save(out_path)
        print(f"  Generated in {elapsed:.1f}s -> {out_path}")

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
