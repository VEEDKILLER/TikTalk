"""
Prepare a small cartoon-style dataset for LoRA fine-tuning.
Downloads a subset from HuggingFace and creates caption files.
"""

import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data" / "cartoon_lora"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_cute_style_dataset(num_images=50):
    """Download a small subset of cartoon/cute style images from HuggingFace."""
    print("Downloading cute-style dataset from HuggingFace...")

    try:
        ds = load_dataset("bunyi/cute-style", split="train", streaming=True)
        count = 0
        metadata = []

        for item in ds:
            if count >= num_images:
                break

            img = item.get("image")
            caption = item.get("text", "") or item.get("caption", "")

            if img is None:
                continue

            # Resize to 1024x1024 for SDXL
            img = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
            img_name = f"img_{count:04d}.png"
            img_path = DATA_DIR / img_name
            img.save(img_path)

            # Add style prefix to caption for LoRA training
            if caption:
                caption = f"cute cartoon illustration, {caption}"
            else:
                caption = "cute cartoon illustration, colorful children's book style"

            metadata.append({"file_name": img_name, "text": caption})
            count += 1
            if count % 10 == 0:
                print(f"  Downloaded {count}/{num_images} images...")

        # Save metadata
        meta_path = DATA_DIR / "metadata.jsonl"
        with open(meta_path, "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

        print(f"Dataset ready: {count} images in {DATA_DIR}")
        print(f"Metadata saved to {meta_path}")
        return count

    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Falling back to synthetic dataset generation...")
        return generate_synthetic_dataset(num_images)


def generate_synthetic_dataset(num_images=30):
    """
    Generate a synthetic dataset using solid-color placeholder images
    with appropriate captions. These will be replaced with real images later.
    """
    print("Generating synthetic placeholder dataset...")

    captions = [
        "cute cartoon cat sitting on a chair, simple illustration, bright colors",
        "cartoon dog running in a park, children's book style, flat colors",
        "happy cartoon children playing together, colorful illustration",
        "cartoon bear eating honey, cute style, simple background",
        "cartoon bird flying over a rainbow, children's illustration",
        "cartoon fish swimming in the sea, cute underwater scene",
        "cartoon rabbit reading a book, cozy illustration style",
        "cartoon elephant spraying water, playful children's art",
        "cartoon family at the beach, simple colorful illustration",
        "cartoon boy riding a bicycle, happy scene, flat illustration",
        "cartoon girl drawing with crayons, classroom scene, cute style",
        "cartoon panda eating bamboo, simple kawaii illustration",
        "cartoon lion in the jungle, friendly children's book art",
        "cartoon penguin on ice, winter scene, cute illustration",
        "cartoon monkey swinging on a tree, playful cartoon style",
        "cartoon owl reading at night, cozy illustration, stars",
        "cartoon frog on a lily pad, pond scene, children's art",
        "cartoon butterfly in a garden, colorful flowers, simple style",
        "cartoon boy and dog walking, sunny day, children's illustration",
        "cartoon girl with an umbrella in the rain, cute pastel style",
        "cartoon train going through mountains, adventure illustration",
        "cartoon spaceship flying to the moon, fun children's art",
        "cartoon baker making cookies, kitchen scene, warm colors",
        "cartoon doctor with a stethoscope, friendly illustration",
        "cartoon firefighter with a truck, action scene, bold colors",
        "cartoon teacher at a blackboard, classroom illustration",
        "cartoon farmer with animals, farm scene, simple composition",
        "cartoon chef cooking soup, kitchen illustration, cute style",
        "cartoon astronaut floating in space, adventure illustration",
        "cartoon pirate on a ship, ocean adventure, fun style",
    ]

    metadata = []
    for i in range(min(num_images, len(captions))):
        # Create a simple colored placeholder image
        import random
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img = Image.new("RGB", (1024, 1024), color)
        img_name = f"img_{i:04d}.png"
        img.save(DATA_DIR / img_name)
        metadata.append({"file_name": img_name, "text": captions[i]})

    meta_path = DATA_DIR / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Synthetic dataset created: {len(metadata)} images in {DATA_DIR}")
    print("NOTE: Replace placeholder images with real cartoon images for better results.")
    return len(metadata)


if __name__ == "__main__":
    count = download_cute_style_dataset(num_images=50)
    if count == 0:
        generate_synthetic_dataset(30)
