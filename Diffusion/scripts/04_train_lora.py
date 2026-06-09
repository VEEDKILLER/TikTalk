"""
LoRA Fine-tuning for SDXL on cartoon/children's illustration style.
Optimized for Apple Silicon (M4, 16GB RAM).

Usage:
    python 03_train_lora.py

This trains a LoRA adapter on the prepared dataset to bias SDXL
toward generating simple, child-friendly cartoon illustrations.
"""

import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer
import torch.nn.functional as F

# ─── Config ───
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR = Path(__file__).parent.parent / "data" / "cartoon_lora"
OUTPUT_DIR = Path(__file__).parent.parent / "lora_output"
OUTPUT_DIR.mkdir(exist_ok=True)

LORA_RANK = 4
LORA_ALPHA = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 1
RESOLUTION = 512  # Use 512 to save memory (vs 1024 for SDXL)
GRADIENT_ACCUMULATION_STEPS = 4
MAX_TRAIN_STEPS = 500
DEVICE = "mps"


class CaptionImageDataset(Dataset):
    """Simple dataset that loads images and captions from metadata.jsonl."""

    def __init__(self, data_dir, resolution=512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.items = []

        meta_path = self.data_dir / "metadata.jsonl"
        with open(meta_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                img_path = self.data_dir / entry["file_name"]
                if img_path.exists():
                    self.items.append({
                        "image_path": img_path,
                        "caption": entry["text"],
                    })

        print(f"Loaded {len(self.items)} image-caption pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Normalize to [-1, 1]
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        pixel_values = transform(image)
        return {"pixel_values": pixel_values, "caption": item["caption"]}


def main():
    print("=" * 60)
    print("SDXL LoRA Fine-tuning for Cartoon Style")
    print("=" * 60)

    # Check dataset
    meta_path = DATA_DIR / "metadata.jsonl"
    if not meta_path.exists():
        print(f"ERROR: Dataset not found at {DATA_DIR}")
        print("Run 02_prepare_dataset.py first!")
        return

    dataset = CaptionImageDataset(DATA_DIR, resolution=RESOLUTION)
    if len(dataset) == 0:
        print("ERROR: No images found in dataset!")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load pipeline components
    print("\nLoading SDXL components...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,  # float32 for training stability on MPS
        use_safetensors=True,
    )

    # Extract components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze everything except UNet LoRA
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to UNet
    print(f"\nAdding LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Move to device
    vae.to(DEVICE)
    text_encoder_1.to(DEVICE)
    text_encoder_2.to(DEVICE)
    unet.to(DEVICE)

    # Optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-2)

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Max steps: {MAX_TRAIN_STEPS}")
    print(f"  Resolution: {RESOLUTION}x{RESOLUTION}")

    global_step = 0
    unet.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            captions = batch["caption"]

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Encode text
            with torch.no_grad():
                # Tokenize for both text encoders
                tokens_1 = tokenizer_1(
                    captions, padding="max_length",
                    max_length=tokenizer_1.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(DEVICE)

                tokens_2 = tokenizer_2(
                    captions, padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(DEVICE)

                encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
                encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)

                # SDXL uses penultimate hidden states
                text_embeds_1 = encoder_output_1.hidden_states[-2]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                prompt_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)

                # Pooled output from text_encoder_2
                pooled_prompt_embeds = encoder_output_2[0]

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=DEVICE
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL requires additional conditioning
            add_time_ids = torch.tensor(
                [[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]],
                dtype=prompt_embeds.dtype, device=DEVICE
            ).repeat(latents.shape[0], 1)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Loss
            loss = F.mse_loss(noise_pred, noise)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Step {global_step}/{MAX_TRAIN_STEPS}, Loss: {avg_loss:.4f}")

                if global_step >= MAX_TRAIN_STEPS:
                    break

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {avg_epoch_loss:.4f}")

        if global_step >= MAX_TRAIN_STEPS:
            break

    # Save LoRA weights
    print(f"\nSaving LoRA weights to {OUTPUT_DIR}...")
    unet.save_pretrained(OUTPUT_DIR / "unet_lora")
    print("Training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR / 'unet_lora'}")


if __name__ == "__main__":
    main()
