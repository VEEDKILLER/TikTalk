"""
merge_and_convert.py
====================
Step 1: Merge LoRA adapter weights into the base Whisper large-v3 model.
Step 2: Convert the merged model to CTranslate2 format for faster-whisper.

Usage:
    uv run python merge_and_convert.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_MODEL_ID = "openai/whisper-large-v3"
ADAPTER_DIR = Path("best_adapter")
MERGED_MODEL_DIR = Path("merged_model")
CT2_MODEL_DIR = Path("ct2_model")

# CTranslate2 conversion settings
CT2_QUANTIZATION = "float16"  # Options: float16, float32, int8, int8_float16

# ─── Step 1: Merge LoRA weights ─────────────────────────────────────────────


def merge_lora_weights() -> None:
    """Load base model + LoRA adapter, merge weights, and save."""
    print("=" * 60)
    print("Step 1: Merging LoRA adapter into base model")
    print("=" * 60)

    # Skip if already merged
    if MERGED_MODEL_DIR.exists() and any(MERGED_MODEL_DIR.iterdir()):
        print(f"✓ Merged model already exists at '{MERGED_MODEL_DIR}', skipping.")
        return

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"Using CUDA device with FP16")
    elif torch.backends.mps.is_available():
        device = "cpu"  # MPS has limited support; use CPU for merging
        dtype = torch.float32
        print(f"MPS detected, using CPU with FP32 for safe merging")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"Using CPU with FP32")

    # Load base model
    print(f"\nLoading base model: {BASE_MODEL_ID} ...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Load processor (tokenizer + feature extractor)
    print(f"Loading processor from: {BASE_MODEL_ID} ...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))

    # Merge and unload adapter weights
    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    # Save merged model
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {MERGED_MODEL_DIR} ...")
    model.save_pretrained(MERGED_MODEL_DIR)
    processor.save_pretrained(MERGED_MODEL_DIR)

    # Save feature extractor separately as preprocessor_config.json
    # (faster-whisper needs this file to know feature_size=128 for large-v3)
    processor.feature_extractor.save_pretrained(MERGED_MODEL_DIR)

    print(f"✓ Merged model saved to '{MERGED_MODEL_DIR}'")


# ─── Step 2: Convert to CTranslate2 ─────────────────────────────────────────


def convert_to_ctranslate2() -> None:
    """Convert the merged HF model to CTranslate2 format for faster-whisper."""
    print("\n" + "=" * 60)
    print("Step 2: Converting to CTranslate2 format")
    print("=" * 60)

    # Skip if already converted (check for actual model file)
    if (CT2_MODEL_DIR / "model.bin").exists():
        print(f"✓ CTranslate2 model already exists at '{CT2_MODEL_DIR}', skipping.")
        return

    if not MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Merged model not found at '{MERGED_MODEL_DIR}'. "
            "Run Step 1 first."
        )

    CT2_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Run ct2-transformers-converter
    print(f"Running ct2-transformers-converter ...")
    print(f"  Model: {MERGED_MODEL_DIR}")
    print(f"  Output: {CT2_MODEL_DIR}")
    print(f"  Quantization: {CT2_QUANTIZATION}")

    cmd = [
        "ct2-transformers-converter",
        "--model", str(MERGED_MODEL_DIR),
        "--output_dir", str(CT2_MODEL_DIR),
        "--quantization", CT2_QUANTIZATION,
        "--copy_files", "tokenizer.json", "processor_config.json",
        "--force",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR — ct2-transformers-converter failed:")
        if result.stderr:
            print(result.stderr)
        raise RuntimeError("CTranslate2 conversion failed.")
    if result.stdout:
        print(result.stdout)

    # Copy tokenizer_config.json
    tokenizer_config_src = MERGED_MODEL_DIR / "tokenizer_config.json"
    tokenizer_config_dst = CT2_MODEL_DIR / "tokenizer_config.json"
    if tokenizer_config_src.exists() and not tokenizer_config_dst.exists():
        shutil.copy2(tokenizer_config_src, tokenizer_config_dst)

    # Ensure preprocessor_config.json exists in ct2_model
    # (faster-whisper needs this to set feature_size=128 for large-v3)
    preprocessor_dst = CT2_MODEL_DIR / "preprocessor_config.json"
    preprocessor_src = MERGED_MODEL_DIR / "preprocessor_config.json"
    if preprocessor_src.exists():
        shutil.copy2(preprocessor_src, preprocessor_dst)
    elif not preprocessor_dst.exists():
        # Generate from processor_config.json if preprocessor_config.json
        # was not saved during merge (e.g. merged_model already existed)
        proc_cfg_path = MERGED_MODEL_DIR / "processor_config.json"
        if proc_cfg_path.exists():
            proc_cfg = json.loads(proc_cfg_path.read_text())
            fe_cfg = proc_cfg.get("feature_extractor", {})
            preprocessor_dst.write_text(json.dumps(fe_cfg, indent=2))
            print(f"  Generated preprocessor_config.json from processor_config.json")

    print(f"✓ CTranslate2 model saved to '{CT2_MODEL_DIR}'")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("TikTalk ASR — Merge LoRA & Convert to CTranslate2")
    print("─" * 60)
    print(f"Base model      : {BASE_MODEL_ID}")
    print(f"LoRA adapter    : {ADAPTER_DIR}")
    print(f"Merged output   : {MERGED_MODEL_DIR}")
    print(f"CTranslate2 out : {CT2_MODEL_DIR}")
    print(f"Quantization    : {CT2_QUANTIZATION}")
    print()

    merge_lora_weights()
    convert_to_ctranslate2()

    print("\n" + "=" * 60)
    print("✓ All done! The CTranslate2 model is ready for deployment.")
    print(f"  Upload '{CT2_MODEL_DIR}/' to Modal using: modal run serve.py::upload_model")
    print("=" * 60)


if __name__ == "__main__":
    main()
