"""
merge_and_convert.py — Merge LoRA adapter into Whisper and convert to CTranslate2.

Run from the TikTalk-backend directory after placing the LoRA adapter in
asr_deploy/best_adapter/:

    python asr_deploy/merge_and_convert.py

Then upload the resulting ct2_model/ to Modal:

    modal volume put tiktalk-asr-model asr_deploy/ct2_model/ /
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

_HERE = Path(__file__).parent
ADAPTER_DIR = _HERE / "best_adapter"
MERGED_MODEL_DIR = _HERE / "merged_model"
CT2_MODEL_DIR = _HERE / "ct2_model"

CT2_QUANTIZATION = "float16"


def merge_lora_weights() -> None:
    print("=" * 60)
    print("Step 1: Merging LoRA adapter into base model")
    print("=" * 60)

    if MERGED_MODEL_DIR.exists() and any(MERGED_MODEL_DIR.iterdir()):
        print(f"Merged model already exists at '{MERGED_MODEL_DIR}', skipping.")
        return

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print(f"Loading base model: {BASE_MODEL_ID} ...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True,
    )
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    print(f"Loading LoRA adapter from: {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model = model.merge_and_unload()

    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_DIR)
    processor.save_pretrained(MERGED_MODEL_DIR)
    processor.feature_extractor.save_pretrained(MERGED_MODEL_DIR)
    print(f"Merged model saved to '{MERGED_MODEL_DIR}'")


def convert_to_ctranslate2() -> None:
    print("\n" + "=" * 60)
    print("Step 2: Converting to CTranslate2 format")
    print("=" * 60)

    if (CT2_MODEL_DIR / "model.bin").exists():
        print(f"CTranslate2 model already exists at '{CT2_MODEL_DIR}', skipping.")
        return

    CT2_MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError(f"CTranslate2 conversion failed:\n{result.stderr}")

    for src_name, dst_name in [
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("preprocessor_config.json", "preprocessor_config.json"),
    ]:
        src = MERGED_MODEL_DIR / src_name
        dst = CT2_MODEL_DIR / dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    print(f"CTranslate2 model saved to '{CT2_MODEL_DIR}'")


if __name__ == "__main__":
    merge_lora_weights()
    convert_to_ctranslate2()
    print("\nDone! Upload with: modal volume put tiktalk-asr-model asr_deploy/ct2_model/ /")
