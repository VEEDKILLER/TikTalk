"""Sanity check: 用 baseline + checkpoint-1000 LoRA 各跑 5 条样本，对比输出。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import pandas as pd

from config import load_config
from models.cohere_adapter import CohereAdapter

CFG_PATH = "configs/cohere_transcribe_2026.yaml"
ADAPTER_PATH = "results/cohere-lora/checkpoint-1000"
N = 5

cfg = load_config(CFG_PATH)
adapter = CohereAdapter()
processor = adapter.load_processor(cfg)

df = pd.read_csv(cfg.data.metadata_path).head(N)
audios, refs = [], []
for _, row in df.iterrows():
    path = Path(cfg.data.audio_base_dir) / row["file_name"]
    wav, _ = librosa.load(str(path), sr=cfg.data.sampling_rate)
    audios.append(wav)
    refs.append(row["transcription"])

print("\n" + "=" * 60)
print("BASELINE")
print("=" * 60)
base = adapter.load_baseline_model(cfg)
base_out = adapter.transcribe_batch(base, processor, audios, cfg.data.sampling_rate, cfg.device)
for i, (r, h) in enumerate(zip(refs, base_out)):
    print(f"\n[{i+1}] REF: {r[:120]}")
    print(f"    HYP: {h[:120]}")

del base
import torch, gc
gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print(f"LoRA (from {ADAPTER_PATH})")
print("=" * 60)
lora = adapter.load_finetuned_model(cfg, ADAPTER_PATH)
lora_out = adapter.transcribe_batch(lora, processor, audios, cfg.data.sampling_rate, cfg.device)
for i, (r, h) in enumerate(zip(refs, lora_out)):
    print(f"\n[{i+1}] REF: {r[:120]}")
    print(f"    HYP: {h[:120]}")
