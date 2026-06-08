# Training Commands

## Prerequisites

Ensure the preprocessed dataset is available at `dataset/` (produced by `data_preprocessing_src`):
```
dataset/
├── metadata.csv
└── audio/
    ├── 4001_000.wav
    ├── 4001_001.wav
    └── ...
```

Install dependencies:
```bash
uv sync
```

---

## Step 1: Run Training

The `--config` flag is **required**:

```bash
uv run python train.py --config configs/whisper_large_v3.yaml
```

To run the ablation against the other backbones:

```bash
uv run python train.py --config configs/qwen3_asr_1_7b.yaml
uv run python train.py --config configs/cohere_transcribe_2026.yaml
```

This will:
- Auto-detect device (CUDA > MPS > CPU)
- Enable 4-bit quantization and FP16 on CUDA automatically
- Load `openai/whisper-large-v3` and apply LoRA (r=16, alpha=32)
- Split dataset into train/val/test (80/10/10)
- Train with early stopping (patience=3, metric=WER)
- Save the best LoRA adapter to `results/whisper-lora/best_adapter/`

Training logs are written to `results/logs/` and can be viewed with TensorBoard:
```bash
uv run tensorboard --logdir results/logs
```

---

## Step 2: Evaluate (Baseline vs Fine-tuned)

Exactly one of `--config` or `--all` is **required**:

```bash
# single model
uv run python eval_compare.py --config configs/whisper_large_v3.yaml

# all models, writes results/ablation_summary.csv
uv run python eval_compare.py --all

# subset of models (filter by config filename keyword)
uv run python eval_compare.py --all --models whisper qwen3
```

This runs both the baseline model and the LoRA fine-tuned model on the test split and prints a WER/CER/RTFx comparison table.

---

## Key Configuration

Hyperparameters live in `configs/*.yaml` (which override the defaults in `config.py`).
Notable values for `configs/whisper_large_v3.yaml`:

| Parameter | Value |
|---|---|
| Model | `openai/whisper-large-v3` |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Batch size | 8 (per device) |
| Gradient accumulation | 2 steps (effective batch size 16) |
| Learning rate | 1e-4 |
| Warmup steps | 50 |
| Max epochs | 5 |
| Early stopping patience | 3 |
| Eval/save interval | every 500 steps |

---

## Output Structure

```
results/
├── whisper-lora/
│   ├── best_adapter/        # Best LoRA adapter weights + processor
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── ...
│   └── checkpoint-*/        # Intermediate checkpoints
└── logs/                    # TensorBoard logs
```
