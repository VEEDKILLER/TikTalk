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

```bash
uv run python train.py
```

This will:
- Auto-detect device (CUDA > MPS > CPU)
- Enable 4-bit quantization and FP16 on CUDA automatically
- Load `openai/whisper-large-v3` and apply LoRA (r=32, alpha=64)
- Split dataset into train/val/test (80/10/10)
- Train with early stopping (patience=3, metric=WER)
- Save the best LoRA adapter to `results/whisper-lora/best_adapter/`

Training logs are written to `results/logs/` and can be viewed with TensorBoard:
```bash
uv run tensorboard --logdir results/logs
```

---

## Step 2: Evaluate (Baseline vs Fine-tuned)

```bash
uv run python eval_compare.py
```

This runs both the baseline Whisper model and the LoRA fine-tuned model on the test split and prints a WER/CER comparison table.

---

## Key Configuration

All hyperparameters are in `config.py`. Notable defaults:

| Parameter | Value |
|---|---|
| Model | `openai/whisper-large-v3` |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj` |
| Batch size | 2 (per device) |
| Gradient accumulation | 8 steps |
| Learning rate | 1e-4 |
| Max epochs | 20 |
| Early stopping patience | 3 |
| Eval/save interval | every 50 steps |

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
