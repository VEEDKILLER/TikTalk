# TikTalk ASR — LoRA Fine-Tuning

A config-driven framework for parameter-efficient (LoRA) fine-tuning of automatic
speech recognition (ASR) backbones on child speech. Three backbones are supported
through interchangeable YAML configs. **Whisper-large-v3** is the backbone selected
for deployment, after a baseline-vs-fine-tuned ablation showed it achieves the lowest
WER/CER on the held-out test set.

## Dataset

Training consumes the preprocessed corpus produced by [`../data_preprocessing_src/`](../data_preprocessing_src)
(the Ohio Child Speech Corpus, OCSC):

```
dataset/
├── metadata.csv        # audio path + reference transcript, one row per clip
└── audio/*.wav         # clips of at most 30 s, 16 kHz mono
```

The data is split into train/validation/test at **80/10/10** with a fixed **random
seed 42**, so baseline and fine-tuned runs are directly comparable.

## Backbones and configs

Each backbone is defined by one YAML under [`configs/`](configs):

| Config | Backbone | LoRA (r / α / dropout) | Learning rate |
|---|---|---|---|
| `whisper_large_v3.yaml` | `openai/whisper-large-v3` | 16 / 32 / 0.05 | 1e-4 |
| `qwen3_asr_1_7b.yaml` | `Qwen/Qwen3-ASR-1.7B` | 16 / 32 / 0.05 | 5e-5 |
| `cohere_transcribe_2026.yaml` | `CohereLabs/cohere-transcribe-03-2026` | 8 / 16 / 0.1 | 3e-5 |

LoRA target modules differ per architecture (attention projections; see each YAML header).

## Training setup (Whisper, deployed config)

- **LoRA**: r=16, α=32, dropout 0.05, target modules `q_proj, v_proj`.
- **Optimizer / schedule**: AdamW, learning rate 1e-4, linear warmup (50 steps).
- **Effective batch size 16**: per-device batch 8 × gradient accumulation 2.
- **Early stopping**: monitors validation WER with patience 3 (stops after 3 epochs
  without improvement); up to 5 epochs total.
- **Evaluation**: `predict_with_generate=True`; WER/CER are computed after applying the
  Whisper English text normalizer (lowercase, strip punctuation) to both predictions and
  references, so the metrics reflect recognition quality rather than formatting.
- **Padding**: a custom `DataCollatorSpeechSeq2SeqWithPadding` pads input features and
  replaces label padding with `-100` to mask it from the loss.

The Qwen3 and Cohere configs use their own LoRA settings, learning rates, target modules,
and epoch counts (see each YAML).

## Device and precision

The device is auto-detected: **CUDA > MPS > CPU**. On CUDA, 4-bit bitsandbytes
quantization and FP16 are enabled automatically; on MPS/CPU they are disabled. The Cohere
config is an exception — it disables 4-bit quantization and trains in bf16 (its
`from_pretrained` quantization path cannot be patched after loading, and fp16's `-1e9`
attention-mask value overflows half precision).

## Commands

```bash
uv sync

# Train one backbone
uv run python train.py --config configs/whisper_large_v3.yaml

# Evaluate baseline vs LoRA fine-tuned (WER / CER / RTFx)
uv run python eval_compare.py --all                                  # all backbones
uv run python eval_compare.py --all --models whisper                 # subset
uv run python eval_compare.py --config configs/whisper_large_v3.yaml # single backbone

# Training curves
uv run tensorboard --logdir results/logs
```

> **Environments.** The three backbones require different — and partly conflicting —
> library versions, notably `transformers` and the Qwen `qwen-asr` package. The committed
> `pyproject.toml` targets the Whisper configuration used for deployment; reproduce the
> Qwen3 / Cohere ablations in a separate environment.

## Outputs

The best LoRA adapter is written to `results/<model>-lora/best_adapter/`, and each run's
evaluation log (`eval_output_*.log`) records baseline-vs-fine-tuned WER/CER/RTFx. The
deployed Whisper adapter is merged, converted, and served from
[`../deployment_src/`](../deployment_src) (see its `DEPLOYMENT.md`).
