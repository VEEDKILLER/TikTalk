# Fine-Tuning Notes

This document records how the LoRA adapter in `best_adapter/` was produced. The full
training framework, configs, and evaluation live in [`../training_src/`](../training_src).

## Model and adaptation

- **Backbone**: `openai/whisper-large-v3`
- **Method**: Low-Rank Adaptation (LoRA) via PEFT
- **LoRA config**: rank r=16, α=32, dropout 0.05, target modules `q_proj, v_proj`

## Training setup

- **Optimizer / schedule**: AdamW, learning rate 1e-4, linear warmup (50 steps)
- **Effective batch size 16**: per-device batch 8 × gradient accumulation 2
- **Precision**: FP16 on CUDA, with 4-bit bitsandbytes quantization for the base weights
- **Early stopping**: monitors validation WER, patience 3, up to 5 epochs
- **Metrics**: WER/CER computed after applying the Whisper English text normalizer
  (lowercase, strip punctuation) to both predictions and references

## Dataset

Ohio Child Speech Corpus (OCSC), preprocessed by [`../data_preprocessing_src/`](../data_preprocessing_src):
clips of at most 30 s, 16 kHz mono, split 80/10/10 with random seed 42.

## Result

LoRA fine-tuning reduced WER on the held-out OCSC test set from **0.1675** (untuned
Whisper-large-v3 baseline) to **0.1217** — a 27.3% relative reduction — which is why this
adapter was selected for deployment. The adapter in `best_adapter/` is then merged into
the base model and converted for serving as described in `DEPLOYMENT.md`.
