# ASR Module Documentation

## Overview

The ASR module is organized into three main components:

```
data_preprocessing_src/   # Data preprocessing
training_src/             # Model training (multi-model ablation framework)
deployment_src/           # Model deployment
```

The deployed model is **Whisper large-v3 fine-tuned with LoRA**. The training code,
however, is a **multi-model ablation framework**: it can fine-tune and compare three
ASR backbones (Whisper, Qwen3-ASR, Cohere) under the same data/eval pipeline, so the
choice of Whisper is the result of a controlled comparison rather than the only option.
See [Model & Training Approach](#model--training-approach) for details.

---

## API Usage

The ASR module has been successfully deployed as part of the **TikTalk** project.

**Endpoint:** `https://howdybunny--tiktalk-asr-serve-app.modal.run/transcribe`
**Method:** `POST`

### Testing with Postman

| Field | Value |
|---|---|
| Body type | `form-data` |
| Key | `audio` (set type to **File**, not Text) |
| Value | Select an audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`) |

### Backend Pipeline Integration

**Request Parameters**

| Parameter | Type | Location | Required | Description |
|---|---|---|---|---|
| `audio` | File | `form-data` | ✅ | Audio file to transcribe (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`) |

**Response Format**

On success, the endpoint returns a JSON object:

```json
{
  "text": "full transcription as a single string...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.13,
      "text": "your dad help you tie your shoe too ?"
    }
  ],
  "language": "en",
  "language_probability": 1.0,
  "duration": 26.71
}
```

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Full transcription of the audio as a single continuous string |
| `segments` | `array` | List of transcription segments with timestamps |
| `segments[].start` | `float` | Segment start time in seconds |
| `segments[].end` | `float` | Segment end time in seconds |
| `segments[].text` | `string` | Transcribed text for this segment |
| `language` | `string` | Detected language code (e.g. `"en"`) |
| `language_probability` | `float` | Confidence score for the detected language (0–1) |
| `duration` | `float` | Total audio duration in seconds |

On error (unsupported format or empty file) the endpoint returns HTTP `400` with an
`{"error": "..."}` body.

## Model & Training Approach

The deployed ASR model is built on **Whisper large-v3**, fine-tuned using **LoRA**.

**Training Dataset:** [CHILDES OCSC Corpus](https://talkbank.org/childes/access/Eng-NA/OCSC.html)

### Multi-model ablation framework

`training_src` is configuration-driven. Each backbone has its own YAML under
`training_src/configs/` and a matching adapter under `training_src/models/`:

| Config | Model | Adapter |
|---|---|---|
| `configs/whisper_large_v3.yaml` | `openai/whisper-large-v3` | `whisper` |
| `configs/qwen3_asr_1_7b.yaml` | `Qwen/Qwen3-ASR-1.7B` | `qwen3` |
| `configs/cohere_transcribe_2026.yaml` | Cohere transcribe | `cohere` |

`eval_compare.py` runs **baseline vs LoRA** for each model on the same held-out test
split and reports **WER / CER / RTFx**, which is how Whisper was selected for
deployment.

### LoRA (PEFT) configuration — Whisper

These are the values actually used by `configs/whisper_large_v3.yaml` (and the defaults
in `config.py`); they match the deployed `best_adapter`:

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |

### Training hyperparameters — Whisper

| Parameter | Value |
|---|---|
| Per-device batch size | 8 |
| Gradient accumulation | 2 steps (effective batch size 16) |
| Learning rate | 1e-4 (AdamW + linear warmup) |
| Warmup steps | 50 |
| Max epochs | 5 |
| Eval / save interval | every 500 steps |
| Early stopping | patience = 3, metric = WER |
| Best model selection | lowest WER (`load_best_model_at_end`) |
| Seed | 42 |

**Device-aware behavior** (`config.py` auto-detects CUDA > MPS > CPU):

- On **CUDA**: FP16 training and 4-bit bitsandbytes quantization (nf4, double-quant) are
  enabled automatically.
- On **MPS / CPU** (e.g. local Mac smoke test): quantization and FP16 are disabled so the
  same code runs unchanged.

**Other details**: dynamic padding via a custom data collator (label padding masked to
`-100`); `predict_with_generate=True` during evaluation; WER/CER computed after applying
the Whisper `BasicTextNormalizer` (lowercase, strip punctuation) so metrics reflect true
recognition quality.

---

## Getting Started

### Step 1 — Data Preprocessing

1. Download the dataset from the link above and place it in `data_preprocessing_src/dataset_orig/`
   (audio in `dataset_orig/4 … 9`, transcripts in `dataset_orig/OCSC/`).
2. Follow the step-by-step instructions in `data_preprocessing_src/command.md`:
   ```bash
   uv run python preprocess.py      # clean the 303 .cha transcripts
   uv run python align.py           # stable-ts / faster-whisper alignment (slow)
   uv run python build_dataset.py   # slice audio + build AudioFolder dataset
   ```
3. The processed output is saved to `data_preprocessing_src/clean_dataset/`
   (16 kHz mono WAV in `audio/`, plus `metadata.csv` with `file_name` and
   `transcription` columns).

### Step 2 — Training

1. Copy the contents of `clean_dataset/` into `training_src/dataset/`.
2. Install dependencies and run training **with an explicit config** (the `--config`
   flag is required):
   ```bash
   uv sync
   uv run python train.py --config configs/whisper_large_v3.yaml
   ```
   To run the ablation against the other backbones:
   ```bash
   uv run python train.py --config configs/qwen3_asr_1_7b.yaml
   uv run python train.py --config configs/cohere_transcribe_2026.yaml
   ```
   The best LoRA adapter is saved to `results/whisper-lora/best_adapter/`. Training logs
   go to `results/logs/` (view with `uv run tensorboard --logdir results/logs`).
   See `training_src/COMMAND.md` for more detail.

### Step 3 — Evaluation (Baseline vs Fine-tuned)

Compare the un-fine-tuned baseline against the LoRA model on the test split:

```bash
# single model
uv run python eval_compare.py --config configs/whisper_large_v3.yaml

# all models, writes results/ablation_summary.csv
uv run python eval_compare.py --all

# subset of models
uv run python eval_compare.py --all --models whisper qwen3
```

This prints a WER / CER / RTFx comparison table and demonstrates the effect of
fine-tuning.

### Step 4 — Deployment

1. Copy the best adapter checkpoint from `training_src/results/whisper-lora/best_adapter/`
   into `deployment_src/best_adapter/`.
2. Follow the instructions in `deployment_src/DEPLOYMENT.md` to merge the LoRA weights,
   convert to CTranslate2 (float16), upload to a Modal Volume, and deploy the
   faster-whisper inference service.
