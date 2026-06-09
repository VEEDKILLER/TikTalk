# TikTalk ASR Cloud Deployment Guide

This document explains how to deploy the LoRA fine-tuned Whisper large-v3 model to the Modal cloud platform.

---

## 1. Project Code Structure

```
tiktalk_asr_cloud_deploy/
├── best_adapter/              # LoRA fine-tuning output files
│   ├── adapter_config.json    # LoRA config (r=16, alpha=32, dropout=0.05, target_modules=q_proj/v_proj)
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── processor_config.json  # Whisper processor config
│   ├── tokenizer.json         # Tokenizer vocabulary
│   └── tokenizer_config.json  # Tokenizer config
├── merge_and_convert.py       # Local script: merge weights + convert format
├── serve.py                   # Modal deployment script: inference service
├── pyproject.toml             # Python dependency declarations
├── DEPLOYMENT.md              # This document
└── FineTuneInstruction.md     # Fine-tuning methodology notes
```

### Generated directories (not committed to Git)

| Directory       | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `merged_model/` | Full Whisper large-v3 model after merging LoRA weights       |
| `ct2_model/`    | CTranslate2 format model, used by faster-whisper             |

---

## 2. Code Overview

### 2.1 `merge_and_convert.py` — Weight merging and format conversion

**Purpose**: completes two steps locally to prepare for deployment.

| Step          | Operation                                              | Key technology                    |
| ------------- | ------------------------------------------------------ | --------------------------------- |
| Step 1 Merge  | Load whisper-large-v3 + LoRA adapter → merge           | PEFT `merge_and_unload()`         |
| Step 2 Convert| Convert merged model to CTranslate2 format             | `ct2-transformers-converter` CLI  |

**Key parameters**:

- **CTranslate2 quantization**: `float16` — reduces model size while preserving inference precision

### 2.2 `serve.py` — Modal inference service

**Purpose**: defines the Modal app and runs faster-whisper inference on a T4 GPU in the cloud.

#### Inference hyperparameters

| Parameter      | Value    | Description                                             |
| -------------- | -------- | ------------------------------------------------------- |
| `beam_size`    | 5        | Beam search width — higher is more accurate but slower  |
| `language`     | `"en"`   | Target language (English, child speech)                 |
| `vad_filter`   | `True`   | Enable Voice Activity Detection to skip silence         |
| `min_silence_duration_ms` | 500 | Minimum silence duration for VAD split (ms)       |
| `compute_type` | `float16`| GPU inference precision                                 |

#### Modal resource configuration

| Parameter                 | Value | Description                              |
| ------------------------- | ----- | ---------------------------------------- |
| `gpu`                     | `T4`  | Use T4 GPU                               |
| `timeout`                 | 300s  | Per-request timeout                      |
| `scaledown_window`        | 120s  | Auto-shutdown after container idle       |
| `@modal.concurrent(max_inputs=...)` | 5 | Max concurrent requests              |

#### API endpoint

- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Field name**: `audio`
- **Supported formats**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`

**Response example**:

```json
{
  "text": "hello how are you doing today",
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "hello how are you" },
    { "start": 2.5, "end": 4.0, "text": "doing today" }
  ],
  "language": "en",
  "language_probability": 0.98,
  "duration": 4.0
}
```

---

## 3. Deployment Steps (command-line)

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- [Modal](https://modal.com/) account registered
- Fine-tuned LoRA files present in `best_adapter/`

### Step 1: Install dependencies

```bash
uv sync
```

### Step 2: Set up Modal

After registering/logging in to Modal, run:

```bash
uv run modal setup
```

Follow the prompt to complete authentication in your browser.

### Step 3: Merge weights and convert format

```bash
uv run python merge_and_convert.py
```

This will:
1. Download `openai/whisper-large-v3` base model from Hugging Face (~6GB, first run only)
2. Merge LoRA weights → output to `merged_model/`
3. Convert to CTranslate2 format → output to `ct2_model/`

> ⏱️ First run may take 10–20 minutes (depending on network and hardware)

### Step 4: Upload model to Modal Volume

```bash
uv run modal volume create tiktalk-asr-model
uv run modal volume put tiktalk-asr-model ct2_model/ /
```

### Step 5: Deploy to Modal

```bash
uv run modal deploy serve.py
```

After successful deployment, a web endpoint URL will be printed, in the format:
```
https://your-workspace--tiktalk-asr-serve-app.modal.run
```
The transcription route is `POST /transcribe`, e.g.
`https://your-workspace--tiktalk-asr-serve-app.modal.run/transcribe`.

### Step 6 (Development mode, optional): Hot-reload debugging

```bash
uv run modal serve serve.py
```

This starts a temporary endpoint that auto-redeploys on code changes.

---

## 4. Postman Testing

1. Open Postman and create a new request
2. Set method to **POST**
3. Enter the Modal endpoint URL in the URL field
4. Switch to the **Body** tab → select **form-data**
5. Add a field:
   - **Key**: `audio` (set type to **File**)
   - **Value**: select a local `.wav` or `.mp3` file
6. Click **Send**
7. Inspect the returned JSON, which contains `text` (full transcript) and `segments` (per-segment transcription with timestamps)

---

## 5. Common Modal Commands

| Command                                             | Description                      |
| --------------------------------------------------- | -------------------------------- |
| `modal deploy serve.py`                             | Deploy (production mode)         |
| `modal serve serve.py`                              | Development / debug mode         |
| `modal app list`                                    | List deployed apps               |
| `modal app stop tiktalk-asr`                        | Stop the app                     |
| `modal volume ls tiktalk-asr-model`                 | List files in the Volume         |
| `modal volume put tiktalk-asr-model ct2_model/ /`   | Upload model to Volume           |
| `modal volume delete tiktalk-asr-model`             | Delete Volume (use with caution) |
