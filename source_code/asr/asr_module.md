# ASR Module Documentation

## Overview

The ASR module is organized into three main components:

```
data_preprocessing_src/   # Data preprocessing
training_src/             # Model training
deployment_src/           # Model deployment
```

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
| Value | Select a `.wav` or `.mp3` audio file |

### Backend Pipeline Integration

**Request Parameters**

| Parameter | Type | Location | Required | Description |
|---|---|---|---|---|
| `audio` | File | `form-data` | ✅ | Audio file to transcribe (`.wav` or `.mp3`) |

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
  "language_probability": 1,
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

## Model & Training Approach

The ASR module is built on **Whisper v3 Large**, fine-tuned using **LoRA**.

**Training Dataset:** [CHILDES OCSC Corpus](https://talkbank.org/childes/access/Eng-NA/OCSC.html)

---

## Getting Started

### Step 1 — Data Preprocessing

1. Download the dataset from the link above and place it in `data_preprocessing_src/dataset_orig/`.
2. Follow the step-by-step instructions in `data_preprocessing_src/command.md` to clean and prepare the data.
3. The processed output will be saved to `data_preprocessing_src/clean_dataset/`.

### Step 2 — Training

1. Copy the contents of `clean_dataset/` into `training_src/dataset/`.
2. Run the training pipeline as described in `training_src/`, follow the step-by-step instructions in `training_src/COMMAND.md`.

### Step 3 — Deployment

1. Copy the best adapter checkpoint from `training_src/results/whisper-lora/best_adapter/` into `deployment_src/`.
2. Follow the instructions in `deployment_src/DEPLOYMENT.md` to deploy the service.