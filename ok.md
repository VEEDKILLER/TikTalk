Here's the Backend Pipeline section:

---

### Backend Pipeline Integration

The backend (FastAPI) calls the ASR endpoint asynchronously — the audio file is submitted and the transcription result is returned via callback or queue.

**Request Parameters**

| Parameter | Type | Location | Required | Description |
|---|---|---|---|---|
| `audio` | File | `form-data` | ✅ | Audio file to transcribe (`.wav` or `.mp3`) |

**Response Format**

On success, the endpoint returns a JSON object:

```json
{
  "transcription": "the transcribed text content here"
}
```

| Field | Type | Description |
|---|---|---|
| `transcription` | `string` | The transcribed text from the submitted audio file |

**Example — Async Call in FastAPI**

```python
import httpx

ASR_ENDPOINT = "https://howdybunny--tiktalk-asr-serve-app.modal.run/transcribe"

async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            ASR_ENDPOINT,
            files={"audio": (filename, audio_bytes, "audio/wav")},
        )
        response.raise_for_status()
        return response.json()["transcription"]
```

> **Note:** The ASR service may take several seconds to respond depending on audio length. Ensure the HTTP client timeout is set accordingly (recommended: ≥ 60s). If integrating with a task queue (e.g. Celery, ARQ), submit `transcribe_audio` as a background task and pass the result downstream once resolved.

---

如果你已经知道实际返回的 JSON 结构或者有 queue/callback 的具体实现方式，告诉我，我可以进一步补充。


Here's the updated **Response Format** section:

---

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

---

直接替换原来的 Response Format 部分即可，其他内容不变。