# TikTalk

TikTalk is an AI-powered English speaking practice application for young learners. It generates a child-friendly practice image, analyzes the image to build a reference description, records the learner's spoken response, and returns scoring plus feedback across content, grammar, pronunciation, and fluency.

## Features

- Generate practice scenes with DALL-E 3 from predefined categories and characters.
- Analyze generated images with a VLM pipeline using Qwen, OpenAI, and Gemini judging.
- Record audio directly in the browser with microphone access.
- Transcribe speech through the ASR service endpoint.
- Evaluate learner responses with structured scoring and child-friendly feedback.
- Display total score, score breakdown, transcript, risk flags, and improvement tips.

## Project Structure

```text
.
|-- TikTalk-backend/      # FastAPI gateway for image generation, VLM, ASR, and evaluation
|-- TikTalk-frontend/     # Vite + React frontend
|-- ASR/                  # ASR training, preprocessing, and deployment materials
|-- VLM/                  # Standalone VLM ground-truth package
|-- Diffusion/            # Local SDXL-Turbo baseline & cartoon-style LoRA experiments
`-- README.md
```

## Tech Stack

- Frontend: React 18, Vite, Tailwind CSS
- Backend: FastAPI, Uvicorn, python-dotenv
- AI services: OpenAI, Alibaba DashScope, Google Gemini
- ASR: Whisper Large V3 based remote transcription endpoint

## Prerequisites

- Node.js 18 or newer
- Python 3.11 or newer
- `uv` for Python dependency management, or an equivalent virtual environment workflow
- API keys for OpenAI, DashScope, and Gemini
- Browser microphone permission enabled for local testing

## Backend Setup

From the project root:

```bash
cd TikTalk-backend
cp .env.example .env
```

Edit `.env` and provide the required keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Optional environment variables are also listed in `TikTalk-backend/.env.example`, including ASR endpoint and model overrides.

Install dependencies and start the FastAPI server:

```bash
uv sync
uv run uvicorn main:app --reload --port 8080
```

The backend will be available at:

- API base: `http://localhost:8080`
- Swagger docs: `http://localhost:8080/docs`
- Health check: `http://localhost:8080/health`

To verify API keys:

```bash
cd TikTalk-backend
uv run python test_apis.py
```

## Frontend Setup

Open a second terminal from the project root:

```bash
cd TikTalk-frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:3000
```

During development, Vite proxies frontend `/api` requests to the backend at `http://localhost:8080`.

## Application Flow

1. The learner selects a scene category and character.
2. The frontend calls `POST /session/start` to generate a practice image.
3. The frontend calls `POST /session/vlm` in the background to create image ground truth.
4. The learner records a spoken description in the browser.
5. The frontend submits audio to `POST /session/evaluate`.
6. The backend runs ASR, evaluates the transcript against the image ground truth, and returns scores and feedback.

## Main Backend Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check backend status and configured modules. |
| `GET` | `/session/categories` | Return available scene categories and characters. |
| `POST` | `/session/start` | Generate a practice image and return `image_id` plus base64 image data. |
| `POST` | `/session/vlm` | Analyze the generated image and persist ground truth for evaluation. |
| `POST` | `/session/evaluate` | Submit learner audio and receive transcript, scores, and feedback. |

## Development Commands

Backend:

```bash
cd TikTalk-backend
uv run uvicorn main:app --reload --port 8080
uv run python test_apis.py
```

Frontend:

```bash
cd TikTalk-frontend
npm run dev
npm run build
npm run preview
```

## Notes

- Generated images are stored in `/tmp/tiktalk-images` by default. Set `TIKTALK_IMAGE_DIR` to override this path.
- The frontend requires a secure browser context for microphone access. `localhost` is supported by modern browsers.
- If evaluation fails with an image lookup error, regenerate the image and retry. The backend needs the generated image file and its VLM ground truth to exist before scoring.
- If ASR fails, check `ASR_ENDPOINT` in `TikTalk-backend/.env`.

## Related Modules

- `ASR/asr/asr_module.md` documents the ASR pipeline, training approach, and deployed transcription endpoint.
- `VLM/README.md` documents the standalone VLM ground-truth workflow.
- `Diffusion/README.md` documents the local SDXL-Turbo baseline and cartoon-style LoRA experiments (paper §4.1.1 / §4.2.1, Tables 4 & 9).
