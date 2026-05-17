# TikTalk VLM Ground Truth

This package implements the VLM-only portion of the TikTalk image ground-truth workflow:

1. Accept a generated image from the upstream diffusion model.
2. Call `qwen3-vl-plus` through DashScope's OpenAI-compatible API.
3. Call OpenAI through the Chat Completions API using a GPT vision model.
4. Ask Gemini to judge which JSON answer is more accurate and more suitable for junior learners.
5. Return only the selected JSON ground truth.

## Install

```bash
pip install -e .
```

## Environment

Create a `.env` file in the project root with these variables:

```bash
OPENAI_API_KEY=...
DASHSCOPE_API_KEY=...
GEMINI_API_KEY=...
```

Optional overrides:

```bash
OPENAI_VLM_MODEL=gpt-5.4
QWEN_VLM_MODEL=qwen3-vl-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
GEMINI_JUDGE_MODEL=gemini-3.1-pro-preview
```

The pipeline auto-loads `.env` via `python-dotenv` when you call `generate_ground_truth`.

`gpt-5.4` is the default OpenAI VLM branch model. `gemini-3.1-pro-preview` is used as the Gemini judge. If your account exposes a different model id, override the corresponding environment variable.

## Usage

```python
from tiktalk_vlm import generate_ground_truth

ground_truth = generate_ground_truth("C:/path/to/generated-image.png")
print(ground_truth)
```

The return value is a single JSON-compatible `dict` representing the chosen ground truth.
