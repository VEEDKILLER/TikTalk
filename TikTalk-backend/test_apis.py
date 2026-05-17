"""
Quick sanity check for all three API keys.
Run from TikTalk-backend/:
    python test_apis.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PASS = "✅"
FAIL = "❌"

def test_openai():
    print("Testing OpenAI...", end=" ", flush=True)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print(f"{FAIL} OPENAI_API_KEY not set in .env")
        return False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
        )
        reply = resp.choices[0].message.content.strip().lower()
        print(f"{PASS} ({reply})")
        return True
    except Exception as e:
        print(f"{FAIL} {e}")
        return False

def test_dashscope():
    print("Testing DashScope (Qwen)...", end=" ", flush=True)
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        print(f"{FAIL} DASHSCOPE_API_KEY not set in .env")
        return False
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=key,
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        model = os.getenv("QWEN_VLM_MODEL", "qwen-vl-plus")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
        )
        reply = resp.choices[0].message.content.strip().lower()
        print(f"{PASS} ({reply})")
        return True
    except Exception as e:
        print(f"{FAIL} {e}")
        return False

def test_gemini():
    print("Testing Gemini...", end=" ", flush=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print(f"{FAIL} GEMINI_API_KEY not set in .env")
        return False
    try:
        from google import genai
        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-pro-preview")
        resp = client.models.generate_content(
            model=model,
            contents="Reply with the single word: ok",
        )
        reply = resp.text.strip().lower()
        print(f"{PASS} ({reply})")
        return True
    except Exception as e:
        print(f"{FAIL} {e}")
        return False

if __name__ == "__main__":
    print("=" * 40)
    print("TikTalk API Key Test")
    print("=" * 40)
    results = [
        test_openai(),
        test_dashscope(),
        test_gemini(),
    ]
    print("=" * 40)
    passed = sum(results)
    print(f"Result: {passed}/3 passed")
    if passed < 3:
        print("Fix the failing keys in your .env file.")
    sys.exit(0 if passed == 3 else 1)
