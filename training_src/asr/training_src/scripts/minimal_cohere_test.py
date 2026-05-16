"""最小化测试：完全按 Cohere 模型卡用法，看 baseline 能否转录任何东西。"""
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "CohereLabs/cohere-transcribe-03-2026"
AUDIO = "dataset/audio/4001_000.wav"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

# 试 3 种 dtype
for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
    print(f"\n=== dtype = {dtype_name} ===")
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
        )
        # BatchNorm fix
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.float()

        wav, sr = librosa.load(AUDIO, sr=16000)
        print(f"  audio: {len(wav)/sr:.1f}s")
        out = model.transcribe(
            processor=processor,
            language="en",
            audio_arrays=[wav],
            sample_rates=[sr],
        )
        print(f"  HYP: {out[0][:200]}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    finally:
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
