"""测试不同长度 / 不同样本 + 简单合成英语，排除"特定音频问题"。"""
import torch, librosa, numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "CohereLabs/cohere-transcribe-03-2026"

print("Loading model in float32...")
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float32, device_map="auto"
)
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()

# Test 1: 截短到 5s
print("\n=== Test 1: 5s 截短 of 4001_000.wav ===")
wav, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
out = model.transcribe(processor=processor, language="en",
                       audio_arrays=[wav[:5*sr]], sample_rates=[sr])
print(f"HYP: {out[0][:200]}")

# Test 2: 不同样本
print("\n=== Test 2: 4001_005.wav (full) ===")
wav, sr = librosa.load("dataset/audio/4001_005.wav", sr=16000)
print(f"len={len(wav)/sr:.2f}s")
out = model.transcribe(processor=processor, language="en",
                       audio_arrays=[wav], sample_rates=[sr])
print(f"HYP: {out[0][:200]}")

# Test 3: 静音
print("\n=== Test 3: 5s 静音 ===")
silence = np.zeros(5*16000, dtype=np.float32)
out = model.transcribe(processor=processor, language="en",
                       audio_arrays=[silence], sample_rates=[16000])
print(f"HYP: {out[0][:200]}")

# Test 4: 使用 audio_files 路径方式（让 cohere 自己加载）
print("\n=== Test 4: audio_files (内部加载, 4001_000.wav) ===")
out = model.transcribe(processor=processor, language="en",
                       audio_files=["dataset/audio/4001_000.wav"])
print(f"HYP: {out[0][:200]}")
