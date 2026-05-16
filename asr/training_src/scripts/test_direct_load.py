"""绕过 from_pretrained，直接 load_state_dict 看转录是否正常。"""
import torch, librosa, safetensors.torch as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

CKPT = "/home/howdybunny/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/76b8b23e8607f35f0265a23d481b338fb0e26aea/model.safetensors"

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True)

# 1. 用 from_pretrained 加载（看似已加载）
DTYPE = torch.bfloat16
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True,
    dtype=DTYPE, device_map="cuda"
)

# 2. 直接 load_state_dict 强制覆盖
print("Force loading state_dict...")
ckpt = st.load_file(CKPT)
# 移除 ckpt 里 model 没有的 keys
ckpt = {k: v for k, v in ckpt.items() if k in model.state_dict()}
# 转 dtype + device
ckpt = {k: v.to(DTYPE) for k, v in ckpt.items()}
incompatible = model.load_state_dict(ckpt, strict=False)
print(f"missing: {len(incompatible.missing_keys)}, unexpected: {len(incompatible.unexpected_keys)}")

model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
model = model.cuda()

# 验证 weight 现在对了
sd = model.state_dict()
print(f"\nVerify pre_encode.conv.0.weight std after manual load: {sd['encoder.pre_encode.conv.0.weight'].std().item():.4f}")

# 转录
wav, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
out = model.transcribe(processor=processor, language="en", audio_arrays=[wav], sample_rates=[sr])
print(f"\nHYP: {out[0][:300]}")
