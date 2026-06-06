"""手动加载 pre_encode + 任何被 from_pretrained 漏掉的权重，看 baseline 能否转录。"""
import torch, librosa, safetensors.torch as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

CKPT = "/home/howdybunny/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/76b8b23e8607f35f0265a23d481b338fb0e26aea/model.safetensors"

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True,
    dtype=torch.float32, device_map="auto"
)

# 检测哪些 weights 与 ckpt 不一致
ckpt = st.load_file(CKPT)
sd = model.state_dict()
mismatched = []
for k, v_ckpt in ckpt.items():
    if k in sd:
        v_model = sd[k].cpu().float()
        v_ck = v_ckpt.float()
        if v_model.shape != v_ck.shape:
            continue
        diff = (v_model - v_ck).abs().max().item()
        if diff > 1e-3:
            mismatched.append((k, diff))

print(f"Mismatched weights (diff > 1e-3): {len(mismatched)}")
for k, d in mismatched[:20]:
    print(f"  {k}: max_diff={d:.4f}")

# 手动加载 mismatched 的权重
print(f"\n正在手动加载 {len(mismatched)} 个权重...")
for k, _ in mismatched:
    target_param = dict(model.named_parameters()).get(k) or dict(model.named_buffers()).get(k)
    if target_param is None:
        continue
    src = ckpt[k].to(target_param.dtype).to(target_param.device)
    with torch.no_grad():
        target_param.copy_(src)

model.eval()
# BN fp32
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()

# 测试转录
wav, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
out = model.transcribe(processor=processor, language="en", audio_arrays=[wav], sample_rates=[sr])
print(f"\nHYP: {out[0][:300]}")
