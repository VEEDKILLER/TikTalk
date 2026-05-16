"""手动 load 后再 tie weight，并验证 encoder 现在能区分输入。"""
import torch, librosa, numpy as np, safetensors.torch as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

CKPT = "/home/howdybunny/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/76b8b23e8607f35f0265a23d481b338fb0e26aea/model.safetensors"

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True,
    dtype=torch.float32, device_map="cuda"
)

ckpt = st.load_file(CKPT)
ckpt = {k: v.to(torch.float32) for k, v in ckpt.items() if k in model.state_dict()}
model.load_state_dict(ckpt, strict=False)

# Re-tie shared weights (load_state_dict 把 tied weights 拆成两份独立 tensors)
model.log_softmax.mlp.layer0.weight = model.transf_decoder._embedding.token_embedding.weight

model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
model = model.cuda()

# 验证 encoder 现在能区分
w1, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
w2 = np.zeros(5*16000, dtype=np.float32)
with torch.inference_mode():
    enc_outs = []
    for w in [w1[:5*sr], w2]:
        inp = processor(audio=[w], text=["<|startofcontext|>"], sampling_rate=16000, return_tensors="pt")
        feat = inp["input_features"].to("cuda")
        length = inp["length"].to("cuda")
        out = model.encoder(feat, length)
        if isinstance(out, tuple): out = out[0]
        enc_outs.append(out)
    diff = (enc_outs[0] - enc_outs[1]).abs()
    print(f"Encoder audio vs silence: max_diff={diff.max().item():.4f}, mean_diff={diff.mean().item():.4f}, std={enc_outs[0].std().item():.4f}")

# 转录测试
out = model.transcribe(processor=processor, language="en", audio_arrays=[w1], sample_rates=[sr])
print(f"\n4001_000 (full): {out[0][:300]}")

w3, _ = librosa.load("dataset/audio/4001_005.wav", sr=16000)
out = model.transcribe(processor=processor, language="en", audio_arrays=[w3], sample_rates=[16000])
print(f"\n4001_005 (full): {out[0][:300]}")
