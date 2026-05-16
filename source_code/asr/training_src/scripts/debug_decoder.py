"""检查 decoder forward 是否正常，logits 分布正常吗？"""
import torch, librosa, safetensors.torch as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

CKPT = "/home/howdybunny/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/76b8b23e8607f35f0265a23d481b338fb0e26aea/model.safetensors"
MODEL = "CohereLabs/cohere-transcribe-03-2026"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32, device_map="cuda")

# 手动加载
ckpt = st.load_file(CKPT)
ckpt_f = {k: v.to(torch.float32) for k, v in ckpt.items() if k in model.state_dict()}
model.load_state_dict(ckpt_f, strict=False)
model.log_softmax.mlp.layer0.weight = model.transf_decoder._embedding.token_embedding.weight

model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
model = model.cuda()

# 手动 encode + 一步 decoder
wav, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
wav = wav[:5*sr]

inp = processor(audio=[wav], text=["<|startofcontext|>"], sampling_rate=16000, return_tensors="pt")
feat = inp["input_features"].to("cuda")
length = inp["length"].to("cuda")
input_ids = inp["input_ids"].to("cuda")
print(f"Input_ids: {input_ids.tolist()}")
print(f"Decoded start tokens: {[processor.tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

with torch.inference_mode():
    # Call forward
    out = model(input_ids=input_ids, input_features=feat, length=length)
    logits = out.logits if hasattr(out, "logits") else out
    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}, min={logits.min():.4f}, max={logits.max():.4f}")
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    topk = torch.topk(probs, 10)
    print("Top-10 predicted tokens:")
    for p, i in zip(topk.values.tolist(), topk.indices.tolist()):
        tok = processor.tokenizer.decode([i])
        print(f"  {i:6d} p={p:.4f}  {tok!r}")
