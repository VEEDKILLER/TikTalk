"""双修：tokenizer.split_special_tokens + 手动 load_state_dict"""
import torch, librosa, safetensors.torch as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "CohereLabs/cohere-transcribe-03-2026"
CKPT = "/home/howdybunny/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/76b8b23e8607f35f0265a23d481b338fb0e26aea/model.safetensors"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
processor.tokenizer.split_special_tokens = False

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float32, device_map="cuda"
)

# 手动强制 load
ckpt = st.load_file(CKPT)
ckpt_f = {k: v.to(torch.float32) for k, v in ckpt.items() if k in model.state_dict()}
inc = model.load_state_dict(ckpt_f, strict=False)
print(f"Load: missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)}")

# 重新 tie lm head
model.log_softmax.mlp.layer0.weight = model.transf_decoder._embedding.token_embedding.weight

model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
model = model.cuda()

for fn in ["dataset/audio/4001_000.wav", "dataset/audio/4001_005.wav"]:
    wav, sr = librosa.load(fn, sr=16000)
    out = model.transcribe(processor=processor, language="en", audio_arrays=[wav], sample_rates=[sr])
    print(f"\n{fn}:\n  {out[0][:300]}")
