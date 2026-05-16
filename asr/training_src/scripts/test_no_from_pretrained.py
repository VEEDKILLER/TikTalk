"""不调 from_pretrained，直接构造模型 + load_state_dict。"""
import torch, librosa, safetensors.torch as st
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

CKPT = "/home/howdybunny/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/76b8b23e8607f35f0265a23d481b338fb0e26aea/model.safetensors"
MODEL_NAME = "CohereLabs/cohere-transcribe-03-2026"

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 直接拿到模型类
model_cls = get_class_from_dynamic_module(
    "modeling_cohere_asr.CohereAsrForConditionalGeneration",
    MODEL_NAME,
)
print(f"Model class: {model_cls.__name__}")

# 在 meta device 创建空模型 → load_state_dict
model = model_cls(config)

print("Loading state dict...")
ckpt = st.load_file(CKPT)
ckpt = {k: v.to(torch.float32) for k, v in ckpt.items() if k in model.state_dict()}
inc = model.load_state_dict(ckpt, strict=False)
print(f"missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)}")

# 验证 weight 加载了
sd = model.state_dict()
print(f"pre_encode.conv.0.weight std after load: {sd['encoder.pre_encode.conv.0.weight'].std().item():.4f}")
print(f"log_softmax.mlp.layer0.weight std after load: {sd['log_softmax.mlp.layer0.weight'].std().item():.4f}")

model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
model = model.cuda()

# 转录
wav, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
out = model.transcribe(processor=processor, language="en", audio_arrays=[wav], sample_rates=[sr])
print(f"\nHYP: {out[0][:300]}")
