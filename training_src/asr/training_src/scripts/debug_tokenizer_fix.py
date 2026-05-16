"""只修 tokenizer.split_special_tokens，不手动 load_state_dict，看能否转录。"""
import torch, librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "CohereLabs/cohere-transcribe-03-2026"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
processor.tokenizer.split_special_tokens = False

print("Tokenizer check: '<|startofcontext|>' =>", processor.tokenizer("<|startofcontext|>")["input_ids"])

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float32, device_map="cuda"
)
model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()

wav, sr = librosa.load("dataset/audio/4001_000.wav", sr=16000)
out = model.transcribe(processor=processor, language="en", audio_arrays=[wav], sample_rates=[sr])
print(f"\n4001_000: {out[0][:300]}")

wav2, _ = librosa.load("dataset/audio/4001_005.wav", sr=16000)
out = model.transcribe(processor=processor, language="en", audio_arrays=[wav2], sample_rates=[16000])
print(f"\n4001_005: {out[0][:300]}")
