"""
评估脚本 —— Baseline vs LoRA 微调模型对比
在测试集上分别运行两个模型，计算 WER/CER，输出对比结果。
"""

from pathlib import Path

import jiwer
import soundfile as sf
import torch
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import PeftModel

from config import ProjectConfig
from data import load_and_split_dataset
from model import load_baseline_model, load_processor


def transcribe_dataset(model, processor, dataset, device, audio_dir, sampling_rate=16000, batch_size=4):
    """
    用给定模型对数据集进行推理，返回预测文本列表。
    使用 soundfile 按需加载音频，与 data.py 保持一致。
    """
    model.eval()
    predictions = []
    audio_dir = Path(audio_dir)

    for i in tqdm(range(0, len(dataset), batch_size), desc="推理中"):
        batch = dataset[i : i + batch_size]
        file_names = batch["file_name"] if isinstance(batch["file_name"], list) else [batch["file_name"]]

        # 用 soundfile 加载音频并提取特征
        input_features_list = []
        for fname in file_names:
            audio_path = audio_dir / Path(fname).name
            audio_array, sr = sf.read(str(audio_path), dtype="float32")
            features = processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ).input_features
            input_features_list.append(features.squeeze(0))

        input_features = torch.stack(input_features_list).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=processor.tokenizer.language,
                task="transcribe",
                max_new_tokens=225,
            )

        decoded = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def main():
    cfg = ProjectConfig()
    device = cfg.device
    print(f"🔧 评估设备: {device}")

    processor = load_processor(cfg)
    normalizer = BasicTextNormalizer()

    # ── 加载测试集 ────────────────────────────────────────
    print("📦 加载数据集...")
    splits = load_and_split_dataset(cfg)
    test_dataset = splits["test"]
    print(f"   测试集: {len(test_dataset)} 条")

    # 提取真实标签
    references = test_dataset["transcription"]

    # ── Baseline 模型评估 ─────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("📊 评估 Baseline 模型（无微调）")
    print("═══════════════════════════════════════════")

    baseline_model = load_baseline_model(cfg)
    if device != "cuda":
        baseline_model = baseline_model.to(device)

    baseline_preds = transcribe_dataset(
        baseline_model, processor, test_dataset, device,
        cfg.data.audio_dir, cfg.data.sampling_rate,
    )
    del baseline_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # ── LoRA 微调模型评估 ─────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("📊 评估 LoRA 微调模型")
    print("═══════════════════════════════════════════")

    adapter_path = f"{cfg.training.output_dir}/best_adapter"

    from transformers import WhisperForConditionalGeneration

    # 加载基础模型
    model_kwargs = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    base_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.model_name,
        **model_kwargs,
    )
    base_model.generation_config.language = cfg.model.language
    base_model.generation_config.task = cfg.model.task
    base_model.generation_config.forced_decoder_ids = None

    # 加载 LoRA adapter
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    finetuned_model = finetuned_model.merge_and_unload()
    if device != "cuda":
        finetuned_model = finetuned_model.to(device)

    finetuned_preds = transcribe_dataset(
        finetuned_model, processor, test_dataset, device,
        cfg.data.audio_dir, cfg.data.sampling_rate,
    )
    del finetuned_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # ── 计算指标 ─────────────────────────────────────────
    # 标准化文本
    norm_refs = [normalizer(r) for r in references]
    norm_baseline = [normalizer(p) for p in baseline_preds]
    norm_finetuned = [normalizer(p) for p in finetuned_preds]

    baseline_wer = jiwer.wer(norm_refs, norm_baseline)
    baseline_cer = jiwer.cer(norm_refs, norm_baseline)
    finetuned_wer = jiwer.wer(norm_refs, norm_finetuned)
    finetuned_cer = jiwer.cer(norm_refs, norm_finetuned)

    # ── 输出对比结果 ─────────────────────────────────────
    print("\n")
    print("╔══════════════════════════════════════════════╗")
    print("║         模型对比评估结果                       ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  {'模型':<20} {'WER':>8} {'CER':>8}       ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  {'Baseline (原始)':<20} {baseline_wer:>8.4f} {baseline_cer:>8.4f}       ║")
    print(f"║  {'LoRA 微调':<20} {finetuned_wer:>8.4f} {finetuned_cer:>8.4f}       ║")
    print("╠══════════════════════════════════════════════╣")
    wer_improve = (baseline_wer - finetuned_wer) / baseline_wer * 100 if baseline_wer > 0 else 0
    cer_improve = (baseline_cer - finetuned_cer) / baseline_cer * 100 if baseline_cer > 0 else 0
    print(f"║  {'WER 提升':<20} {wer_improve:>+7.2f}%               ║")
    print(f"║  {'CER 提升':<20} {cer_improve:>+7.2f}%               ║")
    print("╚══════════════════════════════════════════════╝")

    # ── 打印部分样本对比 ─────────────────────────────────
    print("\n📝 部分样本对比（前 5 条）：")
    print("─" * 80)
    for i in range(min(5, len(references))):
        print(f"\n【样本 {i+1}】")
        print(f"  真实文本:   {references[i][:100]}...")
        print(f"  Baseline:   {baseline_preds[i][:100]}...")
        print(f"  LoRA 微调:  {finetuned_preds[i][:100]}...")


if __name__ == "__main__":
    main()
