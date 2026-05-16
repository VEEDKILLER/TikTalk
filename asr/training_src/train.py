"""
训练脚本 —— 多模型 LoRA 微调
使用 Seq2SeqTrainer + PEFT + Accelerate (+ bitsandbytes on CUDA)

用法:
    python train.py --config configs/whisper_large_v3.yaml
    python train.py --config configs/qwen3_asr_1_7b.yaml
    python train.py --config configs/cohere_transcribe_2026.yaml
"""

import argparse

import jiwer
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from config import load_config
from data import load_and_split_dataset, prepare_dataset
from models import get_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="多模型 ASR LoRA 微调")
    parser.add_argument(
        "--config", required=True,
        help="YAML 配置文件路径，例如 configs/whisper_large_v3.yaml",
    )
    parser.add_argument(
        "--resume_from_checkpoint", default=None,
        help="从指定 checkpoint 目录恢复训练，例如 results/cohere-lora/checkpoint-1000",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="覆盖 yaml 的 num_train_epochs，强制只跑 N 步后停止（-1 表示不覆盖）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print(f"🔧 配置文件: {args.config}")
    print(f"🔧 模型:     {cfg.model.model_name}  (adapter={cfg.model.adapter})")
    print(f"🔧 设备:     {cfg.device}")
    print(f"🔧 量化:     {'启用 4-bit' if cfg.use_quantization else '禁用'}")
    print(f"🔧 FP16:     {cfg.use_fp16}")
    print(f"🔧 输出目录: {cfg.training.output_dir}")

    # ── 获取适配器 ────────────────────────────────────────────────────────
    adapter = get_adapter(cfg)

    # ── 加载 Processor & Model ────────────────────────────────────────────
    processor = adapter.load_processor(cfg)
    model = adapter.load_model_for_training(cfg)

    # ── 加载与预处理数据集 ────────────────────────────────────────────────
    print("📦 加载数据集...")
    splits = load_and_split_dataset(cfg)
    print(f"   训练集: {len(splits['train'])} 条")
    print(f"   验证集: {len(splits['validation'])} 条")
    print(f"   测试集: {len(splits['test'])} 条")

    print("🔄 预处理数据集...")
    train_dataset = prepare_dataset(splits["train"], processor, adapter, cfg)
    eval_dataset = prepare_dataset(splits["validation"], processor, adapter, cfg)

    # ── Data Collator ─────────────────────────────────────────────────────
    data_collator = adapter.build_data_collator(processor, model)

    # ── 评估指标：WER & CER ───────────────────────────────────────────────
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # predict_with_generate=False 时 predictions 是 logits (3D)，不是 token IDs
        # Decoder-only 模型（Qwen3）关闭 generate 时走此路径，直接返回空（只看 loss）
        if pred_ids.ndim == 3:
            return {}

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        # Seq2SeqTrainer 用 -100 padding 对齐生成序列；部分 tokenizer（如 Cohere 的
        # Rust 后端）将负数转成无符号整数时会 OverflowError，需先替换为 pad_token_id。
        pred_ids[pred_ids < 0] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalizer(t) for t in pred_str]
        label_str = [normalizer(t) for t in label_str]

        return {
            "wer": jiwer.wer(label_str, pred_str),
            "cer": jiwer.cer(label_str, pred_str),
        }

    # ── 训练参数 ──────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=cfg.training.warmup_steps,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        logging_dir=cfg.training.logging_dir,
        report_to=cfg.training.report_to,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        predict_with_generate=cfg.training.predict_with_generate,
        prediction_loss_only=cfg.training.prediction_loss_only,
        generation_max_length=225,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg.training.early_stopping_patience,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[early_stopping],
    )

    # ── 开始训练 ──────────────────────────────────────────────────────────
    print("🚀 开始训练...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── 保存最佳 LoRA adapter ─────────────────────────────────────────────
    best_model_dir = f"{cfg.training.output_dir}/best_adapter"
    model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"✅ 最佳模型已保存至: {best_model_dir}")


if __name__ == "__main__":
    main()
