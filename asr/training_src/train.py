"""
训练脚本 —— Whisper LoRA 微调
使用 Seq2SeqTrainer + PEFT + Accelerate + bitsandbytes (CUDA)
"""

import jiwer
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from config import ProjectConfig
from data import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_and_split_dataset,
    prepare_dataset,
)
from model import load_model_for_training, load_processor


def main():
    # ── 初始化配置 ──────────────────────────────────────────
    cfg = ProjectConfig()
    print(f"🔧 设备: {cfg.device}")
    print(f"🔧 量化: {'启用 4-bit' if cfg.use_quantization else '禁用'}")
    print(f"🔧 FP16: {cfg.use_fp16}")

    # ── 加载 Processor & Model ─────────────────────────────
    processor = load_processor(cfg)
    model = load_model_for_training(cfg)

    # ── 加载与预处理数据集 ──────────────────────────────────
    print("📦 加载数据集...")
    splits = load_and_split_dataset(cfg)
    print(f"   训练集: {len(splits['train'])} 条")
    print(f"   验证集: {len(splits['validation'])} 条")
    print(f"   测试集: {len(splits['test'])} 条")

    print("🔄 预处理数据集...")
    train_dataset = prepare_dataset(splits["train"], processor, cfg)
    eval_dataset = prepare_dataset(splits["validation"], processor, cfg)

    # ── Data Collator ─────────────────────────────────────
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── 评估指标：WER & CER（使用 jiwer）───────────────────
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # 将 -100 替换为 pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # 解码
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Whisper 标准化：移除标点、转小写
        pred_str = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]

        wer = jiwer.wer(label_str, pred_str)
        cer = jiwer.cer(label_str, pred_str)

        return {"wer": wer, "cer": cer}

    # ── 训练参数 ─────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        warmup_steps=cfg.training.warmup_steps,
        fp16=cfg.training.fp16,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        logging_dir=cfg.training.logging_dir,
        report_to=cfg.training.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # ── 早停回调 ─────────────────────────────────────────
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg.training.early_stopping_patience,
    )

    # ── Trainer ──────────────────────────────────────────
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

    # ── 开始训练 ─────────────────────────────────────────
    print("🚀 开始训练...")
    trainer.train()

    # ── 保存最佳 LoRA adapter ────────────────────────────
    best_model_dir = f"{cfg.training.output_dir}/best_adapter"
    model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"✅ 最佳模型已保存至: {best_model_dir}")


if __name__ == "__main__":
    main()
