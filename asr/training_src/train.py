"""
Training script —— Whisper LoRA fine-tuning
Uses Seq2SeqTrainer + PEFT + Accelerate + bitsandbytes (CUDA)
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
    # ── Initialize configuration ─────────────────────────────────────────────
    cfg = ProjectConfig()
    print(f"🔧 Device: {cfg.device}")
    print(f"🔧 Quantization: {'enabled (4-bit)' if cfg.use_quantization else 'disabled'}")
    print(f"🔧 FP16: {cfg.use_fp16}")

    # ── 加载 Processor & Model ─────────────────────────────
    processor = load_processor(cfg)
    model = load_model_for_training(cfg)

    # ── Load and preprocess dataset ──────────────────────────────────────────
    print("📦 Loading dataset...")
    splits = load_and_split_dataset(cfg)
    print(f"   Train: {len(splits['train'])} samples")
    print(f"   Validation: {len(splits['validation'])} samples")
    print(f"   Test: {len(splits['test'])} samples")

    print("🔄 Preprocessing dataset...")
    train_dataset = prepare_dataset(splits["train"], processor, cfg)
    eval_dataset = prepare_dataset(splits["validation"], processor, cfg)

    # ── Data Collator ─────────────────────────────────────
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Evaluation metrics: WER & CER (using jiwer) ─────────────────────────
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Whisper normalization: remove punctuation, convert to lowercase
        pred_str = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]

        wer = jiwer.wer(label_str, pred_str)
        cer = jiwer.cer(label_str, pred_str)

        return {"wer": wer, "cer": cer}

    # ── Training arguments ───────────────────────────────────────────────────
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

    # ── Early stopping callback ──────────────────────────────────────────────
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

    # ── Start training ───────────────────────────────────────────────────────
    print("🚀 Starting training...")
    trainer.train()

    # ── Save best LoRA adapter ───────────────────────────────────────────────
    best_model_dir = f"{cfg.training.output_dir}/best_adapter"
    model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"✅ Best model saved to: {best_model_dir}")


if __name__ == "__main__":
    main()
