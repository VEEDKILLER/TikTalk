"""
Whisper 模型适配器
迁移自原 model.py + data.py 中的 Whisper 专用逻辑，行为与原版完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch

from models import ModelAdapter

if TYPE_CHECKING:
    from config import ProjectConfig


class WhisperAdapter(ModelAdapter):

    # ── Processor ──────────────────────────────────────────────────────────

    def load_processor(self, cfg: "ProjectConfig"):
        from transformers import WhisperProcessor
        return WhisperProcessor.from_pretrained(
            cfg.model.model_name,
            language=cfg.model.language,
            task=cfg.model.task,
        )

    # ── 模型加载 ────────────────────────────────────────────────────────────

    def load_baseline_model(self, cfg: "ProjectConfig"):
        from transformers import WhisperForConditionalGeneration

        model_kwargs = {}
        if cfg.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = None

        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model.model_name, **model_kwargs
        )
        model.generation_config.language = cfg.model.language
        model.generation_config.task = cfg.model.task
        model.generation_config.forced_decoder_ids = None
        return model

    def load_model_for_training(self, cfg: "ProjectConfig"):
        from transformers import WhisperForConditionalGeneration

        model_kwargs = {}
        if cfg.use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=cfg.quantization.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = None

        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model.model_name, **model_kwargs
        )
        model.generation_config.language = cfg.model.language
        model.generation_config.task = cfg.model.task
        model.generation_config.forced_decoder_ids = None

        if cfg.use_quantization:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)

        return self._apply_lora(model, cfg)

    def load_finetuned_model(self, cfg: "ProjectConfig", adapter_path: str):
        from transformers import WhisperForConditionalGeneration
        from peft import PeftModel

        model_kwargs = {}
        if cfg.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base = WhisperForConditionalGeneration.from_pretrained(
            cfg.model.model_name, **model_kwargs
        )
        base.generation_config.language = cfg.model.language
        base.generation_config.task = cfg.model.task
        base.generation_config.forced_decoder_ids = None

        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        if cfg.device != "cuda":
            model = model.to(cfg.device)
        return model

    # ── 推理 ───────────────────────────────────────────────────────────────

    def transcribe_batch(
        self,
        model,
        processor,
        audio_arrays: List,
        sampling_rate: int,
        device: str,
    ) -> List[str]:
        input_features_list = []
        for audio_array in audio_arrays:
            features = processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ).input_features
            input_features_list.append(features.squeeze(0))

        input_features = torch.stack(input_features_list).to(device)
        model_dtype = next(model.parameters()).dtype
        input_features = input_features.to(dtype=model_dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=processor.tokenizer.language,
                task="transcribe",
                max_new_tokens=225,
            )

        return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    # ── 数据预处理 ─────────────────────────────────────────────────────────

    def preprocess_example(
        self,
        batch: Dict[str, Any],
        processor,
        cfg: "ProjectConfig",
    ) -> Dict[str, Any]:
        audio_path = Path(cfg.data.audio_dir) / Path(batch["file_name"]).name
        audio_array, sr = sf.read(str(audio_path), dtype="float32")

        if sr != cfg.data.sampling_rate:
            import warnings
            warnings.warn(f"{audio_path} sr={sr}, resampling to {cfg.data.sampling_rate}")
            target_length = int(len(audio_array) / sr * cfg.data.sampling_rate)
            audio_array = np.interp(
                np.linspace(0, len(audio_array) - 1, target_length),
                np.arange(len(audio_array)),
                audio_array,
            )

        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=cfg.data.sampling_rate,
            return_tensors="np",
        ).input_features[0]

        labels = processor.tokenizer(batch["transcription"]).input_ids
        return {"input_features": input_features, "labels": labels}

    def build_data_collator(self, processor, model) -> "WhisperDataCollator":
        return WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )


# ── Data Collator ─────────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """动态 padding：补齐 input_features，标签 padding 替换为 -100。"""
    processor: Any
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (
            self.decoder_start_token_id is not None
            and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
