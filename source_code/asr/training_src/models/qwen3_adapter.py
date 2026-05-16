"""
Qwen3-ASR 模型适配器

架构说明：
  Qwen3-ASR 是 decoder-only 因果语言模型（非 seq2seq），由两个子模块组成：
    - audio_tower: 音频编码器（提取音频特征）
    - model (thinker): Qwen3 文本解码器（自回归生成）

  训练时使用 Qwen3ASRThinkerForConditionalGeneration（有正确的 forward/labels→loss），
  不使用 Qwen3ASRForConditionalGeneration 顶层包装器（只有 generate，无 forward）。

  输入格式（序列 = 音频前缀 + 转录文本）：
    input_ids            : [<|audio_start|> <|audio_pad|>×N <|audio_end|>  t1 t2 ... tn <eos>]
    labels               : [        -100        ×(N+2)                     t1 t2 ... tn <eos>]
    input_features       : (128, T) 变长 mel 特征
    feature_attention_mask: (T,)    音频帧有效 mask

依赖：
  需安装 qwen-asr 包（pip install qwen-asr），该包会把 transformers 固定到 4.57.6。
  因此 Cohere（需要 transformers>=5.4.0）和 Qwen3-ASR 必须使用独立环境。
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


class Qwen3Adapter(ModelAdapter):

    # ── Processor ──────────────────────────────────────────────────────────

    def load_processor(self, cfg: "ProjectConfig"):
        import qwen_asr  # noqa: F401 — 触发 qwen3_asr 模型类型注册
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        return Qwen3ASRProcessor.from_pretrained(cfg.model.model_name)

    # ── 模型加载 ────────────────────────────────────────────────────────────

    def _load_thinker(self, cfg: "ProjectConfig", quantize: bool = True):
        """
        加载 Qwen3ASRThinkerForConditionalGeneration（有 forward() + labels→loss）。
        顶层 Qwen3ASRForConditionalGeneration 只有 generate()，不能直接用于训练。

        Args:
            quantize: 是否启用 4-bit 量化。合并 LoRA 时必须传 False，
                      因为量化权重的内部 shape 经过 pack/重排，
                      merge_and_unload() 无法对其做张量加法。
        """
        import qwen_asr  # noqa: F401
        from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration

        kwargs = {}
        if cfg.device == "cuda":
            if cfg.use_quantization and quantize:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=cfg.quantization.load_in_4bit,
                    bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
                )
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = torch.bfloat16
                kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
            kwargs["device_map"] = None

        full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            cfg.model.model_name, **kwargs
        )
        # 取出 thinker 子模块（含 audio_tower + text decoder + lm_head）
        return full_model.thinker

    def load_baseline_model(self, cfg: "ProjectConfig"):
        return self._load_thinker(cfg)

    def load_model_for_training(self, cfg: "ProjectConfig"):
        thinker = self._load_thinker(cfg)

        if cfg.use_quantization:
            from peft import prepare_model_for_kbit_training
            thinker = prepare_model_for_kbit_training(
                thinker,
                use_gradient_checkpointing=cfg.training.gradient_checkpointing,
            )

        # gradient checkpointing 要求输入 embedding 带 requires_grad，
        # prepare_model_for_kbit_training 在某些架构下不能正确注册 hook，
        # 显式调用确保梯度能传到 checkpoint 节点。
        if cfg.training.gradient_checkpointing:
            thinker.enable_input_require_grads()

        peft_thinker = self._apply_lora(thinker, cfg)

        # bitsandbytes 量化后会在 config 及子 config 里注入
        # _pre_quantization_dtype = torch.dtype 对象，Python 的 json.dumps
        # 无法序列化它，会导致 TensorBoard on_train_begin callback 崩溃。
        # 将所有 torch.dtype 属性转为字符串表示。
        if cfg.use_quantization:
            _fix_config_dtypes(peft_thinker.config)

        return peft_thinker

    def load_finetuned_model(self, cfg: "ProjectConfig", adapter_path: str):
        from peft import PeftModel
        # merge_and_unload() 需要全精度权重（bfloat16），不能用 4-bit 量化加载：
        # 量化后权重被 pack 成 uint8，形状发生变化，与 LoRA 矩阵做加法时 shape 不匹配。
        thinker = self._load_thinker(cfg, quantize=False)
        thinker = PeftModel.from_pretrained(thinker, adapter_path)
        thinker = thinker.merge_and_unload()
        if cfg.device not in ("cuda",):
            thinker = thinker.to(cfg.device)
        return thinker

    # ── 推理 ───────────────────────────────────────────────────────────────

    def transcribe_batch(
        self,
        model,
        processor,
        audio_arrays: List,
        sampling_rate: int,
        device: str,
    ) -> List[str]:
        prompt = (
            f"{processor.audio_bos_token}"
            f"{processor.audio_token}"
            f"{processor.audio_eos_token}"
        )
        inputs = processor(
            text=[prompt] * len(audio_arrays),
            audio=audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # processor 始终返回 float32；模型权重 compute dtype 是 float16。
        # 4-bit 量化后参数存储为 uint8，next(model.parameters()).dtype 会返回 uint8，
        # 必须跳过量化参数，取第一个浮点参数的 dtype。
        model_dtype = next(
            (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16, torch.float32)),
            torch.float16,
        )
        inputs["input_features"] = inputs["input_features"].to(model_dtype)

        with torch.no_grad():
            # decoder-only 模型的 generate() 直接返回 tensor (batch, full_seq_len)，
            # 不是有 .sequences 属性的 GenerateOutput
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=225,
            )

        # 去掉 input_ids 长度的前缀，只解码生成部分
        prefix_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prefix_len:]
        return processor.tokenizer.batch_decode(generated, skip_special_tokens=True)

    # ── 数据预处理 ─────────────────────────────────────────────────────────

    def preprocess_example(
        self,
        batch: Dict[str, Any],
        processor,
        cfg: "ProjectConfig",
    ) -> Dict[str, Any]:
        # 加载音频
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

        # 音频前缀 prompt（<|audio_start|><|audio_pad|><|audio_end|>）
        prompt = (
            f"{processor.audio_bos_token}"
            f"{processor.audio_token}"
            f"{processor.audio_eos_token}"
        )

        # 通过 Processor 得到扩展后的 input_ids（音频 pad 已展开为 N 个 token）
        audio_out = processor(
            text=prompt,
            audio=audio_array,
            sampling_rate=cfg.data.sampling_rate,
            return_tensors="np",
        )
        # input_ids shape: (1, prefix_len)  — 包含 BOS/EOS/pad audio tokens
        prefix_ids = audio_out["input_ids"][0].tolist()
        prefix_len = len(prefix_ids)
        input_features = audio_out["input_features"][0]          # (128, T)
        feature_attention_mask = audio_out["feature_attention_mask"][0]  # (T,)

        # 转录文本 tokenize（不加 special tokens，手动附加 EOS）
        eos_id = processor.tokenizer.eos_token_id
        transcription_ids = (
            processor.tokenizer(
                batch["transcription"], add_special_tokens=False
            ).input_ids
            + [eos_id]
        )

        # 拼接完整序列
        full_input_ids = prefix_ids + transcription_ids
        full_attention_mask = [1] * len(full_input_ids)
        # labels: 前缀部分 -100（不参与 loss），转录部分保留
        labels = [-100] * prefix_len + transcription_ids

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
        }

    def build_data_collator(self, processor, model):
        pad_id = processor.tokenizer.pad_token_id or 0
        return Qwen3DataCollator(processor=processor, pad_token_id=pad_id)


# ── Data Collator ─────────────────────────────────────────────────────────────

@dataclass
class Qwen3DataCollator:
    """
    Qwen3-ASR 专用 DataCollator。

    需要对以下四个变长字段做 padding：
      - input_ids / attention_mask / labels : 文本序列（右 pad）
      - input_features                      : (128, T) 音频特征（右 zero-pad）
      - feature_attention_mask              : (T,) 音频帧 mask（右 zero-pad）
    """
    processor: Any
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── 文本序列 padding ────────────────────────────────────────────────
        max_text_len = max(len(f["input_ids"]) for f in features)
        input_ids_list, attn_mask_list, labels_list = [], [], []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_text_len - seq_len

            input_ids_list.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attn_mask_list.append(f["attention_mask"] + [0] * pad_len)
            labels_list.append(f["labels"] + [-100] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)

        # ── 音频特征 padding (128, T_i) → (128, T_max) ─────────────────────
        raw_feats = [np.array(f["input_features"]) for f in features]
        t_max = max(x.shape[-1] for x in raw_feats)
        padded = np.zeros((len(raw_feats), raw_feats[0].shape[0], t_max), dtype=np.float32)
        for i, x in enumerate(raw_feats):
            padded[i, :, : x.shape[-1]] = x
        input_features = torch.from_numpy(padded)

        # ── 音频帧 mask padding (T_i,) → (T_max,) ──────────────────────────
        raw_masks = [np.array(f["feature_attention_mask"]) for f in features]
        feat_masks = np.zeros((len(raw_masks), t_max), dtype=np.int32)
        for i, m in enumerate(raw_masks):
            feat_masks[i, : len(m)] = m
        feature_attention_mask = torch.from_numpy(feat_masks)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
        }


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _fix_config_dtypes(config) -> None:
    """
    将 config 及其子 config 中所有 torch.dtype 属性转为字符串。

    bitsandbytes 量化后会注入 _pre_quantization_dtype = torch.float16 等
    torch.dtype 对象到 model.config（及嵌套子 config）。Python json.dumps
    不认识 torch.dtype，导致 TensorBoard on_train_begin callback 崩溃。
    """
    for attr, val in list(vars(config).items()):
        if isinstance(val, torch.dtype):
            setattr(config, attr, str(val).replace("torch.", ""))
        elif hasattr(val, "__dict__") and hasattr(val, "to_dict"):
            # 递归处理嵌套 config（如 audio_config, text_config）
            _fix_config_dtypes(val)
