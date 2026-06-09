"""
Cohere Transcribe 模型适配器
使用 AutoModelForSpeechSeq2Seq + AutoProcessor（trust_remote_code=True）。

注意事项：
1. lora.target_modules 已根据模型实际的 Linear 层名确定，写在
   configs/cohere_transcribe_2026.yaml 中；若更换模型，需检查模型结构后相应更新。
2. 确认 HF 仓库 ID 后更新 yaml 中的 model.model_name（在 HF 模型页 URL 中可见）。
3. 若模型推理接口不是标准 SpeechSeq2Seq（例如使用 CTC 或 decoder-only 架构），
   请根据模型 HF README 调整 transcribe_batch 中的调用方式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch

from models import ModelAdapter

if TYPE_CHECKING:
    from config import ProjectConfig


class CohereAdapter(ModelAdapter):

    # ── Processor ──────────────────────────────────────────────────────────

    def load_processor(self, cfg: "ProjectConfig"):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            cfg.model.model_name,
            trust_remote_code=True,
        )
        # tokenizer_config.json 里写了 split_special_tokens=True（给训练态用的），
        # 推理/构造 prompt 时必须关掉，否则 `<|startofcontext|>` 会被拆成 char 级 id，
        # decoder 被喂错误 prompt 后只会输出 `>>>>>` 或多语言乱码。
        processor.tokenizer.split_special_tokens = False

        # CohereAsrFeatureExtractor 有一个 `_filterbank` 懒加载属性（FilterbankFeatures
        # nn.Module 实例）。第一次调 processor 后会被填充，于是 save_pretrained 的
        # to_dict → json.dumps 走到它就抛 TypeError。patch 掉 to_dict 把它滤掉。
        fe = processor.feature_extractor
        _orig_to_dict = fe.to_dict

        def _patched_to_dict():
            d = _orig_to_dict()
            # _filterbank 是懒加载的 nn.Module；to_dict 是我们自己 patch 进
            # fe.__dict__ 的函数对象（原 to_dict 做的是 deepcopy(self.__dict__)）
            d.pop("_filterbank", None)
            d.pop("to_dict", None)
            # 兜底：过滤剩余不可 JSON 序列化的值
            import torch.nn as nn
            d = {
                k: v
                for k, v in d.items()
                if not callable(v) and not isinstance(v, nn.Module)
            }
            return d

        fe.to_dict = _patched_to_dict
        return processor

    # ── 模型加载 ────────────────────────────────────────────────────────────

    def _force_load_checkpoint(self, model, cfg: "ProjectConfig") -> None:
        """HF 5.x `from_pretrained` 在这个模型上静默只加载 ~40%
        （`Conv*` / `Linear` / `Embedding`—即 `_init_weights` 命中的模块—全部没写进去，
        LayerNorm / BatchNorm / pos_bias 能载）。手动补刀：从本地 safetensors
        load_state_dict(strict=False) 然后重新 tie lm_head。"""
        import safetensors.torch as st
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id=cfg.model.model_name, filename="model.safetensors"
        )
        ckpt = st.load_file(ckpt_path)
        sd_keys = set(model.state_dict().keys())
        # 跟 model param 对齐 dtype/device 做 copy，避免破坏 bf16/量化层的 storage
        filtered = {}
        for k, v in ckpt.items():
            if k not in sd_keys:
                continue
            target = model.state_dict()[k]
            filtered[k] = v.to(dtype=target.dtype, device=target.device)
        model.load_state_dict(filtered, strict=False)
        # load_state_dict 把 tied weight 拆成两份独立 tensor —— 重新 tie
        if hasattr(model, "log_softmax") and hasattr(model, "transf_decoder"):
            model.log_softmax.mlp.layer0.weight = (
                model.transf_decoder._embedding.token_embedding.weight
            )

    def _base_model_kwargs(self, cfg: "ProjectConfig") -> dict:
        kwargs = {"trust_remote_code": True}
        if cfg.device == "cuda":
            if cfg.use_quantization:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=cfg.quantization.load_in_4bit,
                    bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
                )
                kwargs["device_map"] = "auto"
            else:
                # fp16 会让 -1e9 attention-mask 值溢出 half range，必须 bf16。
                kwargs["torch_dtype"] = torch.bfloat16
                kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
            kwargs["device_map"] = None
        return kwargs

    def load_baseline_model(self, cfg: "ProjectConfig"):
        from transformers import AutoModelForSpeechSeq2Seq

        # Cohere baseline 推理不量化：4-bit NF4 在 Cohere 上累计误差过大，
        # baseline 会退化成"appropriate in..."这种英文 gibberish。
        # 1.5B 参数 bfloat16 仅 ~3GB，3080Ti 完全够用。
        model_kwargs = {"trust_remote_code": True}
        if cfg.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model.model_name, **model_kwargs
        )
        self._force_load_checkpoint(model, cfg)
        # bf16 加载下 BatchNorm 的 running_mean/var buffer 仍为 bf16，
        # 与 fp32 计算路径不匹配，会导致输出多语言乱码。强制 fp32。
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.float()
        return model

    def load_model_for_training(self, cfg: "ProjectConfig"):
        from transformers import AutoModelForSpeechSeq2Seq

        if cfg.use_quantization:
            raise RuntimeError(
                "Cohere 训练不能启用 4-bit 量化：from_pretrained 量化路径会静默只加载 "
                "~40% 权重（HF 5.x bug），之后 load_state_dict 写不进 Linear4bit。"
                "请在 configs/cohere_transcribe_2026.yaml 设 quantization.load_in_4bit: false。"
            )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model.model_name,
            **self._base_model_kwargs(cfg),
        )
        # 见 _force_load_checkpoint 注释：from_pretrained 只加载了 ~40% 权重
        self._force_load_checkpoint(model, cfg)
        # BatchNorm buffer 必须 fp32（bf16 autocast 下 buffer 仍是 bf16 会 dtype 冲突）
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()

        return self._apply_lora(model, cfg)

    def load_finetuned_model(self, cfg: "ProjectConfig", adapter_path: str):
        from transformers import AutoModelForSpeechSeq2Seq
        from peft import LoraConfig, get_peft_model
        import safetensors.torch as st

        # bfloat16 是必须的：Cohere 用 -1e9 作 attention mask padding，
        # fp16 (max 65504) 会溢出。
        model_kwargs = {"trust_remote_code": True}
        if cfg.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model.model_name, **model_kwargs
        )
        self._force_load_checkpoint(base, cfg)
        # BatchNorm buffer 必须 fp32（同 baseline）
        for m in base.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.float()

        # 不用 PeftModel.from_pretrained：它在 PEFT 0.19 + 旧格式 ckpt
        # (lora_A.weight, 无 .default.) 上要么静默 missing 要么和我们自己
        # 的 remap 叠加成 .default.default.，512 个键全部加载失败。
        # 改成：用 saved adapter_config 重建一个 PEFT wrapper，再手动
        # normalize keys 直接 load_state_dict。
        from peft import PeftConfig
        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        model = get_peft_model(base, peft_cfg)

        adapter_file = Path(adapter_path) / "adapter_model.safetensors"
        saved = st.load_file(str(adapter_file))

        # saved 形如:  base_model.model.X.lora_A.weight
        # model 期待:  base_model.model.X.lora_A.default.weight
        def _norm(k: str) -> str:
            if ".default." in k:
                return k
            return (k.replace(".lora_A.weight", ".lora_A.default.weight")
                     .replace(".lora_B.weight", ".lora_B.default.weight"))

        normalized = {_norm(k): v for k, v in saved.items()}
        model_keys = set(model.state_dict().keys())
        loaded_keys = model_keys & set(normalized.keys())
        if not loaded_keys:
            raise RuntimeError(
                f"LoRA 键名与模型完全不匹配。saved 示例: {next(iter(normalized))}; "
                f"model 期待示例: {next(iter([k for k in model_keys if 'lora_' in k]), '<no lora keys>')}"
            )
        # 用 load_state_dict(strict=False)，base 权重作为 missing 是预期的
        incompatible = model.load_state_dict(normalized, strict=False)
        unexpected_lora = [k for k in incompatible.unexpected_keys if "lora_" in k]
        if unexpected_lora:
            import warnings
            warnings.warn(
                f"{len(unexpected_lora)} 个 LoRA 键未加载，示例: {unexpected_lora[:3]}"
            )
        print(f"   ✅ 已加载 {len(loaded_keys)} / {len(saved)} 个 LoRA 权重")

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
        # 必须走 model.transcribe()（官方 API），原因：
        # 1. Cohere 是多任务模型，需要完整 decoder prompt
        #    <|startofcontext|><|startoftranscript|>...<|en|><|en|><|pnc|>...<|nodiarize|>
        #    才知道要"转录英语"。只给 decoder_start_token_id (<|startofcontext|>)
        #    baseline 会退化（只生成 `.`），LoRA 会生成多语言乱码。
        # 2. Cohere encoder forward 需要 length 参数来构造 cross-attention mask。
        #    length=None 时，padding 帧被当成有效音频，模型忽略真实音频。
        # transcribe() 内部处理 build_prompt、length、chunking、内部批处理，
        # 返回每条 audio 的转录字符串。
        return model.transcribe(
            processor=processor,
            language="en",
            audio_arrays=audio_arrays,
            sample_rates=[sampling_rate] * len(audio_arrays),
        )

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

        inputs = processor(
            audio_array,
            sampling_rate=cfg.data.sampling_rate,
            return_tensors="np",
        )
        input_features = inputs["input_features"][0]
        # length = 有效特征帧数；collator 会打包成 batch 的 length 张量，
        # encoder forward 靠它构造 cross-attention mask，不传就把 padding 帧当音频。
        feature_length = int(inputs["length"][0])

        # tokenizer 默认不会在尾部加 eos；不手动补的话 collator 的 trans_ids[:-1]
        # 会砍掉末词，且 labels 里永远不含 <|endoftext|>(3)，LoRA 学不到停止信号，
        # 推理时跑到 max_new_tokens 吐噪声（baseline WER 0.19 → LoRA WER 1.77）。
        labels = processor.tokenizer(batch["transcription"]).input_ids
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None and (len(labels) == 0 or labels[-1] != eos_id):
            labels = labels + [eos_id]
        return {
            "input_features": input_features,
            "feature_length": feature_length,
            "labels": labels,
        }

    def build_data_collator(self, processor, model):
        # 完整 prompt（等价于 model.build_prompt("en", punctuation=True)）
        # 必须训练时就按这个格式喂，否则 LoRA 学到的是"在 position 1 立即输出转录"的分布，
        # 推理时用完整 prompt 会扰乱语言/任务 token 的条件化，输出多语言乱码。
        prompt_text = (
            "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
            "<|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"
        )
        return CohereDataCollator(
            processor=processor,
            prompt_text=prompt_text,
        )


# ── Data Collator ─────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CohereDataCollator:
    """
    Cohere ASR 专用 DataCollator。

    关键设计：训练 decoder 输入必须包含完整 prompt
      <|startofcontext|><|startoftranscript|><|emo:undefined|>
      <|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
    这是 Cohere 多任务模型的必需条件化。若只给 decoder_start_token_id，LoRA 会学成
    "position 1 就吐转录"，推理用完整 prompt 时在 prompt 处理阶段抢跑，输出多语言乱码。

    prompt 位置的 loss 被 -100 屏蔽，只在转录位置计算训练信号。
    feature_length 从每个样本传进来，打包成 length 张量喂给 encoder 构造 cross-attn mask。
    """
    processor: Any
    prompt_text: str = ""
    _prompt_ids: torch.Tensor = None

    def _get_prompt_ids(self) -> torch.Tensor:
        if self._prompt_ids is None:
            # add_special_tokens=False：prompt 里的特殊 token 是字面 `<|...|>` 字符串，
            # 依赖 additional_special_tokens 映射到对应 id；不要让 tokenizer 再加 sot/eos
            self._prompt_ids = self.processor.tokenizer(
                self.prompt_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
        return self._prompt_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── input_features: 各样本形状 (freq, T_i)，zero-pad 到 T_max ──────
        raw_feats = [np.array(f["input_features"]) for f in features]
        t_max = max(x.shape[-1] for x in raw_feats)
        padded = np.zeros(
            (len(raw_feats), raw_feats[0].shape[0], t_max), dtype=np.float32
        )
        for i, x in enumerate(raw_feats):
            padded[i, :, : x.shape[-1]] = x
        input_features = torch.from_numpy(padded)
        length = torch.tensor(
            [f["feature_length"] for f in features], dtype=torch.long
        )

        # ── 转录 token：tokenizer.pad 后得到 (B, T_trans_max) ──────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        trans_ids = labels_batch["input_ids"]
        trans_mask = labels_batch["attention_mask"]
        # tokenizer 默认 add_special_tokens=True 会在头部加 <|startoftranscript|>，
        # 它已经是 prompt 的一部分，去掉避免重复。
        bos_tok = self.processor.tokenizer.bos_token_id
        if bos_tok is not None and trans_ids.shape[1] > 0 and (trans_ids[:, 0] == bos_tok).all().cpu().item():
            trans_ids = trans_ids[:, 1:]
            trans_mask = trans_mask[:, 1:]

        prompt_ids = self._get_prompt_ids()  # (P,)
        P = prompt_ids.shape[0]
        B = input_features.shape[0]

        # ── decoder_input_ids = [prompt, trans[:-1]]，长度 P + T - 1 ───────
        # 位置 P-1（prompt 最后一个 token）需要预测 trans[0]，这是第一个训练信号。
        # trans[:-1] 丢弃的是 eos / pad 尾部，模型从 t_{T-2} 的位置去预测 eos。
        prompt_batch = prompt_ids.unsqueeze(0).expand(B, -1)
        decoder_input_ids = torch.cat([prompt_batch, trans_ids[:, :-1]], dim=1)

        # ── labels：[-100 × (P-1), trans_with_pad_as_minus100] ─────────────
        # 前 P-1 位屏蔽（不训练 prompt→prompt 预测）；trans 部分保持 pad_mask 屏蔽。
        trans_labels = trans_ids.masked_fill(trans_mask.ne(1), -100)
        mask_prefix = torch.full((B, P - 1), -100, dtype=trans_labels.dtype)
        labels = torch.cat([mask_prefix, trans_labels], dim=1)

        return {
            "input_features": input_features,
            "length": length,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
