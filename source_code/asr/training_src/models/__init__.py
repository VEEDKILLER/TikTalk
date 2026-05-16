"""
模型适配器注册表
每个 ASR 模型家族对应一个 ModelAdapter 子类，封装加载、推理与数据预处理逻辑。

使用方式:
    from models import get_adapter
    adapter = get_adapter(cfg)           # 根据 cfg.model.adapter 字段分发
    processor = adapter.load_processor(cfg)
    model     = adapter.load_baseline_model(cfg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from config import ProjectConfig


class ModelAdapter(ABC):
    """所有 ASR 模型适配器的抽象基类。"""

    # ── 模型 / Processor 加载 ─────────────────────────────────────────────

    @abstractmethod
    def load_processor(self, cfg: "ProjectConfig"):
        """返回 processor（feature_extractor + tokenizer）。"""

    @abstractmethod
    def load_baseline_model(self, cfg: "ProjectConfig"):
        """加载未经微调的基础模型（用于 baseline 评估）。"""

    @abstractmethod
    def load_model_for_training(self, cfg: "ProjectConfig"):
        """加载基础模型并应用 LoRA（用于微调训练）。"""

    @abstractmethod
    def load_finetuned_model(self, cfg: "ProjectConfig", adapter_path: str):
        """从 LoRA adapter 目录加载合并后的微调模型（用于评估）。"""

    # ── 推理 ──────────────────────────────────────────────────────────────

    @abstractmethod
    def transcribe_batch(
        self,
        model,
        processor,
        audio_arrays: List,
        sampling_rate: int,
        device: str,
    ) -> List[str]:
        """对一批音频数组进行推理，返回转录文本列表。"""

    # ── 数据预处理 ────────────────────────────────────────────────────────

    @abstractmethod
    def preprocess_example(
        self,
        batch: Dict[str, Any],
        processor,
        cfg: "ProjectConfig",
    ) -> Dict[str, Any]:
        """将单条样本（含 file_name + transcription）转换为模型输入。"""

    @abstractmethod
    def build_data_collator(self, processor, model) -> Any:
        """返回用于 Trainer 的 DataCollator。"""

    # ── 工具 ──────────────────────────────────────────────────────────────

    def _apply_lora(self, model, cfg: "ProjectConfig"):
        """给已加载的基础模型应用 LoRA，通用实现。子类可覆盖。"""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model


# ── 导入子类（延迟导入，避免循环依赖）────────────────────────────────────────

def get_adapter(cfg: "ProjectConfig") -> ModelAdapter:
    """根据 cfg.model.adapter 字段返回对应适配器实例。"""
    name = cfg.model.adapter.lower()
    if name == "whisper":
        from models.whisper_adapter import WhisperAdapter
        return WhisperAdapter()
    elif name == "qwen3":
        from models.qwen3_adapter import Qwen3Adapter
        return Qwen3Adapter()
    elif name == "cohere":
        from models.cohere_adapter import CohereAdapter
        return CohereAdapter()
    else:
        raise ValueError(
            f"未知 adapter: '{name}'。可选值: whisper | qwen3 | cohere"
        )
