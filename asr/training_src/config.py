"""
项目配置模块 —— Whisper LoRA 微调
使用 dataclass 统一管理所有超参数、路径与设备配置。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch


def _detect_device() -> str:
    """自动检测可用设备：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name: str = "openai/whisper-large-v3"
    language: str = "english"
    task: str = "transcribe"


@dataclass
class LoRAConfig:
    """LoRA (PEFT) 微调参数"""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """训练超参数"""
    output_dir: str = "./results/whisper-lora"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 20           # 上限，由早停控制实际轮数
    warmup_steps: int = 50
    logging_steps: int = 10
    eval_steps: int = 50                 # Mac 测试时可减小
    save_steps: int = 50
    early_stopping_patience: int = 3
    fp16: bool = False                   # 运行时根据设备动态设置
    seed: int = 42
    report_to: str = "tensorboard"
    logging_dir: str = "./results/logs"


@dataclass
class DataConfig:
    """数据集路径与划分配置"""
    audio_base_dir: str = "dataset"          # audiofolder 加载目录（含 metadata.csv + audio/）
    metadata_path: str = "dataset/metadata.csv"
    audio_dir: str = "dataset/audio"
    sampling_rate: int = 16_000
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class QuantizationConfig:
    """bitsandbytes 4-bit 量化（仅 CUDA 时启用）"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ProjectConfig:
    """项目总配置，聚合所有子配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    device: str = field(default_factory=_detect_device)

    @property
    def use_quantization(self) -> bool:
        """仅在 CUDA 设备上启用量化"""
        return self.device == "cuda"

    @property
    def use_fp16(self) -> bool:
        """仅在 CUDA 设备上启用 FP16"""
        return self.device == "cuda"

    def __post_init__(self):
        self.training.fp16 = self.use_fp16
        # 确保输出目录存在
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.logging_dir).mkdir(parents=True, exist_ok=True)
