"""
Project configuration module —— Whisper LoRA fine-tuning
Uses dataclasses to centrally manage all hyperparameters, paths, and device configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch


def _detect_device() -> str:
    """Auto-detect available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    """Model-related configuration"""
    model_name: str = "openai/whisper-large-v3"
    language: str = "english"
    task: str = "transcribe"


@dataclass
class LoRAConfig:
    """LoRA (PEFT) fine-tuning parameters"""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    output_dir: str = "./results/whisper-lora"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 20           # Upper limit; actual number of epochs controlled by early stopping
    warmup_steps: int = 50
    logging_steps: int = 10
    eval_steps: int = 50                 # Can be reduced for Mac testing
    save_steps: int = 50
    early_stopping_patience: int = 3
    fp16: bool = False                   # Dynamically set at runtime based on device
    seed: int = 42
    report_to: str = "tensorboard"
    logging_dir: str = "./results/logs"


@dataclass
class DataConfig:
    """Dataset path and split configuration"""
    audio_base_dir: str = "dataset"          # audiofolder load directory (contains metadata.csv + audio/)
    metadata_path: str = "dataset/metadata.csv"
    audio_dir: str = "dataset/audio"
    sampling_rate: int = 16_000
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class QuantizationConfig:
    """bitsandbytes 4-bit quantization (enabled only on CUDA)"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ProjectConfig:
    """Top-level project configuration, aggregating all sub-configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    device: str = field(default_factory=_detect_device)

    @property
    def use_quantization(self) -> bool:
        """Enable quantization only on CUDA devices"""
        return self.device == "cuda"

    @property
    def use_fp16(self) -> bool:
        """Enable FP16 only on CUDA devices"""
        return self.device == "cuda"

    def __post_init__(self):
        self.training.fp16 = self.use_fp16
        # Ensure output directories exist
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.logging_dir).mkdir(parents=True, exist_ok=True)
