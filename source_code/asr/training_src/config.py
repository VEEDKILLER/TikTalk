"""
项目配置模块
使用 dataclass 统一管理所有超参数、路径与设备配置。
支持从 YAML 文件加载（多模型消融实验），也可直接实例化（向后兼容）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import yaml


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
    # 用于多模型框架：指定使用哪个 adapter 类
    # 可选值: "whisper" | "qwen3" | "cohere"
    adapter: str = "whisper"
    # HuggingFace 仓库 ID（通常与 model_name 相同，但可单独覆盖）
    hf_id: Optional[str] = None
    language: str = "english"
    task: str = "transcribe"

    def __post_init__(self):
        if self.hf_id is None:
            self.hf_id = self.model_name


@dataclass
class LoRAConfig:
    """LoRA (PEFT) 微调参数"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """训练超参数"""
    output_dir: str = "./results/whisper-lora"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    num_train_epochs: int = 5
    warmup_steps: int = 50
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    early_stopping_patience: int = 3
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    predict_with_generate: bool = True
    prediction_loss_only: bool = False
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    seed: int = 42
    report_to: str = "tensorboard"
    logging_dir: str = "./results/logs"


@dataclass
class DataConfig:
    """数据集路径与划分配置"""
    audio_base_dir: str = "dataset"
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
        """CUDA + yaml 显式 load_in_4bit 才启用（Cohere 必须关：
        from_pretrained 量化路径不能后补权重，导致 base 仍是 60% 随机初始化）"""
        return self.device == "cuda" and self.quantization.load_in_4bit

    @property
    def use_fp16(self) -> bool:
        """仅在 CUDA 设备上启用 FP16"""
        return self.device == "cuda"

    def __post_init__(self):
        # bf16 和 fp16 互斥：若显式请求 bf16，则关闭 fp16 自动推断
        if self.training.bf16:
            self.training.fp16 = False
        else:
            self.training.fp16 = self.use_fp16
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.logging_dir).mkdir(parents=True, exist_ok=True)


# ── YAML 加载 ─────────────────────────────────────────────────────────────────

def _deep_update(base: dict, override: dict) -> dict:
    """将 override 深度合并到 base，返回新 dict。"""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(yaml_path: str) -> ProjectConfig:
    """
    从 YAML 文件加载配置并构建 ProjectConfig。
    YAML 结构与 dataclass 嵌套对应，仅需填写需要覆盖的字段。

    示例 yaml 结构:
        model:
          adapter: qwen3
          model_name: Qwen/Qwen3-ASR-1.7B
        lora:
          target_modules: [q_proj, k_proj, v_proj, o_proj]
        training:
          output_dir: ./results/qwen3-lora
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 为各子配置构建 dict，仅覆盖 yaml 中出现的字段
    def _build(cls, section_dict: dict):
        import dataclasses
        defaults = {f.name: f.default if f.default is not dataclasses.MISSING
                    else f.default_factory()
                    for f in dataclasses.fields(cls)
                    if f.name != "device"}
        merged = _deep_update(defaults, section_dict)
        # 只传 cls 接受的字段，并按 field type 做基础类型强转
        # （PyYAML 6.x 中无小数点的科学记数法如 1e-4 会被解析为字符串）
        field_map = {f.name: f for f in dataclasses.fields(cls)}
        coerced = {}
        for k, v in merged.items():
            if k not in field_map:
                continue
            ftype = field_map[k].type
            if ftype is float and isinstance(v, str):
                v = float(v)
            elif ftype is int and isinstance(v, str):
                v = int(v)
            coerced[k] = v
        return cls(**coerced)

    model_cfg = _build(ModelConfig, raw.get("model", {}))
    lora_cfg = _build(LoRAConfig, raw.get("lora", {}))
    training_cfg = _build(TrainingConfig, raw.get("training", {}))
    data_cfg = _build(DataConfig, raw.get("data", {}))
    quant_cfg = _build(QuantizationConfig, raw.get("quantization", {}))

    cfg = ProjectConfig(
        model=model_cfg,
        lora=lora_cfg,
        training=training_cfg,
        data=data_cfg,
        quantization=quant_cfg,
    )
    return cfg
