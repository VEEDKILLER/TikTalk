"""
模型加载模块 —— Whisper 模型加载、量化、LoRA 应用。
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import ProjectConfig


def load_processor(cfg: ProjectConfig) -> WhisperProcessor:
    """加载 WhisperProcessor（feature_extractor + tokenizer）"""
    processor = WhisperProcessor.from_pretrained(
        cfg.model.model_name,
        language=cfg.model.language,
        task=cfg.model.task,
    )
    return processor


def load_model_for_training(cfg: ProjectConfig) -> WhisperForConditionalGeneration:
    """
    加载用于 LoRA 微调的 Whisper 模型。
    - CUDA：启用 bitsandbytes 4-bit 量化 + prepare_model_for_kbit_training + LoRA
    - MPS/CPU：直接加载 + LoRA（无量化）
    """
    model_kwargs = {}

    if cfg.use_quantization:
        from transformers import BitsAndBytesConfig
        import torch

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        import torch
        # MPS/CPU: 显式指定 float32 避免 whisper-large-v3 默认的 fp16 导致 dtype 不匹配
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.model_name,
        **model_kwargs,
    )

    # 设置 generation config
    model.generation_config.language = cfg.model.language
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None

    # CUDA 上先 prepare_model_for_kbit_training，再应用 LoRA
    if cfg.use_quantization:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # 应用 LoRA
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


def load_baseline_model(cfg: ProjectConfig) -> WhisperForConditionalGeneration:
    """
    加载 baseline 模型（无 LoRA、无量化或按需低精度加载）。
    用于评估对比。
    """
    import torch

    model_kwargs = {}

    if cfg.device == "cuda":
        # 在 3080Ti 16G 上，允许以 FP16 或 8-bit 加载以避免 OOM
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        # MPS/CPU: 显式指定 float32 避免 whisper-large-v3 默认的 fp16 导致 dtype 不匹配
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.model_name,
        **model_kwargs,
    )

    # 设置 generation config
    model.generation_config.language = cfg.model.language
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None

    return model
