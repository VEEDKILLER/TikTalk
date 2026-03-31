"""
Model loading module —— Whisper model loading, quantization, and LoRA application.
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import ProjectConfig


def load_processor(cfg: ProjectConfig) -> WhisperProcessor:
    """Load WhisperProcessor (feature_extractor + tokenizer)"""
    processor = WhisperProcessor.from_pretrained(
        cfg.model.model_name,
        language=cfg.model.language,
        task=cfg.model.task,
    )
    return processor


def load_model_for_training(cfg: ProjectConfig) -> WhisperForConditionalGeneration:
    """
    Load the Whisper model for LoRA fine-tuning.
    - CUDA: enables bitsandbytes 4-bit quantization + prepare_model_for_kbit_training + LoRA
    - MPS/CPU: loads directly + LoRA (no quantization)
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
        # MPS/CPU: explicitly specify float32 to avoid dtype mismatch caused by whisper-large-v3's default fp16
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.model_name,
        **model_kwargs,
    )

    # Set generation config
    model.generation_config.language = cfg.model.language
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None

    # On CUDA, first run prepare_model_for_kbit_training, then apply LoRA
    if cfg.use_quantization:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
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
    Load the baseline model (no LoRA, no quantization, or low-precision loading as needed).
    Used for evaluation comparison.
    """
    import torch

    model_kwargs = {}

    if cfg.device == "cuda":
        # On 3080Ti 16GB, allow loading in FP16 or 8-bit to avoid OOM
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.model_name,
        **model_kwargs,
    )

    # Set generation config
    model.generation_config.language = cfg.model.language
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None

    return model
