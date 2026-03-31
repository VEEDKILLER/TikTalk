"""
数据处理模块 —— 数据集加载、预处理、划分与自定义 Data Collator。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, load_dataset
from transformers import WhisperProcessor

from config import ProjectConfig


def load_and_split_dataset(cfg: ProjectConfig) -> dict:
    """
    从 metadata.csv 加载数据集并按 seed=42 划分为 train/val/test。

    使用 CSV 加载器读取 metadata.csv，音频文件在预处理阶段
    通过 soundfile 按需加载，避免 cast_column / audiofolder 兼容性问题。

    Returns:
        dict with keys "train", "validation", "test", each a datasets.Dataset
    """
    ds = load_dataset(
        "csv",
        data_files=cfg.data.metadata_path,
        split="train",
    )

    # 过滤：仅保留音频文件实际存在的样本
    # （README：目前只挑选了5个样本作为代码训练测试）
    audio_dir = Path(cfg.data.audio_dir)
    ds = ds.filter(
        lambda row: (audio_dir / Path(row["file_name"]).name).exists(),
    )
    print(f"   过滤后保留 {len(ds)} 条有效样本（音频文件存在）")

    # 按比例划分：先分出 test，再从剩余中分出 validation
    # train:val:test = 80:10:10
    train_test = ds.train_test_split(
        test_size=cfg.data.test_ratio,
        seed=cfg.data.seed,
    )
    remaining_val_ratio = cfg.data.val_ratio / (cfg.data.train_ratio + cfg.data.val_ratio)
    train_val = train_test["train"].train_test_split(
        test_size=remaining_val_ratio,
        seed=cfg.data.seed,
    )

    return {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_test["test"],
    }


def preprocess_function(
    batch: Dict,
    processor: WhisperProcessor,
    audio_base_dir: str,
    sampling_rate: int,
) -> Dict:
    """
    预处理单个样本：用 soundfile 读取音频 → 提取 mel 特征 → tokenize 转录文本。
    """
    # file_name 字段格式: "audio/4001_000.wav"
    audio_path = Path(audio_base_dir) / Path(batch["file_name"]).name
    audio_array, sr = sf.read(str(audio_path), dtype="float32")

    # 如果采样率不匹配，进行简单重采样（librosa 风格线性插值）
    if sr != sampling_rate:
        import warnings
        warnings.warn(f"Audio {audio_path} has sr={sr}, resampling to {sampling_rate}")
        duration = len(audio_array) / sr
        target_length = int(duration * sampling_rate)
        audio_array = np.interp(
            np.linspace(0, len(audio_array) - 1, target_length),
            np.arange(len(audio_array)),
            audio_array,
        )

    # 提取 mel spectrogram 输入特征
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="np",
    ).input_features[0]

    # tokenize 转录文本作为标签
    labels = processor.tokenizer(batch["transcription"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


def prepare_dataset(
    dataset_split: Dataset,
    processor: WhisperProcessor,
    cfg: ProjectConfig,
) -> Dataset:
    """
    对一个 split 的数据集应用预处理。
    """
    dataset_split = dataset_split.map(
        lambda batch: preprocess_function(
            batch, processor, cfg.data.audio_dir, cfg.data.sampling_rate,
        ),
        remove_columns=dataset_split.column_names,
    )
    return dataset_split


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    自定义 Data Collator：
    - 将音频 input_features 补齐
    - 将标签序列的 padding token 替换为 -100，以屏蔽 Loss 计算
    """
    processor: WhisperProcessor
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离 input_features 和 labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # 将 padding token 替换为 -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 如果所有标签都以 bos token 开头，则移除它
        if (
            self.decoder_start_token_id is not None
            and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
