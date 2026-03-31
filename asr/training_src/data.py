"""
Data processing module —— dataset loading, preprocessing, splitting, and custom Data Collator.
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
    Load the dataset from metadata.csv and split into train/val/test with seed=42.

    Uses the CSV loader to read metadata.csv; audio files are loaded on demand
    via soundfile during preprocessing, avoiding cast_column / audiofolder compatibility issues.

    Returns:
        dict with keys "train", "validation", "test", each a datasets.Dataset
    """
    ds = load_dataset(
        "csv",
        data_files=cfg.data.metadata_path,
        split="train",
    )

    # Filter: keep only samples whose audio files actually exist
    # (README: currently only 5 samples are selected for code training/testing)
    audio_dir = Path(cfg.data.audio_dir)
    ds = ds.filter(
        lambda row: (audio_dir / Path(row["file_name"]).name).exists(),
    )
    print(f"   Retained {len(ds)} valid samples after filtering (audio file exists)")

    # Split proportionally: first split off test, then split validation from the remainder
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
    Preprocess a single sample: read audio with soundfile → extract mel features → tokenize transcript.
    """
    # file_name 字段格式: "audio/4001_000.wav"
    audio_path = Path(audio_base_dir) / Path(batch["file_name"]).name
    audio_array, sr = sf.read(str(audio_path), dtype="float32")

    # If sample rate does not match, perform simple resampling (librosa-style linear interpolation)
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

    # Extract mel spectrogram input features
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="np",
    ).input_features[0]

    # Tokenize transcript text as labels
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
    Apply preprocessing to a dataset split.
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
    Custom Data Collator:
    - Pads audio input_features
    - Replaces padding tokens in label sequences with -100 to mask loss computation
    """
    processor: WhisperProcessor
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input_features and labels
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

        # Replace padding tokens with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If all labels start with bos token, remove it
        if (
            self.decoder_start_token_id is not None
            and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
