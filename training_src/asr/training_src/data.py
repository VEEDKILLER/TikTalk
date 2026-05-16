"""
数据处理模块（通用部分）
仅负责从 metadata.csv 加载数据集并按固定比例划分，与具体模型无关。

模型专属的预处理逻辑（feature extraction、tokenize、DataCollator）
位于对应的 models/<name>_adapter.py 中。
"""

from pathlib import Path

from datasets import Dataset, load_dataset

from config import ProjectConfig


def load_and_split_dataset(cfg: ProjectConfig) -> dict:
    """
    从 metadata.csv 加载数据集并按 seed=42 划分为 train/val/test。

    使用 CSV 加载器读取 metadata.csv，音频文件在预处理阶段通过 soundfile
    按需加载，避免 cast_column / audiofolder 兼容性问题。

    Returns:
        dict with keys "train", "validation", "test", each a datasets.Dataset
    """
    ds = load_dataset(
        "csv",
        data_files=cfg.data.metadata_path,
        split="train",
    )

    # 过滤：仅保留音频文件实际存在的样本
    audio_dir = Path(cfg.data.audio_dir)
    ds = ds.filter(
        lambda row: (audio_dir / Path(row["file_name"]).name).exists(),
    )
    print(f"   过滤后保留 {len(ds)} 条有效样本（音频文件存在）")

    # 划分 train:val:test = 80:10:10
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


def prepare_dataset(dataset_split: Dataset, processor, adapter, cfg: ProjectConfig) -> Dataset:
    """
    对一个 split 的数据集应用模型专属的预处理。

    Args:
        dataset_split: 原始 split（含 file_name + transcription 列）
        processor:     模型的 processor
        adapter:       ModelAdapter 实例（提供 preprocess_example 实现）
        cfg:           项目配置
    """
    return dataset_split.map(
        lambda batch: adapter.preprocess_example(batch, processor, cfg),
        remove_columns=dataset_split.column_names,
    )
