"""
Step 3: 音频切片与 HuggingFace AudioFolder 数据集构建
根据对齐结果，切割音频为短片段，转换格式，生成 HuggingFace AudioFolder 数据集。

输出:
  clean_dataset/audio/{id}_{seg_idx:03d}.wav
  clean_dataset/metadata.csv

用法:
  uv run python build_dataset.py          # 处理全部文件
  uv run python build_dataset.py --test   # 仅处理第一个文件（测试）
"""

import csv
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


# ── 参数 ──────────────────────────────────────────────────────
MAX_SEGMENT_DURATION = 30.0   # 每个切片最大时长 (秒)
TARGET_SR = 16000             # Whisper 要求 16kHz
MIN_SEGMENT_DURATION = 0.1    # 过滤过短的 segment

# ── 项目路径 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_orig"
OCSC_DIR = DATASET_DIR / "OCSC"
OUTPUT_DIR = BASE_DIR / "clean_dataset"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
GROUPS = ["4", "5", "6", "7", "8", "9"]


def merge_segments(segments: list[dict], max_duration: float) -> list[dict]:
    """
    合并连续 segments，使每个合并后的切片时长 ≤ max_duration。
    过滤零时长或过短的 segment。
    """
    # 先过滤掉无效 segment
    valid_segments = [
        s for s in segments
        if (s["end"] - s["start"]) >= MIN_SEGMENT_DURATION
        and s["text"].strip()
    ]

    if not valid_segments:
        return []

    merged = []
    current = {
        "start": valid_segments[0]["start"],
        "end": valid_segments[0]["end"],
        "text": valid_segments[0]["text"],
    }

    for seg in valid_segments[1:]:
        potential_duration = seg["end"] - current["start"]
        if potential_duration <= max_duration:
            # 可以合并
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            # 不能合并，保存当前并开始新的
            merged.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }

    # 最后一个
    merged.append(current)

    return merged


def process_single_file(
    audio_path: Path,
    aligned_json_path: Path,
    file_id: str,
) -> list[dict]:
    """
    处理单个音频文件：切割、转码、保存。
    返回 metadata 条目列表。
    """
    # 加载对齐结果
    with open(aligned_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # 合并 segments
    merged = merge_segments(segments, MAX_SEGMENT_DURATION)
    if not merged:
        return []

    # 加载完整音频 (转为 16kHz 单声道)
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

    metadata_entries = []
    for idx, seg in enumerate(merged):
        start_sample = int(seg["start"] * TARGET_SR)
        end_sample = int(seg["end"] * TARGET_SR)

        # 防越界
        end_sample = min(end_sample, len(audio))
        if start_sample >= end_sample:
            continue

        chunk = audio[start_sample:end_sample]

        # 验证时长
        duration = len(chunk) / TARGET_SR
        if duration < MIN_SEGMENT_DURATION:
            continue

        # 保存 WAV
        wav_filename = f"{file_id}_{idx:03d}.wav"
        wav_path = AUDIO_OUTPUT_DIR / wav_filename
        sf.write(str(wav_path), chunk, TARGET_SR)

        metadata_entries.append({
            "file_name": f"audio/{wav_filename}",
            "transcription": seg["text"].strip(),
        })

    return metadata_entries


def get_completed_file_ids() -> set[str]:
    """
    从已有的 metadata.csv 中读取已完成的 file_id 集合。
    通过 file_name 列（如 audio/4001_000.wav）提取 file_id（如 4001）。
    """
    csv_path = OUTPUT_DIR / "metadata.csv"
    if not csv_path.exists():
        return set()

    completed = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # audio/4001_000.wav → 4001
            basename = Path(row["file_name"]).stem  # 4001_000
            file_id = basename.rsplit("_", 1)[0]     # 4001
            completed.add(file_id)
    return completed


def append_metadata(entries: list[dict], write_header: bool = False):
    """
    将 metadata 条目追加写入 metadata.csv。
    仅在文件不存在时写入表头。
    """
    csv_path = OUTPUT_DIR / "metadata.csv"
    mode = "a" if csv_path.exists() and not write_header else "w"
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(entries)


def process_all(test_mode: bool = False):
    """处理所有（或第一个）对齐后的文件，支持断点续传。"""

    # 确保输出目录存在
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取已完成的 file_id
    completed_ids = get_completed_file_ids()
    if completed_ids:
        print(f"已完成 {len(completed_ids)} 个文件，将自动跳过")

    # 收集所有待处理的文件
    tasks = []
    for group in GROUPS:
        group_dir = OCSC_DIR / group
        if not group_dir.exists():
            continue
        for aligned_path in sorted(group_dir.glob("*_aligned.json")):
            file_id = aligned_path.stem.replace("_aligned", "")
            # 跳过已处理的文件
            if file_id in completed_ids:
                continue
            audio_path = DATASET_DIR / group / f"{file_id}.mp3"
            if not audio_path.exists():
                print(f"  ⚠ 跳过 {file_id}: 未找到对应音频")
                continue
            tasks.append((audio_path, aligned_path, file_id))

    if not tasks:
        print("所有文件已处理完成，无需重新处理。")
        return

    if test_mode:
        tasks = tasks[:1]
        print(f"[测试模式] 仅处理: {tasks[0][2]}")

    print(f"待处理文件: {len(tasks)} 个")

    # 如果 metadata.csv 不存在，先写入表头
    csv_path = OUTPUT_DIR / "metadata.csv"
    need_header = not csv_path.exists()
    if need_header:
        append_metadata([], write_header=True)

    total_slices = 0
    for audio_path, aligned_path, file_id in tqdm(tasks, desc="构建数据集"):
        try:
            entries = process_single_file(audio_path, aligned_path, file_id)
            # 每处理完一个文件就追加写入 metadata.csv（断点续传的关键）
            if entries:
                append_metadata(entries)
                total_slices += len(entries)
        except Exception as e:
            print(f"\n  ✗ 处理失败 {file_id}: {e}")
            continue

    print(f"\n构建完成!")
    print(f"  本次新增切片: {total_slices}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  metadata: {csv_path}")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    process_all(test_mode=test_mode)
