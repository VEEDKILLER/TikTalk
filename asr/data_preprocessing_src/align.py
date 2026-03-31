"""
Step 2: 音频-文本对齐
使用 stable-ts + faster-whisper medium 模型，将清洗后的文本与长音频进行毫秒级对齐。
输出: dataset_orig/OCSC/{group}/{id}_aligned.json

用法:
  uv run python align.py          # 处理全部文件
  uv run python align.py --test   # 仅处理第一个文件（测试）
"""

import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
import stable_whisper


# ── 项目路径 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_orig"
OCSC_DIR = DATASET_DIR / "OCSC"
GROUPS = ["4", "5", "6", "7", "8", "9"]


def find_audio_for_cha(cha_stem: str, group: str) -> Path | None:
    """根据 CHA 文件名找到对应的 MP3 音频。"""
    audio_path = DATASET_DIR / group / f"{cha_stem}.mp3"
    if audio_path.exists():
        return audio_path
    return None


def load_cleaned_text(cleaned_path: Path) -> str:
    """
    从清洗后的文本文件中加载所有句子，合并为一个完整文本。
    文件格式: SPEAKER\ttext
    """
    lines = []
    with open(cleaned_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                lines.append(parts[1])  # 只取文本部分
    return " ".join(lines)


def align_single(model, audio_path: Path, cleaned_path: Path, output_path: Path):
    """
    对单个音频文件进行对齐并保存结果。
    """
    text = load_cleaned_text(cleaned_path)
    if not text.strip():
        print(f"  ⚠ 跳过 {cleaned_path.stem}: 清洗后无有效文本")
        return

    # 使用 stable-ts align() 进行对齐
    result = model.align(
        str(audio_path),
        text,
        language="en",
        verbose=False,
    )

    # 提取 segment 级别的时间戳
    segments = []
    for seg in result.segments:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    # 保存 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def process_all(test_mode: bool = False):
    """处理所有（或第一个）音频文件。"""

    # 收集所有待处理的文件
    tasks = []
    for group in GROUPS:
        group_dir = OCSC_DIR / group
        if not group_dir.exists():
            continue
        for cleaned_path in sorted(group_dir.glob("*_cleaned.txt")):
            cha_stem = cleaned_path.stem.replace("_cleaned", "")
            audio_path = find_audio_for_cha(cha_stem, group)
            if audio_path is None:
                print(f"  ⚠ 跳过 {cha_stem}: 未找到对应音频")
                continue
            output_path = cleaned_path.with_name(f"{cha_stem}_aligned.json")
            # 跳过已处理的文件
            if output_path.exists():
                continue
            tasks.append((audio_path, cleaned_path, output_path))

    if not tasks:
        print("所有文件已对齐完成，无需重新处理。")
        return

    if test_mode:
        tasks = tasks[:1]
        print(f"[测试模式] 仅处理: {tasks[0][0].name}")

    print(f"待处理文件: {len(tasks)} 个")
    print("加载 faster-whisper medium 模型...")
    model = stable_whisper.load_faster_whisper("medium")
    print("模型加载完成!")

    for audio_path, cleaned_path, output_path in tqdm(tasks, desc="对齐中"):
        try:
            align_single(model, audio_path, cleaned_path, output_path)
        except Exception as e:
            print(f"\n  ✗ 对齐失败 {audio_path.name}: {e}")
            continue

    print("对齐完成!")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    process_all(test_mode=test_mode)
