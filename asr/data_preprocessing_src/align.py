"""
Step 2: Audio-text alignment
Uses stable-ts + faster-whisper medium model to align cleaned text with long audio at millisecond precision.
Output: dataset_orig/OCSC/{group}/{id}_aligned.json

Usage:
  uv run python align.py          # Process all files
  uv run python align.py --test   # Process only the first file (test mode)
"""

import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
import stable_whisper


# ── Project paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_orig"
OCSC_DIR = DATASET_DIR / "OCSC"
GROUPS = ["4", "5", "6", "7", "8", "9"]


def find_audio_for_cha(cha_stem: str, group: str) -> Path | None:
    """Find the corresponding MP3 audio file given a CHA filename stem."""
    audio_path = DATASET_DIR / group / f"{cha_stem}.mp3"
    if audio_path.exists():
        return audio_path
    return None


def load_cleaned_text(cleaned_path: Path) -> str:
    """
    Load all sentences from the cleaned text file and join them into a single string.
    File format: SPEAKER\ttext
    """
    lines = []
    with open(cleaned_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                lines.append(parts[1])  # Take only the text portion
    return " ".join(lines)


def align_single(model, audio_path: Path, cleaned_path: Path, output_path: Path):
    """
    Align a single audio file and save the result.
    """
    text = load_cleaned_text(cleaned_path)
    if not text.strip():
        print(f"  ⚠ Skipping {cleaned_path.stem}: no valid text after cleaning")
        return

    # Use stable-ts align() to perform alignment
    result = model.align(
        str(audio_path),
        text,
        language="en",
        verbose=False,
    )

    # Extract segment-level timestamps
    segments = []
    for seg in result.segments:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def process_all(test_mode: bool = False):
    """Process all (or just the first) audio files."""

    # Collect all files to process
    tasks = []
    for group in GROUPS:
        group_dir = OCSC_DIR / group
        if not group_dir.exists():
            continue
        for cleaned_path in sorted(group_dir.glob("*_cleaned.txt")):
            cha_stem = cleaned_path.stem.replace("_cleaned", "")
            audio_path = find_audio_for_cha(cha_stem, group)
            if audio_path is None:
                print(f"  ⚠ Skipping {cha_stem}: no corresponding audio found")
                continue
            output_path = cleaned_path.with_name(f"{cha_stem}_aligned.json")
            # Skip already-processed files
            if output_path.exists():
                continue
            tasks.append((audio_path, cleaned_path, output_path))

    if not tasks:
        print("All files are already aligned. Nothing to do.")
        return

    if test_mode:
        tasks = tasks[:1]
        print(f"[Test mode] Processing only: {tasks[0][0].name}")

    print(f"Files to process: {len(tasks)}")
    print("Loading faster-whisper medium model...")
    model = stable_whisper.load_faster_whisper("medium")
    print("Model loaded!")

    for audio_path, cleaned_path, output_path in tqdm(tasks, desc="Aligning"):
        try:
            align_single(model, audio_path, cleaned_path, output_path)
        except Exception as e:
            print(f"\n  ✗ Alignment failed for {audio_path.name}: {e}")
            continue

    print("Alignment complete!")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    process_all(test_mode=test_mode)
