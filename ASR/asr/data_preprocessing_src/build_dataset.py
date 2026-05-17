"""
Step 3: Audio slicing and HuggingFace AudioFolder dataset construction
Based on alignment results, slice audio into short segments, convert format, and generate a HuggingFace AudioFolder dataset.

Output:
  clean_dataset/audio/{id}_{seg_idx:03d}.wav
  clean_dataset/metadata.csv

Usage:
  uv run python build_dataset.py          # Process all files
  uv run python build_dataset.py --test   # Process only the first file (test mode)
"""

import csv
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


# ── Parameters ───────────────────────────────────────────────
MAX_SEGMENT_DURATION = 30.0   # Maximum duration per slice (seconds)
TARGET_SR = 16000             # Whisper requires 16 kHz
MIN_SEGMENT_DURATION = 0.1    # Filter out segments shorter than this

# ── Project paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_orig"
OCSC_DIR = DATASET_DIR / "OCSC"
OUTPUT_DIR = BASE_DIR / "clean_dataset"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
GROUPS = ["4", "5", "6", "7", "8", "9"]


def merge_segments(segments: list[dict], max_duration: float) -> list[dict]:
    """
    Merge consecutive segments so that each merged slice duration ≤ max_duration.
    Filters out zero-duration or too-short segments.
    """
    # Filter out invalid segments first
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
            # Can merge
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            # Cannot merge; save current and start a new one
            merged.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }

    # Append the last segment
    merged.append(current)

    return merged


def process_single_file(
    audio_path: Path,
    aligned_json_path: Path,
    file_id: str,
) -> list[dict]:
    """
    Process a single audio file: slice, transcode, and save.
    Returns a list of metadata entries.
    """
    # Load alignment result
    with open(aligned_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Merge segments
    merged = merge_segments(segments, MAX_SEGMENT_DURATION)
    if not merged:
        return []

    # Load full audio (resample to 16 kHz mono)
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

    metadata_entries = []
    for idx, seg in enumerate(merged):
        start_sample = int(seg["start"] * TARGET_SR)
        end_sample = int(seg["end"] * TARGET_SR)

        # Guard against out-of-bounds
        end_sample = min(end_sample, len(audio))
        if start_sample >= end_sample:
            continue

        chunk = audio[start_sample:end_sample]

        # Validate duration
        duration = len(chunk) / TARGET_SR
        if duration < MIN_SEGMENT_DURATION:
            continue

        # Save WAV
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
    Read the set of already-completed file_ids from an existing metadata.csv.
    Extracts file_id (e.g. 4001) from the file_name column (e.g. audio/4001_000.wav).
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
    Append metadata entries to metadata.csv.
    Writes the header only when the file does not yet exist.
    """
    csv_path = OUTPUT_DIR / "metadata.csv"
    mode = "a" if csv_path.exists() and not write_header else "w"
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(entries)


def process_all(test_mode: bool = False):
    """Process all (or just the first) aligned files, with resume support."""

    # Ensure output directories exist
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get already-completed file_ids
    completed_ids = get_completed_file_ids()
    if completed_ids:
        print(f"{len(completed_ids)} files already completed, will be skipped automatically")

    # Collect all files to process
    tasks = []
    for group in GROUPS:
        group_dir = OCSC_DIR / group
        if not group_dir.exists():
            continue
        for aligned_path in sorted(group_dir.glob("*_aligned.json")):
            file_id = aligned_path.stem.replace("_aligned", "")
            # Skip already-processed files
            if file_id in completed_ids:
                continue
            audio_path = DATASET_DIR / group / f"{file_id}.mp3"
            if not audio_path.exists():
                print(f"  ⚠ Skipping {file_id}: no corresponding audio found")
                continue
            tasks.append((audio_path, aligned_path, file_id))

    if not tasks:
        print("All files already processed. Nothing to do.")
        return

    if test_mode:
        tasks = tasks[:1]
        print(f"[Test mode] Processing only: {tasks[0][2]}")

    print(f"Files to process: {len(tasks)}")

    # Write header first if metadata.csv does not exist yet
    csv_path = OUTPUT_DIR / "metadata.csv"
    need_header = not csv_path.exists()
    if need_header:
        append_metadata([], write_header=True)

    total_slices = 0
    for audio_path, aligned_path, file_id in tqdm(tasks, desc="Building dataset"):
        try:
            entries = process_single_file(audio_path, aligned_path, file_id)
            # Append to metadata.csv after each file (key to resume support)
            if entries:
                append_metadata(entries)
                total_slices += len(entries)
        except Exception as e:
            print(f"\n  ✗ Processing failed for {file_id}: {e}")
            continue

    print(f"\nBuild complete!")
    print(f"  New slices added this run: {total_slices}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  metadata: {csv_path}")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    process_all(test_mode=test_mode)
