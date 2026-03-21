#!/usr/bin/env python3
"""Dataset test runner: process images and generate test reports."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiktalk_vlm import generate_ground_truth_with_trace


SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _collect_images(dataset_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in SUPPORTED_EXTENSIONS:
        images.extend(dataset_dir.glob(pattern))
    return sorted(images)


def _write_reports(
    *,
    dataset_dir: Path,
    results: list[dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
) -> tuple[Path, Path]:
    reports_dir = Path("test_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = dataset_dir.name.replace(" ", "_")
    markdown_path = reports_dir / f"{dataset_slug}_report.md"
    json_path = reports_dir / f"{dataset_slug}_results.json"

    total = len(results)
    success = sum(1 for row in results if row["status"] == "success")
    failed = total - success
    elapsed = (finished_at - started_at).total_seconds()

    json_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_dir),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": elapsed,
                "total_images": total,
                "success_count": success,
                "failed_count": failed,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Dataset Test Report")
    lines.append("")
    lines.append("## Process")
    lines.append(f"- Dataset: {dataset_dir}")
    lines.append(f"- Start (UTC): {started_at.isoformat()}")
    lines.append(f"- End (UTC): {finished_at.isoformat()}")
    lines.append(f"- Duration: {elapsed:.2f}s")
    lines.append("- Pipeline call: tiktalk_vlm.generate_ground_truth(image_path)")
    lines.append(f"- File types: {', '.join(SUPPORTED_EXTENSIONS)}")
    lines.append("")
    lines.append("## Result Summary")
    lines.append(f"- Total images: {total}")
    lines.append(f"- Succeeded: {success}")
    lines.append(f"- Failed: {failed}")
    lines.append("")
    lines.append("## Per-Image Results")

    for row in results:
        lines.append("")
        lines.append(f"### {row['image']}")
        lines.append(f"- Status: {row['status']}")
        lines.append(f"- Duration: {row['duration_seconds']:.2f}s")
        if row["status"] == "success":
            lines.append(f"- Chosen provider: {row['chosen_provider']}")
            lines.append("- Final output JSON:")
            lines.append("```json")
            lines.append(json.dumps(row["result"], indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("- Qwen candidate JSON:")
            lines.append("```json")
            lines.append(json.dumps(row["candidates"].get("qwen"), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("- OpenAI candidate JSON:")
            lines.append("```json")
            lines.append(json.dumps(row["candidates"].get("openai"), indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append(f"- Error: {row['error_type']}: {row['error_message']}")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VLM pipeline over a dataset and build reports.")
    parser.add_argument(
        "--dataset",
        default="test_images",
        help="Dataset folder containing input images. Default: test_images",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset folder not found: {dataset_dir}")
        return 1

    image_files = _collect_images(dataset_dir)
    if not image_files:
        print(f"ERROR: No supported images found in {dataset_dir}")
        return 1

    started_at = datetime.now(timezone.utc)
    print(f"Found {len(image_files)} images in {dataset_dir}\n")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 80)
        t0 = time.perf_counter()
        try:
            trace = generate_ground_truth_with_trace(str(image_path))
            result = trace["final_result"]
            chosen_provider = trace["chosen_provider"]
            candidates = trace["candidates"]
            elapsed = time.perf_counter() - t0
            print(f"Chosen provider: {chosen_provider}")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            results.append(
                {
                    "image": image_path.name,
                    "status": "success",
                    "duration_seconds": elapsed,
                    "chosen_provider": chosen_provider,
                    "candidates": candidates,
                    "result": result,
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(f"ERROR: {type(exc).__name__}: {exc}")
            results.append(
                {
                    "image": image_path.name,
                    "status": "failed",
                    "duration_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    finished_at = datetime.now(timezone.utc)
    report_md, report_json = _write_reports(
        dataset_dir=dataset_dir,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )

    total = len(results)
    success = sum(1 for row in results if row["status"] == "success")
    failed = total - success
    print("\n" + "=" * 80)
    print(f"Summary: {success} succeeded, {failed} failed out of {total} images")
    print(f"Markdown report: {report_md}")
    print(f"JSON report: {report_json}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
