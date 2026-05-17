#!/usr/bin/env python3
"""Standalone evaluation for ensemble vote vs single VLM baselines."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tiktalk_vlm.pipeline import (
    GroundTruthPipeline,
    PipelineConfig,
    _guess_mime_type,
    _to_data_url,
)


SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")
STRATEGIES = ("ensemble_vote", "openai_only", "qwen_only")


def _collect_images(dataset_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in SUPPORTED_EXTENSIONS:
        images.extend(dataset_dir.glob(pattern))
    return sorted(images)


def _load_gold_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if all(isinstance(value, dict) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}
        if "labels" in payload and isinstance(payload["labels"], list):
            return _labels_list_to_mapping(payload["labels"])
    if isinstance(payload, list):
        return _labels_list_to_mapping(payload)

    raise ValueError("Gold label file must be a filename->JSON mapping or a list with image fields.")


def _labels_list_to_mapping(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "image" not in row:
            raise ValueError("Each gold label row must be an object with an 'image' field.")
        image_name = str(row["image"])
        label = row.get("label")
        mapping[image_name] = label if isinstance(label, dict) else row
    return mapping


def _run_ensemble(pipeline: GroundTruthPipeline, image_path: Path) -> dict[str, Any]:
    started_at = time.perf_counter()
    trace = pipeline.generate_ground_truth_with_trace(image_path)
    return {
        "strategy": "ensemble_vote",
        "status": "success",
        "duration_seconds": time.perf_counter() - started_at,
        "output": trace["final_result"],
        "chosen_provider": trace.get("chosen_provider"),
        "candidates": trace.get("candidates", {}),
    }


def _run_single_provider(
    pipeline: GroundTruthPipeline,
    image_path: Path,
    *,
    provider: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_bytes = image_path.read_bytes()
    mime_type = _guess_mime_type(image_path)
    image_data_url = _to_data_url(image_bytes, mime_type)

    if provider == "openai":
        output = pipeline._generate_with_openai(image_data_url)
        strategy = "openai_only"
    elif provider == "qwen":
        output = pipeline._generate_with_qwen(image_data_url)
        strategy = "qwen_only"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported provider: {provider}")

    return {
        "strategy": strategy,
        "status": "success",
        "duration_seconds": time.perf_counter() - started_at,
        "output": output,
    }


def _failure_result(strategy: str, exc: Exception) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "status": "failed",
        "duration_seconds": 0.0,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "output": None,
    }


def _evaluate_one_image(
    pipeline: GroundTruthPipeline,
    image_path: Path,
    *,
    repeat_index: int,
    gold: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_row: dict[str, Any] = {
        "image": image_path.name,
        "repeat": repeat_index,
        "results": {},
    }
    strategy_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_ensemble, pipeline, image_path): "ensemble_vote",
            executor.submit(_run_single_provider, pipeline, image_path, provider="openai"): "openai_only",
            executor.submit(_run_single_provider, pipeline, image_path, provider="qwen"): "qwen_only",
        }
        for future, strategy in futures.items():
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = _failure_result(strategy, exc)

            if result["status"] == "success" and gold is not None and result["output"] is not None:
                result["scores"] = _score_prediction(result["output"], gold)

            image_row["results"][strategy] = result
            strategy_rows.append(
                {
                    "image": image_path.name,
                    "repeat": repeat_index,
                    "strategy": strategy,
                    "status": result["status"],
                    "duration_seconds": float(result["duration_seconds"]),
                    **({"scores": result["scores"]} if "scores" in result else {}),
                    **(
                        {
                            "error_type": result["error_type"],
                            "error_message": result["error_message"],
                        }
                        if result["status"] != "success"
                        else {}
                    ),
                }
            )

    return image_row, strategy_rows


def _score_prediction(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    object_precision, object_recall, object_f1 = _multiset_f1(
        _object_counter(prediction),
        _object_counter(gold),
    )
    action_precision, action_recall, action_f1 = _multiset_f1(
        _string_counter(prediction.get("actions", [])),
        _string_counter(gold.get("actions", [])),
    )
    text_precision, text_recall, text_f1 = _multiset_f1(
        _string_counter(prediction.get("visible_text", [])),
        _string_counter(gold.get("visible_text", [])),
    )
    focus_precision, focus_recall, focus_f1 = _multiset_f1(
        _string_counter(prediction.get("teaching_focus_words", [])),
        _string_counter(gold.get("teaching_focus_words", [])),
    )
    setting_precision, setting_recall, setting_f1 = _token_f1(
        str(prediction.get("setting", "")),
        str(gold.get("setting", "")),
    )
    summary_precision, summary_recall, summary_f1 = _token_f1(
        str(prediction.get("scene_summary", "")),
        str(gold.get("scene_summary", "")),
    )

    grounding_precision = _safe_mean([object_precision, action_precision, text_precision])
    grounding_recall = _safe_mean([object_recall, action_recall, text_recall])
    hallucination_rate = 1.0 - grounding_precision
    quality_score = _weighted_average(
        [
            (object_f1, 0.35),
            (action_f1, 0.20),
            (text_f1, 0.10),
            (focus_f1, 0.10),
            (setting_f1, 0.15),
            (summary_f1, 0.10),
        ]
    )

    return {
        "quality_score": round(_clamp01(quality_score), 4),
        "grounding_precision": round(_clamp01(grounding_precision), 4),
        "grounding_recall": round(_clamp01(grounding_recall), 4),
        "hallucination_rate": round(_clamp01(hallucination_rate), 4),
        "object_f1": round(_clamp01(object_f1), 4),
        "action_f1": round(_clamp01(action_f1), 4),
        "visible_text_f1": round(_clamp01(text_f1), 4),
        "focus_f1": round(_clamp01(focus_f1), 4),
        "setting_f1": round(_clamp01(setting_f1), 4),
        "summary_f1": round(_clamp01(summary_f1), 4),
    }


def _summarize_strategy_rows(
    strategy_rows: list[dict[str, Any]],
    *,
    practical_delta: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        rows = [row for row in strategy_rows if row["strategy"] == strategy]
        success_rows = [row for row in rows if row["status"] == "success"]
        quality_rows = [row for row in success_rows if "scores" in row]

        strategy_summary: dict[str, Any] = {
            "total_runs": len(rows),
            "success_count": len(success_rows),
            "failure_count": len(rows) - len(success_rows),
            "success_rate": round(_ratio(len(success_rows), len(rows)), 4),
            "avg_duration_seconds": round(_average(rows, "duration_seconds"), 4),
        }
        if quality_rows:
            strategy_summary["quality_metrics"] = {
                key: round(_average_nested(quality_rows, key), 4)
                for key in quality_rows[0]["scores"].keys()
            }
        summary[strategy] = strategy_summary

    paired_rows = _build_paired_quality_rows(strategy_rows)
    if paired_rows:
        wins = sum(1 for row in paired_rows if row["delta"] > practical_delta)
        losses = sum(1 for row in paired_rows if row["delta"] < -practical_delta)
        ties = len(paired_rows) - wins - losses
        summary["paired_decision"] = {
            "comparison_unit_count": len(paired_rows),
            "practical_delta": practical_delta,
            "ensemble_wins": wins,
            "ensemble_losses": losses,
            "ensemble_ties": ties,
            "avg_quality_delta_vs_best_single": round(mean(row["delta"] for row in paired_rows), 4),
        }

    return summary


def _build_paired_quality_rows(strategy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in strategy_rows:
        if row["status"] != "success" or "scores" not in row:
            continue
        key = (row["image"], int(row["repeat"]))
        grouped.setdefault(key, {})[row["strategy"]] = float(row["scores"]["quality_score"])

    paired: list[dict[str, Any]] = []
    for (image, repeat), scores in grouped.items():
        if not all(strategy in scores for strategy in STRATEGIES):
            continue
        best_single = max(scores["openai_only"], scores["qwen_only"])
        paired.append(
            {
                "image": image,
                "repeat": repeat,
                "ensemble_quality_score": scores["ensemble_vote"],
                "best_single_quality_score": best_single,
                "delta": scores["ensemble_vote"] - best_single,
            }
        )
    return paired


def _build_blind_review_packet(image_rows: list[dict[str, Any]], seed: int) -> tuple[str, dict[str, Any]]:
    rng = random.Random(seed)
    lines = [
        "# Blind Review Packet",
        "",
        "For each image, rate every option on:",
        "- Image accuracy: 1-5",
        "- Junior-learner suitability: 1-5",
        "- Final winner: choose one option",
        "",
    ]
    key: dict[str, Any] = {}
    review_index = 1

    for row in image_rows:
        outputs = []
        for strategy in STRATEGIES:
            result = row["results"].get(strategy)
            if result and result["status"] == "success" and result["output"] is not None:
                outputs.append((strategy, result["output"]))
        if len(outputs) < 2:
            continue

        rng.shuffle(outputs)
        review_id = f"review_{review_index:03d}"
        key[review_id] = {
            "image": row["image"],
            "repeat": row["repeat"],
            "options": {chr(65 + idx): strategy for idx, (strategy, _) in enumerate(outputs)},
        }

        lines.append(f"## {review_id}: {row['image']} (repeat {row['repeat']})")
        lines.append("")
        for idx, (_, output) in enumerate(outputs):
            option = chr(65 + idx)
            lines.append(f"### Option {option}")
            lines.append("```json")
            lines.append(json.dumps(output, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        review_index += 1

    return "\n".join(lines).rstrip() + "\n", key


def _write_reports(
    *,
    dataset_dir: Path,
    image_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    blind_review_markdown: str,
    blind_review_key: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    reports_dir = Path("evaluation_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = dataset_dir.name.replace(" ", "_")
    raw_path = reports_dir / f"{dataset_slug}_evaluation.json"
    summary_path = reports_dir / f"{dataset_slug}_evaluation.md"
    blind_path = reports_dir / f"{dataset_slug}_blind_review.md"
    blind_key_path = reports_dir / f"{dataset_slug}_blind_review_key.json"

    raw_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_dir),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "image_rows": image_rows,
                "strategy_rows": strategy_rows,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Vote Mechanism Evaluation",
        "",
        f"- Dataset: {dataset_dir}",
        f"- Total image runs: {len(image_rows)}",
        "",
        "## Strategy Summary",
        "",
    ]
    for strategy in STRATEGIES:
        metrics = summary[strategy]
        lines.append(f"### {strategy}")
        lines.append(f"- Success rate: {metrics['success_rate']:.2%}")
        lines.append(f"- Avg duration: {metrics['avg_duration_seconds']:.2f}s")
        lines.append(f"- Successes: {metrics['success_count']}")
        lines.append(f"- Failures: {metrics['failure_count']}")
        if "quality_metrics" in metrics:
            quality = metrics["quality_metrics"]
            lines.append(f"- Quality score: {quality.get('quality_score', 0.0):.4f}")
            lines.append(f"- Grounding precision: {quality.get('grounding_precision', 0.0):.4f}")
            lines.append(f"- Hallucination rate: {quality.get('hallucination_rate', 0.0):.4f}")
        lines.append("")

    paired = summary.get("paired_decision")
    if paired is not None:
        lines.extend(
            [
                "## Ensemble Vs Best Single",
                "",
                f"- Comparison units: {paired['comparison_unit_count']}",
                f"- Practical delta threshold: {paired['practical_delta']:.3f}",
                f"- Ensemble wins: {paired['ensemble_wins']}",
                f"- Ensemble losses: {paired['ensemble_losses']}",
                f"- Ensemble ties: {paired['ensemble_ties']}",
                f"- Avg quality delta: {paired['avg_quality_delta_vs_best_single']:.4f}",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    blind_path.write_text(blind_review_markdown, encoding="utf-8")
    blind_key_path.write_text(json.dumps(blind_review_key, indent=2, ensure_ascii=False), encoding="utf-8")
    return raw_path, summary_path, blind_path, blind_key_path


def _object_counter(payload: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in payload.get("main_objects", []):
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("name", ""))
        count = item.get("count", 1)
        if not name:
            continue
        if not isinstance(count, int) or count < 1:
            count = 1
        counter[name] += count
    return counter


def _string_counter(values: list[Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in values:
        normalized = _normalize_text(item)
        if normalized:
            counter[normalized] += 1
    return counter


def _multiset_f1(predicted: Counter[str], gold: Counter[str]) -> tuple[float, float, float]:
    matched = sum((predicted & gold).values())
    predicted_total = sum(predicted.values())
    gold_total = sum(gold.values())
    precision = _ratio(matched, predicted_total, empty_default=1.0 if gold_total == 0 else 0.0)
    recall = _ratio(matched, gold_total, empty_default=1.0 if predicted_total == 0 else 0.0)
    return precision, recall, _f1(precision, recall)


def _token_f1(predicted: str, gold: str) -> tuple[float, float, float]:
    return _multiset_f1(Counter(_tokenize(predicted)), Counter(_tokenize(gold)))


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ratio(numerator: int, denominator: int, *, empty_default: float = 0.0) -> float:
    if denominator == 0:
        return empty_default
    return numerator / denominator


def _safe_mean(values: list[float]) -> float:
    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return 0.0
    return mean(filtered)


def _weighted_average(values: list[tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if math.isnan(value) or math.isnan(weight) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return 0.0
    return mean(values)


def _average_nested(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["scores"][key]) for row in rows if "scores" in row and key in row["scores"]]
    if not values:
        return 0.0
    return mean(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone evaluation for ensemble vote vs single VLM baselines."
    )
    parser.add_argument(
        "--dataset",
        default="test_images",
        help="Dataset folder containing images. Default: test_images",
    )
    parser.add_argument(
        "--gold",
        help="Optional JSON file with gold labels keyed by image filename.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run each image through each strategy. Default: 1",
    )
    parser.add_argument(
        "--practical-delta",
        type=float,
        default=0.03,
        help="Minimum absolute quality delta to count as a meaningful win. Default: 0.03",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for blind review shuffling. Default: 7",
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

    gold_labels = _load_gold_labels(Path(args.gold)) if args.gold else {}
    pipeline = GroundTruthPipeline(config=PipelineConfig.from_env())

    image_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []

    for repeat_index in range(1, args.repeats + 1):
        for image_path in image_files:
            print(f"Processing {image_path.name} (repeat {repeat_index}/{args.repeats})")
            image_row, per_strategy_rows = _evaluate_one_image(
                pipeline,
                image_path,
                repeat_index=repeat_index,
                gold=gold_labels.get(image_path.name),
            )
            image_rows.append(image_row)
            strategy_rows.extend(per_strategy_rows)

    summary = _summarize_strategy_rows(strategy_rows, practical_delta=args.practical_delta)
    blind_review_markdown, blind_review_key = _build_blind_review_packet(image_rows, args.seed)
    raw_path, summary_path, blind_path, blind_key_path = _write_reports(
        dataset_dir=dataset_dir,
        image_rows=image_rows,
        strategy_rows=strategy_rows,
        summary=summary,
        blind_review_markdown=blind_review_markdown,
        blind_review_key=blind_review_key,
    )

    print(f"Evaluation JSON: {raw_path}")
    print(f"Evaluation summary: {summary_path}")
    print(f"Blind review packet: {blind_path}")
    print(f"Blind review key: {blind_key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
