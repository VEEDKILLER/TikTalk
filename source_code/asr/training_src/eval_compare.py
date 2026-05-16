"""
评估脚本 —— 多模型 Baseline vs LoRA 对比（含 RTFx 速度指标）

用法：
  # 单模型对比（baseline vs lora）
  python eval_compare.py --config configs/whisper_large_v3.yaml

  # 所有模型汇总，输出 results/ablation_summary.csv
  python eval_compare.py --all
"""

import argparse
import csv
import glob
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from config import load_config
from data import load_and_split_dataset
from metrics import InferTimer, compute_cer, compute_wer, compute_rtfx
from models import get_adapter


# ── 日志输出到文件（同时保留终端输出）────────────────────────────────────────

class _Tee:
    """将 stdout 同时写入终端和文件。"""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    # tqdm 等工具会检查 isatty
    def isatty(self):
        return self._stdout.isatty()


@contextmanager
def log_to_file(log_path: str):
    """将本 context 内的所有 print 输出同时写入 log_path。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        tee = _Tee(f)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout
    print(f"📄 日志已保存至: {log_path}")


# ── 推理辅助 ─────────────────────────────────────────────────────────────────

def transcribe_dataset(model, processor, adapter, dataset, cfg, batch_size=4):
    """
    对测试集进行批量推理，同时计算 RTFx。

    Returns:
        predictions (List[str]), timer (InferTimer)
    """
    model.eval()
    predictions = []
    audio_dir = Path(cfg.data.audio_dir)
    timer = InferTimer(device=cfg.device)

    for i in tqdm(range(0, len(dataset), batch_size), desc="推理中"):
        batch = dataset[i: i + batch_size]
        file_names = batch["file_name"] if isinstance(batch["file_name"], list) else [batch["file_name"]]

        audio_arrays = []
        batch_audio_seconds = 0.0
        for fname in file_names:
            audio_path = audio_dir / Path(fname).name
            audio_array, sr = sf.read(str(audio_path), dtype="float32")
            audio_arrays.append(audio_array)
            batch_audio_seconds += len(audio_array) / sr

        timer.add_audio_seconds(batch_audio_seconds)

        with timer.time_batch():
            decoded = adapter.transcribe_batch(
                model, processor, audio_arrays, cfg.data.sampling_rate, cfg.device
            )

        predictions.extend(decoded)

    return predictions, timer


# ── 单模型评估 ────────────────────────────────────────────────────────────────

def evaluate_one_config(config_path: str, batch_size: int = 4) -> list[dict]:
    """
    对一个配置文件运行 baseline + LoRA 评估，返回结果行列表。
    每行为 dict: {model, variant, WER, CER, RTFx, device}
    结果同时写入 eval_output_<config_stem>.log。
    """
    config_stem = Path(config_path).stem          # e.g. "qwen3_asr_1_7b"
    log_path = f"eval_output_{config_stem}.log"

    with log_to_file(log_path):
        try:
            return _evaluate_one_config_inner(config_path, batch_size)
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            raise


def _evaluate_one_config_inner(config_path: str, batch_size: int) -> list[dict]:
    cfg = load_config(config_path)
    device = cfg.device
    print(f"\n🔧 评估设备: {device}  |  模型: {cfg.model.model_name}")

    adapter = get_adapter(cfg)
    processor = adapter.load_processor(cfg)

    splits = load_and_split_dataset(cfg)
    test_dataset = splits["test"]
    print(f"   测试集: {len(test_dataset)} 条")

    references = test_dataset["transcription"]
    rows = []

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print(f"📊 Baseline — {cfg.model.model_name}")
    print("═══════════════════════════════════════════")

    baseline_model = adapter.load_baseline_model(cfg)
    if device != "cuda":
        baseline_model = baseline_model.to(device)

    baseline_preds, baseline_timer = transcribe_dataset(
        baseline_model, processor, adapter, test_dataset, cfg, batch_size
    )
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    baseline_wer = compute_wer(references, baseline_preds)
    baseline_cer = compute_cer(references, baseline_preds)
    baseline_rtfx = baseline_timer.rtfx()

    rows.append({
        "model": cfg.model.model_name,
        "variant": "baseline",
        "WER": round(baseline_wer, 4),
        "CER": round(baseline_cer, 4),
        "RTFx": round(baseline_rtfx, 2),
        "device": device,
    })

    # ── LoRA 微调 ──────────────────────────────────────────────────────────
    adapter_path = f"{cfg.training.output_dir}/best_adapter"
    if not Path(adapter_path).exists():
        print(f"\n⚠️  未找到 LoRA adapter: {adapter_path}，跳过微调模型评估。")
        rows.append({
            "model": cfg.model.model_name,
            "variant": "lora",
            "WER": "N/A",
            "CER": "N/A",
            "RTFx": "N/A",
            "device": device,
        })
    else:
        print("\n═══════════════════════════════════════════")
        print(f"📊 LoRA 微调 — {cfg.model.model_name}")
        print("═══════════════════════════════════════════")

        finetuned_model = adapter.load_finetuned_model(cfg, adapter_path)
        if device != "cuda":
            finetuned_model = finetuned_model.to(device)

        finetuned_preds, finetuned_timer = transcribe_dataset(
            finetuned_model, processor, adapter, test_dataset, cfg, batch_size
        )
        del finetuned_model
        if device == "cuda":
            torch.cuda.empty_cache()

        finetuned_wer = compute_wer(references, finetuned_preds)
        finetuned_cer = compute_cer(references, finetuned_preds)
        finetuned_rtfx = finetuned_timer.rtfx()

        rows.append({
            "model": cfg.model.model_name,
            "variant": "lora",
            "WER": round(finetuned_wer, 4),
            "CER": round(finetuned_cer, 4),
            "RTFx": round(finetuned_rtfx, 2),
            "device": device,
        })

    # ── 打印对比表 ─────────────────────────────────────────────────────────
    _print_comparison_table(cfg.model.model_name, rows, references,
                            baseline_preds,
                            finetuned_preds if len(rows) == 2 and rows[1]["WER"] != "N/A" else None)

    return rows


def _print_comparison_table(model_name, rows, references, baseline_preds, finetuned_preds):
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  {model_name[:48]:<48}  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  {'模型变体':<16} {'WER':>8} {'CER':>8} {'RTFx':>8}       ║")
    print("╠══════════════════════════════════════════════════════╣")
    for row in rows:
        wer_s = f"{row['WER']:>8.4f}" if isinstance(row['WER'], float) else f"{'N/A':>8}"
        cer_s = f"{row['CER']:>8.4f}" if isinstance(row['CER'], float) else f"{'N/A':>8}"
        rtfx_s = f"{row['RTFx']:>8.2f}" if isinstance(row['RTFx'], float) else f"{'N/A':>8}"
        label = "Baseline" if row["variant"] == "baseline" else "LoRA 微调"
        print(f"║  {label:<16}{wer_s}{cer_s}{rtfx_s}       ║")
    print("╠══════════════════════════════════════════════════════╣")

    if len(rows) == 2 and isinstance(rows[0]["WER"], float) and isinstance(rows[1]["WER"], float):
        wer_imp = (rows[0]["WER"] - rows[1]["WER"]) / rows[0]["WER"] * 100 if rows[0]["WER"] > 0 else 0
        cer_imp = (rows[0]["CER"] - rows[1]["CER"]) / rows[0]["CER"] * 100 if rows[0]["CER"] > 0 else 0
        print(f"║  {'WER 提升':<16}{wer_imp:>+7.2f}%                           ║")
        print(f"║  {'CER 提升':<16}{cer_imp:>+7.2f}%                           ║")
        print("╠══════════════════════════════════════════════════════╣")

    print("╚══════════════════════════════════════════════════════╝")

    if finetuned_preds:
        print("\n📝 部分样本对比（前 5 条）：")
        print("─" * 80)
        for i in range(min(5, len(references))):
            print(f"\n【样本 {i+1}】")
            print(f"  真实文本: {references[i][:100]}")
            print(f"  Baseline: {baseline_preds[i][:100]}")
            print(f"  LoRA 微调: {finetuned_preds[i][:100]}")


# ── 全模型汇总 ────────────────────────────────────────────────────────────────

def evaluate_all(configs_dir: str = "configs", batch_size: int = 4, models: list[str] | None = None):
    """遍历 configs/*.yaml，汇总所有模型评估结果，写入 CSV。

    Args:
        models: 若指定，只评估文件名包含这些关键字的配置（如 ["whisper", "qwen3"]）。
                不指定则评估全部。
    """
    yaml_files = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    if not yaml_files:
        print(f"⚠️  在 {configs_dir}/ 中未找到任何 YAML 文件。")
        return

    if models:
        filtered = [f for f in yaml_files if any(m.lower() in os.path.basename(f).lower() for m in models)]
        if not filtered:
            print(f"⚠️  --models {models} 未匹配任何配置文件（候选: {[os.path.basename(f) for f in yaml_files]}）")
            return
        yaml_files = filtered
        print(f"🔍 已筛选配置: {[os.path.basename(f) for f in yaml_files]}")

    all_rows = []
    for yaml_path in yaml_files:
        print(f"\n{'='*60}")
        print(f"处理配置: {yaml_path}")
        print('='*60)
        try:
            rows = evaluate_one_config(yaml_path, batch_size=batch_size)
            all_rows.extend(rows)
        except Exception:
            pass  # 错误已在 evaluate_one_config 内写入对应 log 文件

    # 输出汇总 CSV
    output_path = "results/ablation_summary.csv"
    Path("results").mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "variant", "WER", "CER", "RTFx", "device"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ 汇总结果已写入: {output_path}")

    # 打印汇总表
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                      消融实验汇总结果                             ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  {'模型':<35} {'变体':<10} {'WER':>7} {'CER':>7} {'RTFx':>7}  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    for row in all_rows:
        name = row["model"].split("/")[-1][:34]
        wer_s = f"{row['WER']:>7.4f}" if isinstance(row['WER'], float) else f"{'N/A':>7}"
        cer_s = f"{row['CER']:>7.4f}" if isinstance(row['CER'], float) else f"{'N/A':>7}"
        rtfx_s = f"{row['RTFx']:>7.2f}" if isinstance(row['RTFx'], float) else f"{'N/A':>7}"
        print(f"║  {name:<35} {row['variant']:<10}{wer_s}{cer_s}{rtfx_s}  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="多模型 ASR 评估对比")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        help="单个 YAML 配置文件路径，例如 configs/whisper_large_v3.yaml",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="遍历 configs/*.yaml 跑全部模型，输出 results/ablation_summary.csv",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="KEYWORD",
        help="配合 --all 使用，只评估文件名含指定关键字的模型，可多选。"
             "例: --models whisper qwen3",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="推理 batch size（默认 4）",
    )
    parser.add_argument(
        "--configs_dir", default="configs",
        help="--all 模式下 YAML 文件所在目录（默认 configs/）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all:
        evaluate_all(configs_dir=args.configs_dir, batch_size=args.batch_size, models=args.models)
    else:
        evaluate_one_config(args.config, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
