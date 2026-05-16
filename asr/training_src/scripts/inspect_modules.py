"""
辅助脚本：从 safetensors.index.json 读取所有权重路径，
推荐 LoRA target_modules，完全不需要加载模型权重。

用法：
    python scripts/inspect_modules.py <hf_model_id>

示例：
    python scripts/inspect_modules.py Qwen/Qwen3-ASR-1.7B
    python scripts/inspect_modules.py CohereForAI/cohere-transcribe-03-2026
    python scripts/inspect_modules.py openai/whisper-large-v3

只需下载 model.safetensors.index.json（JSON 文件，几 KB），
在 Mac CPU 上秒级完成，无需 GPU。

若模型为 gated model（需要 HF 账号），请先运行：
    uv run huggingface-cli login
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# 注意力/FFN 关键词（用于推荐）
ATTN_KEYWORDS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "out_proj", "query", "key", "value",
    "query_key_value", "dense", "c_attn", "c_proj",
}
FFN_KEYWORDS = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "fc_in", "fc_out"}


def inspect(model_id: str):
    print(f"\n模型：{model_id}")
    print("（只下载 model.safetensors.index.json，不下载权重）\n")

    # 1. 尝试下载 index 文件（模型分片场景）
    index_path = None
    for fname in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        try:
            index_path = hf_hub_download(model_id, fname)
            print(f"   已下载: {fname}\n")
            break
        except Exception:
            pass

    if index_path is None:
        print("⚠️  未找到 *.index.json（可能是单文件模型）。")
        print("   尝试下载单文件模型的 config 来推断结构...\n")
        _fallback_config_inspect(model_id)
        return

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    weight_keys = sorted(idx["weight_map"].keys())
    _analyze_and_print(weight_keys, model_id)


def _fallback_config_inspect(model_id: str):
    """单文件模型：用 config.json 推断结构。"""
    try:
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"config.json 中的 architectures: {cfg.get('architectures', 'N/A')}")
        print(f"model_type: {cfg.get('model_type', 'N/A')}")
        print("\n请参考该模型的 HuggingFace 文档确认 LoRA target_modules。")
    except Exception as e:
        print(f"❌ 无法获取 config.json：{e}")


def _analyze_and_print(weight_keys: list[str], model_id: str):
    # 取每个权重路径最后一级（去掉 .weight/.bias 后缀）的倒数第二级作为"层类型名"
    layer_type_names: set[str] = set()
    for k in weight_keys:
        if not (k.endswith(".weight") or k.endswith(".bias")):
            continue
        parts = k.rsplit(".", 2)  # [..., "layer_name", "weight"]
        if len(parts) >= 2:
            layer_type_names.add(parts[-2])

    all_names = sorted(layer_type_names)
    attn_recommended = sorted(n for n in all_names if n in ATTN_KEYWORDS)
    ffn_recommended = sorted(n for n in all_names if n in FFN_KEYWORDS)

    print("════ 所有倒数第二级层名（去重）════")
    print(f"  {all_names}\n")

    print("════ 推荐 LoRA target_modules（注意力层）════")
    print(f"  {attn_recommended}")
    print("  → 放入 yaml 的 lora.target_modules（保守策略，显存少）\n")

    print("════ 可选：加入 FFN 层（更充分微调，显存更多）════")
    print(f"  {attn_recommended + ffn_recommended}")
    print("  → 如显存充足可选此列表\n")

    # 打印部分完整路径帮助理解命名
    print("════ 部分完整权重路径（前 20 条，帮助确认层级）════")
    linear_keys = [k for k in weight_keys if k.endswith(".weight") and
                   any(kw in k for kw in ATTN_KEYWORDS | FFN_KEYWORDS)]
    for k in linear_keys[:20]:
        print(f"  {k}")
    if len(linear_keys) > 20:
        print(f"  ... 共 {len(linear_keys)} 条\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/inspect_modules.py <hf_model_id>")
        print("示例: python scripts/inspect_modules.py Qwen/Qwen3-ASR-1.7B")
        sys.exit(1)
    inspect(sys.argv[1])
