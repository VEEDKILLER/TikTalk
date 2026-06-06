"""
评估指标模块
提供 WER、CER、RTFx 的统一计算接口。

RTFx (Real-Time Factor inverse) = 音频总时长(秒) / 推理总耗时(秒)
  > 1 表示比实时更快；值越大推理速度越快。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

import jiwer
import torch
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

# EnglishTextNormalizer 把数字词 ↔ 阿拉伯数字归一（"four zero zero one" ↔ "4001"），
# 能去掉这个数据集 label 里数字写成词造成的虚高 WER。
# 传空 mapping 足够；mapping 是 britism→american 的可选补充。
_normalizer = EnglishTextNormalizer({})
_basic_normalizer = BasicTextNormalizer()


def normalize(texts: List[str]) -> List[str]:
    """对文本列表统一做 Whisper BasicTextNormalizer（移除标点、转小写）。
    用于跨模型可比的 WER/CER 计算。"""
    return [_normalizer(t) for t in texts]


def compute_wer(references: List[str], predictions: List[str], normalize: bool = True) -> float:
    refs = [_normalizer(r) for r in references] if normalize else references
    preds = [_normalizer(p) for p in predictions] if normalize else predictions
    return jiwer.wer(refs, preds)


def compute_cer(references: List[str], predictions: List[str], normalize: bool = True) -> float:
    refs = [_normalizer(r) for r in references] if normalize else references
    preds = [_normalizer(p) for p in predictions] if normalize else predictions
    return jiwer.cer(refs, preds)


def compute_rtfx(total_audio_seconds: float, total_infer_seconds: float) -> float:
    """计算 RTFx。total_infer_seconds 为 0 时返回 inf（理论上不会发生）。"""
    if total_infer_seconds <= 0:
        return float("inf")
    return total_audio_seconds / total_infer_seconds


# ── 计时上下文 ─────────────────────────────────────────────────────────────────

@dataclass
class InferTimer:
    """
    累计推理时间和音频时长的计时器。
    在推理循环中用 record_batch() 更新，最后调用 rtfx() 取结果。

    用法:
        timer = InferTimer(device="cuda")
        with timer.time_batch():
            ids = model.generate(...)
        timer.add_audio_seconds(sum(len(a)/sr for a in batch_audios))
    """
    device: str = "cpu"
    total_audio_seconds: float = field(default=0.0, init=False)
    total_infer_seconds: float = field(default=0.0, init=False)

    def add_audio_seconds(self, seconds: float):
        self.total_audio_seconds += seconds

    @contextmanager
    def time_batch(self):
        """包裹 model.generate() 调用，自动计时（CUDA 使用 synchronize 确保准确）。"""
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.total_infer_seconds += time.perf_counter() - t0

    def rtfx(self) -> float:
        return compute_rtfx(self.total_audio_seconds, self.total_infer_seconds)
