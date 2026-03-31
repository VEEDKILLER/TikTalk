"""
Step 1: CHA 文件解析与文本清洗
解析 TalkBank CHAT 格式的 .cha 文件，提取并清洗转录文本。
输出: dataset_orig/OCSC/{group}/{id}_cleaned.txt
"""

import re
import os
from pathlib import Path
from tqdm import tqdm


# ── 项目路径 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OCSC_DIR = BASE_DIR / "dataset_orig" / "OCSC"
GROUPS = ["4", "5", "6", "7", "8", "9"]


def clean_utterance(text: str) -> str:
    """
    对单条 CHAT 转录文本进行清洗，去除所有特殊符号。

    处理顺序很重要：先处理复杂模式，再处理简单模式。
    """
    # 0. 去除行首的 speaker 标记 (*CHI:\t, *EXP:\t 等)
    #    在 parse_cha 中已经处理，这里做防御性检查
    text = re.sub(r"^\*\w+:\t?", "", text)

    # 1. 处理 [: replacement] 模式 (最佳猜测替换)
    #    si(r)dy [: thirty] → thirty
    #    先找到 word [: replacement]，把 word 替换成 replacement
    text = re.sub(r"\S+\s+\[:\s+([^\]]+)\]", r"\1", text)

    # 2. 删除各种方括号标注
    text = re.sub(r"\[\^[^\]]*\]", "", text)       # [^ text] 行为注释
    text = re.sub(r"\[=![^\]]*\]", "", text)       # [=! text] 解释
    text = re.sub(r"\[=[^\]]*\]", "", text)        # [= text] 解释
    text = re.sub(r"\[/+\]", "", text)             # [/] [//] [///] 重复/修正
    text = re.sub(r"\[\?\]", "", text)             # [?] 最佳猜测
    text = re.sub(r"\[>[^\]]*\]", "", text)        # [>] [>1] 重叠
    text = re.sub(r"\[<[^\]]*\]", "", text)        # [<] [<1] 重叠
    text = re.sub(r"\[%[^\]]*\]", "", text)        # [% text] 注释
    text = re.sub(r"\[\+[^\]]*\]", "", text)       # [+ text] 后置标记

    # 3. 删除 <word> 尖括号 (重叠/重复范围标记)，保留内容
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # 4. 删除 &= 开头的非语言标记 (&=laughs, &=gasps, &=nods, &=pnt)
    text = re.sub(r"&=[a-zA-Z]+", "", text)

    # 5. 删除 &+ 语音碎片 (&+gg, &+col, &+ff)
    text = re.sub(r"&\+[a-zA-Z]+", "", text)

    # 6. 删除 &- 填充停顿 (&-uh, &-eh, &-er)
    text = re.sub(r"&-[a-zA-Z]+", "", text)

    # 7. 删除 &~ 伴随语音 (&~wow, &~t)
    text = re.sub(r"&~[a-zA-Z]+", "", text)

    # 8. 删除 xxx (不可辨识的语音) 和 yyy, www
    text = re.sub(r"\bxxx\b", "", text)
    text = re.sub(r"\byyy\b", "", text)
    text = re.sub(r"\bwww\b", "", text)

    # 9. 处理 word@l, word@u, word@i, word@k, word@b 等后缀 → 保留单词
    text = re.sub(r"(\w+)@[a-z]", r"\1", text)

    # 10. 处理 (th)em → them, (un)til → until 省略音节
    text = re.sub(r"\(([^)]+)\)", r"\1", text)

    # 11. 处理 :: 延长发音 cray::on → crayon
    text = re.sub(r"::", "", text)

    # 12. 删除话语中断/延续标记
    text = re.sub(r"\+/\.\?", "", text)     # +/?.
    text = re.sub(r"\+/\?", "", text)       # +/?
    text = re.sub(r"\+/\.", "", text)       # +/.
    text = re.sub(r"\+\.\.\.", "", text)    # +...
    text = re.sub(r"\+\.\.\?", "", text)    # +..?
    text = re.sub(r'\+"\.*', "", text)      # +"...
    text = re.sub(r"\+<", "", text)         # +<
    text = re.sub(r"\+\^", "", text)        # +^
    text = re.sub(r"\+,", "", text)         # +,

    # 13. 删除 0word (省略词)
    text = re.sub(r"\b0\w+\b", "", text)

    # 14. 清理前后标点多余空格，标准化
    text = re.sub(r"\s+", " ", text)        # 多余空格
    text = text.strip()

    # 15. 去掉只剩标点的情况
    if re.match(r'^[.\s?!,;:]*$', text):
        return ""

    return text


def parse_cha(cha_path: Path) -> list[tuple[str, str]]:
    """
    解析 .cha 文件，返回 [(speaker, cleaned_text), ...] 列表。

    只提取 *SPEAKER: 行的转录文本，跳过所有标注行。
    多行续行（以 tab 开头）会合并到前一行。
    """
    with open(cha_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    utterances: list[tuple[str, str]] = []
    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.rstrip("\n")

        # Speaker 行: *CHI:\ttext text text .
        match = re.match(r"^\*(\w+):\t(.*)$", line)
        if match:
            # 先保存上一条
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    utterances.append((current_speaker, cleaned))
            current_speaker = match.group(1)
            current_text = match.group(2)
            continue

        # 续行 (以 tab 开头，属于同一个 speaker 的 utterance)
        if line.startswith("\t") and current_speaker:
            # 只有不是标注行的续行才追加
            # 标注行通常是 %mor:\t, %gra:\t 等，但续行直接以 \t 开头
            # 这里需要判断：如果上一行是 speaker 行的续行
            current_text += " " + line.strip()
            continue

        # 标注行 (%mor, %gra, %com 等) → 跳过
        if re.match(r"^%\w+:", line):
            # 先保存上一条 speaker utterance
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    utterances.append((current_speaker, cleaned))
                current_speaker = None
                current_text = ""
            continue

        # 元数据行 (@Begin, @End, @Languages, @G: 等) → 跳过
        if line.startswith("@"):
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    utterances.append((current_speaker, cleaned))
                current_speaker = None
                current_text = ""
            continue

    # 处理最后一条
    if current_speaker and current_text:
        cleaned = clean_utterance(current_text)
        if cleaned:
            utterances.append((current_speaker, cleaned))

    return utterances


def process_all():
    """处理所有 CHA 文件并输出清洗后的文本。"""
    cha_files = []
    for group in GROUPS:
        group_dir = OCSC_DIR / group
        if group_dir.exists():
            cha_files.extend(sorted(group_dir.glob("*.cha")))

    print(f"找到 {len(cha_files)} 个 CHA 文件")

    total_utterances = 0
    for cha_path in tqdm(cha_files, desc="解析 CHA 文件"):
        utterances = parse_cha(cha_path)
        total_utterances += len(utterances)

        # 输出到同目录下 {id}_cleaned.txt
        out_path = cha_path.with_name(cha_path.stem + "_cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for speaker, text in utterances:
                f.write(f"{speaker}\t{text}\n")

    print(f"处理完成! 共提取 {total_utterances} 条有效转录")


if __name__ == "__main__":
    process_all()
