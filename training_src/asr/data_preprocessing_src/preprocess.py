"""
Step 1: CHA file parsing and text cleaning
Parse TalkBank CHAT format .cha files, extract and clean transcript text.
Output: dataset_orig/OCSC/{group}/{id}_cleaned.txt
"""

import re
import os
from pathlib import Path
from tqdm import tqdm


# ── Project paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OCSC_DIR = BASE_DIR / "dataset_orig" / "OCSC"
GROUPS = ["4", "5", "6", "7", "8", "9"]


def clean_utterance(text: str) -> str:
    """
    Clean a single CHAT transcript utterance by removing all special symbols.

    Processing order matters: handle complex patterns before simple ones.
    """
    # 0. Remove leading speaker marker (*CHI:\t, *EXP:\t, etc.)
    #    Already handled in parse_cha; this is a defensive check
    text = re.sub(r"^\*\w+:\t?", "", text)

    # 1. Handle [: replacement] pattern (best-guess replacement)
    #    si(r)dy [: thirty] → thirty
    #    Find word [: replacement] and substitute the original word with replacement
    text = re.sub(r"\S+\s+\[:\s+([^\]]+)\]", r"\1", text)

    # 2. Remove various bracket annotations
    text = re.sub(r"\[\^[^\]]*\]", "", text)       # [^ text] action comment
    text = re.sub(r"\[=![^\]]*\]", "", text)       # [=! text] explanation
    text = re.sub(r"\[=[^\]]*\]", "", text)        # [= text] explanation
    text = re.sub(r"\[/+\]", "", text)             # [/] [//] [///] repetition/correction
    text = re.sub(r"\[\?\]", "", text)             # [?] best guess
    text = re.sub(r"\[>[^\]]*\]", "", text)        # [>] [>1] overlap
    text = re.sub(r"\[<[^\]]*\]", "", text)        # [<] [<1] overlap
    text = re.sub(r"\[%[^\]]*\]", "", text)        # [% text] comment
    text = re.sub(r"\[\+[^\]]*\]", "", text)       # [+ text] post-code

    # 3. Remove <word> angle brackets (overlap/repetition scope markers), keep content
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # 4. Remove &= non-verbal markers (&=laughs, &=gasps, &=nods, &=pnt)
    text = re.sub(r"&=[a-zA-Z]+", "", text)

    # 5. Remove &+ phonological fragments (&+gg, &+col, &+ff)
    text = re.sub(r"&\+[a-zA-Z]+", "", text)

    # 6. Remove &- filler pauses (&-uh, &-eh, &-er)
    text = re.sub(r"&-[a-zA-Z]+", "", text)

    # 7. Remove &~ accompanying sounds (&~wow, &~t)
    text = re.sub(r"&~[a-zA-Z]+", "", text)

    # 8. Remove xxx (unintelligible speech) and yyy, www
    text = re.sub(r"\bxxx\b", "", text)
    text = re.sub(r"\byyy\b", "", text)
    text = re.sub(r"\bwww\b", "", text)

    # 9. Handle word@l, word@u, word@i, word@k, word@b suffixes → keep the word
    text = re.sub(r"(\w+)@[a-z]", r"\1", text)

    # 10. Handle (th)em → them, (un)til → until elided syllables
    text = re.sub(r"\(([^)]+)\)", r"\1", text)

    # 11. Handle :: elongated pronunciation cray::on → crayon
    text = re.sub(r"::", "", text)

    # 12. Remove utterance interruption/continuation markers
    text = re.sub(r"\+/\.\?", "", text)     # +/?.
    text = re.sub(r"\+/\?", "", text)       # +/?
    text = re.sub(r"\+/\.", "", text)       # +/.
    text = re.sub(r"\+\.\.\.", "", text)    # +...
    text = re.sub(r"\+\.\.\?", "", text)    # +..?
    text = re.sub(r'\+"\.*', "", text)      # +"...
    text = re.sub(r"\+<", "", text)         # +<
    text = re.sub(r"\+\^", "", text)        # +^
    text = re.sub(r"\+,", "", text)         # +,

    # 13. Remove 0word (omitted words)
    text = re.sub(r"\b0\w+\b", "", text)

    # 14. Clean up extra whitespace and normalize
    text = re.sub(r"\s+", " ", text)        # excess whitespace
    text = text.strip()

    # 15. Discard strings that contain only punctuation
    if re.match(r'^[.\s?!,;:]*$', text):
        return ""

    return text


def parse_cha(cha_path: Path) -> list[tuple[str, str]]:
    """
    Parse a .cha file and return a list of [(speaker, cleaned_text), ...].

    Only extracts transcript text from *SPEAKER: lines; skips all annotation lines.
    Multi-line continuations (lines starting with tab) are merged into the preceding line.
    """
    with open(cha_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    utterances: list[tuple[str, str]] = []
    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.rstrip("\n")

        # Speaker line: *CHI:\ttext text text .
        match = re.match(r"^\*(\w+):\t(.*)$", line)
        if match:
            # Save the previous utterance first
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    utterances.append((current_speaker, cleaned))
            current_speaker = match.group(1)
            current_text = match.group(2)
            continue

        # Continuation line (starts with tab, belongs to the same speaker utterance)
        if line.startswith("\t") and current_speaker:
            # Only append if it is a genuine continuation, not an annotation line
            # Annotation lines are usually %mor:\t, %gra:\t, etc., but continuations start directly with \t
            # Judgment: append only if the previous line was a speaker line continuation
            current_text += " " + line.strip()
            continue

        # Annotation lines (%mor, %gra, %com, etc.) → skip
        if re.match(r"^%\w+:", line):
            # Save the previous speaker utterance first
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    utterances.append((current_speaker, cleaned))
                current_speaker = None
                current_text = ""
            continue

        # Metadata lines (@Begin, @End, @Languages, @G:, etc.) → skip
        if line.startswith("@"):
            if current_speaker and current_text:
                cleaned = clean_utterance(current_text)
                if cleaned:
                    utterances.append((current_speaker, cleaned))
                current_speaker = None
                current_text = ""
            continue

    # Handle the last utterance
    if current_speaker and current_text:
        cleaned = clean_utterance(current_text)
        if cleaned:
            utterances.append((current_speaker, cleaned))

    return utterances


def process_all():
    """Process all CHA files and output cleaned text."""
    cha_files = []
    for group in GROUPS:
        group_dir = OCSC_DIR / group
        if group_dir.exists():
            cha_files.extend(sorted(group_dir.glob("*.cha")))

    print(f"Found {len(cha_files)} CHA files")

    total_utterances = 0
    for cha_path in tqdm(cha_files, desc="Parsing CHA files"):
        utterances = parse_cha(cha_path)
        total_utterances += len(utterances)

        # Write output to {id}_cleaned.txt in the same directory
        out_path = cha_path.with_name(cha_path.stem + "_cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for speaker, text in utterances:
                f.write(f"{speaker}\t{text}\n")

    print(f"Done! Extracted {total_utterances} valid transcriptions in total")


if __name__ == "__main__":
    process_all()
