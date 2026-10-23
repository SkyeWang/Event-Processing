# 论文七：从原始句子构建保留来源行号的英文 JSONL 语料。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Convert every non-empty thesis attachment sentence to traceable JSONL."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "AMR" / "camr" / "camr" / "sentences.txt"
OUTPUT = ROOT / "data" / "original_sentences_full.jsonl"
INFO = ROOT / "data" / "original_sentences_full_info.json"


# 还原 HTML 字符、去掉开头异常问号并压缩空白；不删除重复文本。
def clean(text: str) -> str:
    text = html.unescape(text).replace("\\'", "'").strip()
    text = re.sub(r"^[?\s]+", "", text)
    return re.sub(r"\s+", " ", text)


def main() -> None:
    raw_lines = SOURCE.read_text(encoding="utf-8", errors="ignore").splitlines()
    records = []
    for line_number, raw in enumerate(raw_lines, 1):
        text = clean(raw)
        if text:
            records.append(
                {
                    "id": f"attachment-{line_number:05d}",
                    "source_line": line_number,
                    "text": text,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 忽略大小写的唯一文本数只作统计；实际输出仍保留全部 15,633 条。
    unique = len({record["text"].casefold() for record in records})
    info = {
        "name": "Full English sentence corpus from thesis attachment",
        "source_file": str(SOURCE.resolve()),
        "records": len(records),
        "unique_casefold": unique,
        "duplicate_records": len(records) - unique,
        "annotation": "unlabeled; suitable for prediction, not supervised evaluation",
        "provenance_limit": "the attachment does not include per-sentence article URL/title/date metadata",
    }
    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
