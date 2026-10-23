# 论文八：把英文 JSONL 编为与训练一致的字符输入，输出句级与事件对级结果。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Run the trained DualCor/BERT checkpoint on unlabeled thesis sentences.

The public ECE training pickle uses *character* tokens for the sentence, a
fixed 39-event-type prefix, and a 151-position sentence tail. This script
recreates that input exactly. English is outside the training language, so
the predictions are exploratory and must not be reported as an English F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model import Model  # noqa: E402
from utils import etype_map, max_seq_len  # noqa: E402


# 按固定类型顺序构造前缀及 unused 类型标记，不能改成任意标签顺序。
def event_type_prefix(tokenizer: Any) -> tuple[list[int], list[int]]:
    tokens = ["[CLS]"]
    marker_positions: list[int] = []
    for index, event_type in enumerate(etype_map, 1):
        tokens.extend(tokenizer.tokenize(event_type))
        tokens.append(f"[unused{index}]")
        marker_positions.append(len(tokens) - 1)
    tokens.append("[SEP]")
    if len(tokens) + max_seq_len > tokenizer.model_max_length:
        raise ValueError("Event-type prefix and sentence tail exceed BERT's limit")
    return tokenizer.convert_tokens_to_ids(tokens), marker_positions


# 最多取 150 个原始字符并小写；逐字符编码，保留与网格对应的位置。
def encode_sentence(
    text: str, tokenizer: Any, prefix_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    # Matches the released processed_data pickle: lowercased character
    # tokens, followed by [SEP], then right padding to 151 positions.
    clipped = text[: max_seq_len - 1]
    lowered = clipped.lower()
    if len(lowered) != len(clipped):
        raise ValueError("Lowercasing changed character positions; cannot decode spans safely")
    sentence_tokens = list(lowered) + ["[SEP]"]
    sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    padding = max_seq_len - len(sentence_ids)
    full_ids = prefix_ids + sentence_ids + [tokenizer.pad_token_id] * padding
    mask = [1] * (len(prefix_ids) + len(sentence_ids)) + [0] * padding
    segments = [0] * len(prefix_ids) + [1] * max_seq_len
    return (
        torch.tensor([full_ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
        torch.tensor([segments], dtype=torch.long),
        clipped,
    )


# 将类型和论元投影为去重文本；它不是原句中连续的完整事件跨度。
def describe_event(pair: dict[str, str], side: str) -> str:
    # A transparent projection of structured ECE output for the paper's
    # text-pair similarity step. It is not an original sentence event span.
    parts = [pair[f"{side}_type"]]
    for role in ("product", "region", "industry"):
        value = pair[f"{side}_{role}"]
        if value:
            parts.append(value.replace(",", " "))
    return " ".join(parts)


def iter_input(path: Path, limit: int | None):
    seen_ids: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            text_id = row.get("id", row.get("text_id"))
            text = row.get("text")
            if not isinstance(text_id, (str, int)) or not str(text_id):
                raise ValueError(f"{path}:{line_number}: missing id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: missing text")
            text_id = str(text_id)
            if text_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {text_id}")
            seen_ids.add(text_id)
            yield text_id, text, row
            count += 1
            if limit is not None and count >= limit:
                break


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; check GPU and pytorch_gpu environment")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# 加载指定权重，记录每句是否截断；无预测的句子也保留在句级输出。
def run(args: argparse.Namespace) -> None:
    for path in (args.input, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.bert_path.is_dir():
        raise FileNotFoundError(args.bert_path)
    paths = (args.output, args.pairs_output, args.summary)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Output paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model_args = SimpleNamespace(
        hidden_size=768, bert_path=str(args.bert_path), dropout=0.0, thresh=args.threshold
    )
    model = Model(model_args)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    tokenizer = model.tokenizer
    prefix_ids, marker_positions = event_type_prefix(tokenizer)
    marker_tensor = torch.tensor([marker_positions], dtype=torch.long, device=device)

    sentences = 0
    truncated = 0
    pair_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as sentence_file, \
            args.pairs_output.open("w", encoding="utf-8", newline="\n") as pair_file, \
            torch.inference_mode():
        for text_id, text, source_row in iter_input(args.input, args.limit):
            input_ids, mask, segments, clipped = encode_sentence(text, tokenizer, prefix_ids)
            if len(clipped) < len(text):
                truncated += 1
            prediction = model.inference(
                [text_id], input_ids.to(device), segments.to(device),
                mask.to(device), marker_tensor, raw_text=clipped,
            )
            pairs = prediction["result"]
            sentence_row = {
                "text_id": text_id,
                "source": "bert",
                "text": text,
                "source_line": source_row.get("source_line"),
                "input_characters": len(clipped),
                "truncated": len(clipped) < len(text),
                "result": pairs,
            }
            sentence_file.write(json.dumps(sentence_row, ensure_ascii=False) + "\n")
            for pair in pairs:
                pair_row = {
                    "text_id": text_id,
                    "source": "bert",
                    "text": text,
                    "cause": describe_event(pair, "reason"),
                    "effect": describe_event(pair, "result"),
                    "source_line": source_row.get("source_line"),
                    "truncated": len(clipped) < len(text),
                    **pair,
                }
                pair_file.write(json.dumps(pair_row, ensure_ascii=False) + "\n")
                pair_count += 1
            sentences += 1
            if args.progress_every and sentences % args.progress_every == 0:
                print(f"Processed {sentences} sentences; {pair_count} pairs", flush=True)

    summary = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "bert_path": str(args.bert_path.resolve()),
        "device": str(device),
        "threshold": args.threshold,
        "sentences": sentences,
        "truncated_at_150_characters": truncated,
        "predicted_pairs": pair_count,
        "sentence_predictions": str(args.output.resolve()),
        "pair_predictions": str(args.pairs_output.resolve()),
        "note": "Unlabeled English inference; no English-domain F1 is computed.",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {sentences} sentences, {pair_count} pairs, {truncated} truncated; device={device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="unlabeled JSONL with id and text")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pickle state dict")
    parser.add_argument("--bert-path", type=Path, default=ROOT / "data" / "bert-base-zh")
    parser.add_argument("--output", type=Path, required=True, help="one structured prediction per sentence")
    parser.add_argument("--pairs-output", type=Path, required=True, help="one pair per line for deduplication")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="process only the first N sentences")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
