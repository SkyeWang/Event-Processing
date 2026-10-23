# 论文八：按照论文表 2.4 的英文词表及方向抽取因果候选。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Extract English causal pairs using only the patterns in thesis Table 2.4.

Input:  附录/Code/LLM/data/original_sentences_full.jsonl
Output: one JSON object per extracted pair, with the original text and spans.

The table is on printed pages 11–12 (PDF pages 18–19). Its ``result in``
entry reads ``果 result in 因``; this script preserves that direction literally.
The old ``../causality_extract_en.py`` supplied the nine-rule design idea, but
cannot be imported: it runs a file-dependent test on import, adds expressions
outside Table 2.4, requires NLTK POS data, and calls ruler4 for ruler8.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "paper7_event_attributes" / "LLM" / "data" / "original_sentences_full.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "res" / "tech_regex_predictions.jsonl"
FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class CueRule:
    name: str
    pattern: re.Pattern[str]
    direction: str  # "forward": cause cue effect; "reverse": effect cue cause
    paper_form: str
    exclude_next: tuple[str, ...] = ()


def cue(name: str, expression: str, direction: str, paper_form: str,
        exclude_next: tuple[str, ...] = ()) -> CueRule:
    return CueRule(name, re.compile(r"\b(?:" + expression + r")\b", FLAGS),
                   direction, paper_form, exclude_next)


# Only the lexical items and arrangements in Table 2.4. Verb tense/number
# forms are spelling variants of the listed verb, not additional cue words.
# forward 表示“因 cue 果”，reverse 表示“果 cue 因”；保留论文 result in 方向。
CUE_RULES = (
    cue("cause_for", r"caus(?:e|es|ed|ing)\s+for", "forward", "因 cause for 果"),
    cue("cause", r"cause|causes|caused|causing", "forward", "因 cause 果",
        ("of", "for", "by")),
    cue("arise_from", r"(?:arise|arises|arose|arisen|arising)\s+from",
        "reverse", "果 arise from 因"),
    cue("arise_of", r"(?:arise|arises|arose|arisen|arising)\s+of",
        "reverse", "果 arise of 因"),
    cue("give_rise_to", r"(?:give|gives|gave|given|giving)\s+rise\s+to",
        "forward", "因 give rise to 果"),
    cue("trigger_of", r"(?:trigger|triggers|triggered|triggering)\s+of",
        "forward", "因 trigger of 果"),
    cue("triggered_by", r"(?:(?:is|are|was|were|be|been|being)\s+)?triggered\s+by",
        "reverse", "果 triggerd by 因"),
    cue("trigger", r"trigger|triggers|triggered|triggering", "forward",
        "因 trigger 果", ("of", "by")),
    cue("increase", r"increase|increases|increased|increasing", "forward",
        "因 increase 果", ("by",)),
    cue("effect_on", r"effect\s+on", "forward", "因 effect on 果"),
    cue("affected_by", r"(?:(?:is|are|was|were|be|been|being)\s+)?affected\s+by",
        "reverse", "果 affected by 因"),
    cue("affect", r"affect|affects|affected|affecting", "forward",
        "因 affect 果", ("by",)),
    cue("induce_to", r"(?:induce|induces|induced|inducing)\s+to", "forward",
        "因 induce to 果"),
    cue("induced_by", r"(?:(?:is|are|was|were|be|been|being)\s+)?induced\s+by",
        "reverse", "果 induced by 因"),
    cue("induce", r"induce|induces|induced|inducing", "forward",
        "因 induce 果", ("to", "by")),
    cue("derived_from", r"(?:(?:is|are|was|were|be|been|being)\s+)?derived\s+from",
        "reverse", "果 derived from 因"),
    cue("reveal", r"reveal|reveals|revealed|revealing", "forward",
        "因 reveal 果", ("by",)),
    cue("because_of", r"because\s+of", "reverse", "果 because of 因"),
    cue("because", r"because", "reverse", "果 because 因", ("of",)),
    cue("so", r"so", "forward", "因 so 果", ("as", "that")),
    cue("hence", r"hence", "forward", "因 hence 果"),
    cue("therefore", r"therefore", "forward", "因 therefore 果"),
    cue("thus", r"thus", "forward", "因 thus 果"),
    cue("thereby", r"thereby", "forward", "因 thereby 果"),
    cue("since", r"since", "reverse", "果 since 因"),
    cue("for", r"for", "reverse", "果 for 因"),
    cue("through", r"through", "reverse", "果 through 因"),
    cue("after", r"after", "reverse", "果 after 因"),
    cue("accordingly", r"accordingly", "forward", "因 accordingly 果"),
    cue("consequently", r"consequently", "forward", "因 consequently 果"),
    cue("result_in", r"(?:result|results|resulted|resulting)\s+in", "reverse",
        "果 result in 因"),  # Table 2.4's direction, despite ordinary usage.
    cue("result_from", r"(?:result|results|resulted|resulting)\s+from",
        "reverse", "果 result from 因"),
    cue("lead_to", r"(?:lead|leads|led|leading)\s+to", "forward",
        "因 lead to 果"),
    cue("stem_from", r"(?:stem|stems|stemmed|stemming)\s+from", "reverse",
        "果 stem from 因"),
)

# The three paired forms have two separated markers, so their spans must be
# extracted together. The only verb here is the table's "is" (tense variants).
COPULA = r"(?:is|was|are|were)"
PAIRED_RULES = (
    ("cause_of_is", re.compile(
        rf"^\s*(?:the\s+)?cause\s+of\s+(?P<effect>.+?)\s+{COPULA}\s+(?P<cause>.+?)\s*$",
        FLAGS), "cause of 果 is 因"),
    ("effect_of_is", re.compile(
        rf"^\s*(?:the\s+)?effect\s+of\s+(?P<cause>.+?)\s+{COPULA}\s+(?P<effect>.+?)\s*$",
        FLAGS), "effect of 因 is 果"),
    ("result_of_is", re.compile(
        rf"^\s*the\s+result\s+of\s+(?P<cause>.+?)\s+{COPULA}\s+(?P<effect>.+?)\s*$",
        FLAGS), "The result of 因 is 果"),
)
FRONTED_RULE = re.compile(
    r"^\s*(?P<tag>since|for|through|after)\s+(?P<cause>[^,]+),\s*(?P<effect>.+?)\s*$",
    FLAGS,
)
SEGMENT_BREAK = re.compile(
    r"(?<=;)\s*|(?<=[.!?])(?:[”’\"']\s*)?\s*(?=[A-Z\"'“])"
)
EDGE_CHARS = " \t\r\n,;:.!?—–-\"'“”‘’()[]"


# 先按片段边界切分，再匹配连接词，字符偏移始终相对于原句。
def segments(text: str) -> Iterator[tuple[str, int]]:
    """Yield source-preserving sentence-sized spans and their absolute starts."""
    start = 0
    for match in SEGMENT_BREAK.finditer(text):
        # Avoid splitting abbreviations such as U.S. on their initial dots.
        if (match.start() >= 2 and text[match.start() - 1] == "."
                and text[match.start() - 2].isupper()):
            continue
        if text[start:match.start()].strip():
            yield text[start:match.start()], start
        start = match.end()
    if text[start:].strip():
        yield text[start:], start


def trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start] in EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in EDGE_CHARS:
        end -= 1
    if start >= end or not any(ch.isalnum() for ch in text[start:end]):
        return None
    return start, end


def pair(text: str, cause: tuple[int, int] | None, effect: tuple[int, int] | None,
         rule: str, tag: str, paper_form: str) -> dict | None:
    if cause is None or effect is None:
        return None
    return {
        "cause": text[cause[0]:cause[1]],
        "effect": text[effect[0]:effect[1]],
        "cause_span": list(cause),
        "effect_span": list(effect),
        "rule": rule,
        "tag": tag,
        "paper_form": paper_form,
    }


# 优先处理成对结构和句首连接词，再处理普通单连接词结构。
def extract_pairs(text: str) -> list[dict]:
    """Return all Table 2.4 matches; no semantic/topic filtering is inferred."""
    found: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(candidate: dict | None) -> None:
        if candidate is None:
            return
        key = (candidate["cause"], candidate["effect"], candidate["rule"])
        if key not in seen:
            found.append(candidate)
            seen.add(key)

    for segment, base in segments(text):
        occupied: list[tuple[int, int]] = []
        for name, pattern, paper_form in PAIRED_RULES:
            match = pattern.match(segment)
            if match is None:
                continue
            cause = trimmed_span(text, base + match.start("cause"), base + match.end("cause"))
            effect = trimmed_span(text, base + match.start("effect"), base + match.end("effect"))
            add(pair(text, cause, effect, name, name, paper_form))
            occupied.append(match.span())

        fronted = FRONTED_RULE.match(segment)
        if fronted is not None:
            tag = fronted.group("tag")
            cause = trimmed_span(text, base + fronted.start("cause"), base + fronted.end("cause"))
            effect = trimmed_span(text, base + fronted.start("effect"), base + fronted.end("effect"))
            add(pair(text, cause, effect, f"{tag.lower()}_fronted", tag,
                     f"{tag.title()} 因，果"))
            occupied.append(fronted.span("tag"))

        for rule in CUE_RULES:
            for match in rule.pattern.finditer(segment):
                if any(match.start() < end and start < match.end()
                       for start, end in occupied):
                    continue
                if rule.exclude_next:
                    following = segment[match.end():]
                    if re.match(r"\s+(?:" + "|".join(rule.exclude_next) + r")\b",
                                following, FLAGS):
                        continue
                left = trimmed_span(text, base, base + match.start())
                right = trimmed_span(text, base + match.end(), base + len(segment))
                if rule.direction == "forward":
                    cause, effect = left, right
                else:
                    cause, effect = right, left
                add(pair(text, cause, effect, rule.name, match.group(), rule.paper_form))

    return found


# 每个候选单独输出，保留原文 ID、来源行号、规则名称和原因/结果位置。
def run(input_path: Path, output_path: Path, limit: int | None) -> tuple[int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = predictions = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8") as target:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Line {line_no}: 'text' must be a string")
            records += 1
            text_id = str(record.get("id", f"line-{line_no}"))
            for candidate in extract_pairs(text):
                result = {
                    "text_id": text_id,
                    "source": "regex",
                    "text": text,
                }
                if "source_line" in record:
                    result["source_line"] = record["source_line"]
                result.update(candidate)
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                predictions += 1
            if limit is not None and records >= limit:
                break
    return records, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Process only the first N records (smoke test)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must differ")
    records, predictions = run(args.input, args.output, args.limit)
    print(f"Processed {records} texts; wrote {predictions} regex pairs to {args.output}")


if __name__ == "__main__":
    main()
