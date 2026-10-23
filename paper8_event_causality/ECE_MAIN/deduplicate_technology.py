# 论文八：先读 BERT、后读正则，以逐对 TF-IDF 余弦相似度 >= 0.40 去重。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Merge BERT and regex event-pair JSONL using the thesis section 5.3 rule.

The thesis describes pairwise ``TfidfVectorizer().fit_transform([text1,
text2])`` and cosine similarity >= 0.40, keeping the earlier pair. It does
not specify how a cause/effect pair is rendered as one text. This script uses
``cause + ' ' + effect`` and records every merge for review.

The cosine implementation below is algebraically equivalent to scikit-learn's
default TfidfVectorizer for *two* documents: lowercase word tokens of length
at least two, raw term frequencies, smoothed IDF, and L2 normalization.
An inverted index avoids comparing event pairs that have no shared token.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
UNIQUE_IDF = 1.0 + math.log(3.0 / 2.0)
UNIQUE_IDF_SQUARED = UNIQUE_IDF * UNIQUE_IDF
REQUIRED = ("text_id", "source", "text", "cause", "effect")


# 事件对序列化为 cause + 空格 + effect；这是论文未给出而补足的实现选择。
def tokens_for_pair(row: dict[str, Any]) -> Counter[str]:
    # This matches TfidfVectorizer()'s default lowercasing and token_pattern.
    pair_text = f"{row['cause']} {row['effect']}".lower()
    return Counter(TOKEN_RE.findall(pair_text))


# 只在两条文本上拟合：共享词 IDF=1，独有词 IDF=1+log(3/2)。
def pairwise_tfidf_cosine(
    left: Counter[str], left_sum_squares: int,
    right: Counter[str], right_sum_squares: int,
) -> float:
    """Cosine from TF-IDF fitted to precisely these two documents.

    Tokens in both documents have IDF 1; tokens in only one document have
    smoothed IDF 1 + log(3/2). This avoids refitting sklearn in every loop.
    """
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
        left_sum_squares, right_sum_squares = right_sum_squares, left_sum_squares
    dot = 0
    left_shared_squares = 0
    right_shared_squares = 0
    for token, left_count in left.items():
        right_count = right.get(token)
        if right_count:
            dot += left_count * right_count
            left_shared_squares += left_count * left_count
            right_shared_squares += right_count * right_count
    return cosine_from_overlap(dot, left_sum_squares, right_sum_squares,
                               left_shared_squares, right_shared_squares)


# 用交集词的点积和平方和恢复余弦；等价于两文档默认词级 TF-IDF。
def cosine_from_overlap(
    dot: int, left_sum_squares: int, right_sum_squares: int,
    left_shared_squares: int, right_shared_squares: int,
) -> float:
    if not dot:
        return 0.0
    left_norm_sq = (UNIQUE_IDF_SQUARED * left_sum_squares
                    - (UNIQUE_IDF_SQUARED - 1.0) * left_shared_squares)
    right_norm_sq = (UNIQUE_IDF_SQUARED * right_sum_squares
                     - (UNIQUE_IDF_SQUARED - 1.0) * right_shared_squares)
    return dot / math.sqrt(left_norm_sq * right_norm_sq)


def read_rows(path: Path, expected_source: str, limit: int | None):
    emitted = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            missing = [field for field in REQUIRED if field not in row]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {', '.join(missing)}")
            if row["source"] != expected_source:
                raise ValueError(
                    f"{path}:{line_number}: source must be {expected_source!r}, "
                    f"got {row['source']!r}"
                )
            for field in ("text", "cause", "effect"):
                if not isinstance(row[field], str):
                    raise ValueError(f"{path}:{line_number}: {field} must be a string")
            if not row["cause"].strip() or not row["effect"].strip():
                raise ValueError(f"{path}:{line_number}: empty cause or effect")
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


# 倒排索引缩小候选比较范围；达到阈值归并到最先保留项，并输出审计映射。
def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    if args.bert.resolve() == args.regex.resolve():
        raise SystemExit("--bert and --regex must be different files")

    retained: list[dict[str, Any]] = []
    duplicate_mapping: list[dict[str, Any]] = []
    retained_sum_squares: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    raw_exact: dict[str, int] = {}
    total_by_source: Counter[str] = Counter()
    duplicate_by_source: Counter[str] = Counter()
    cross_source_duplicates = 0
    input_index = 0

    # File order is deliberate: it defines "first appearing" when two
    # methods produce the same pair. Individual file order is preserved.
    for path, source in ((args.bert, "bert"), (args.regex, "regex")):
        for row in read_rows(path, source, args.limit_per_source):
            total_by_source[source] += 1
            pair_text = f"{row['cause']} {row['effect']}".lower()
            tokens = tokens_for_pair(row)
            sum_squares = sum(count * count for count in tokens.values())

            # TfidfVectorizer() cannot vectorize a pair with an empty
            # vocabulary. We retain such a pair unless its rendered text
            # matches a previous empty-vocabulary pair exactly.
            if not tokens:
                overlap = {}
                candidates = [raw_exact[pair_text]] if pair_text in raw_exact else []
            else:
                # Aggregate intersections while scanning the inverted index.
                # A full Cartesian product would require O(N^2) comparisons.
                overlap: dict[int, list[int]] = {}
                for token, count in tokens.items():
                    for index, first_count in postings.get(token, ()):
                        shared = overlap.setdefault(index, [0, 0, 0])
                        shared[0] += count * first_count
                        shared[1] += count * count
                        shared[2] += first_count * first_count
                candidates = sorted(overlap)

            representative_index = None
            similarity = 0.0
            for index in candidates:
                if not tokens:
                    score = 1.0
                else:
                    dot, current_shared_squares, first_shared_squares = overlap[index]
                    score = cosine_from_overlap(
                        dot, sum_squares, retained_sum_squares[index],
                        current_shared_squares, first_shared_squares,
                    )
                if score >= args.threshold:
                    representative_index = index
                    similarity = score
                    break

            if representative_index is None:
                representative_index = len(retained)
                retained.append(row)
                retained_sum_squares.append(sum_squares)
                for token, count in tokens.items():
                    postings[token].append((representative_index, count))
                if not tokens:
                    raw_exact.setdefault(pair_text, representative_index)
            else:
                first = retained[representative_index]
                duplicate_by_source[source] += 1
                if source != first["source"]:
                    cross_source_duplicates += 1
                duplicate_mapping.append({
                    "duplicate_input_index": input_index,
                    "duplicate_text_id": row["text_id"],
                    "duplicate_source": row["source"],
                    "duplicate_cause": row["cause"],
                    "duplicate_effect": row["effect"],
                    "representative_retained_index": representative_index,
                    "representative_text_id": first["text_id"],
                    "representative_source": first["source"],
                    "representative_cause": first["cause"],
                    "representative_effect": first["effect"],
                    "cosine_similarity": round(similarity, 8),
                })
            input_index += 1

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "retained.jsonl", retained)
    write_jsonl(args.output_dir / "duplicate_mapping.jsonl", duplicate_mapping)
    summary = {
        "method": "pairwise TfidfVectorizer() default word features and cosine",
        "threshold": args.threshold,
        "threshold_operator": ">=",
        "input_order": [str(args.bert), str(args.regex)],
        "pair_text": "cause + ' ' + effect",
        "first_occurrence": "first retained pair in input order",
        "input_pairs": input_index,
        "input_by_source": dict(total_by_source),
        "retained_pairs": len(retained),
        "duplicate_pairs": len(duplicate_mapping),
        "duplicate_by_source": dict(duplicate_by_source),
        "cross_source_duplicates": cross_source_duplicates,
        "limit_per_source": args.limit_per_source,
        "limitations": [
            "The paper does not specify exactly how each cause/effect pair becomes one TF-IDF text.",
            "Default bag-of-words similarity does not preserve cause/effect direction.",
            "BERT normalized event descriptions and regex source spans may not be directly comparable.",
            "Similarity removal does not verify causal correctness or topic relevance.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Read {input_index} pairs; retained {len(retained)}; "
          f"mapped {len(duplicate_mapping)} duplicates to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert", type=Path, required=True, help="BERT pairs JSONL")
    parser.add_argument("--regex", type=Path, required=True, help="regex pairs JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--limit-per-source", type=int,
                        help="read only the first N rows from each source for a CPU smoke run")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.limit_per_source is not None and args.limit_per_source < 1:
        parser.error("--limit-per-source must be positive")
    run(args)


if __name__ == "__main__":
    main()
