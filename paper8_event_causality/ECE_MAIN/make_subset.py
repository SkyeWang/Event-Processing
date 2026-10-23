# 论文八：从公开预处理数据的原划分中抽样，不重新标注或混合划分。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Sample the author's already processed ECE data without crossing splits."""

import argparse
import ast
import hashlib
import json
import pickle
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data"
SPLITS = ("train", "dev", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new directory name under data/subsets")
    parser.add_argument("--train", required=True, type=int)
    parser.add_argument("--dev", required=True, type=int)
    parser.add_argument("--test", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    if not args.name.isidentifier() or any(getattr(args, s) <= 0 for s in SPLITS):
        parser.error("name must be an identifier and each split size must be positive")
    destination = SOURCE / "subsets" / args.name
    if destination.exists():
        parser.error(f"destination exists: {destination}")

    # 固定随机种子；在各自划分内部抽样并保持原记录顺序。
    rng = random.Random(args.seed)
    chosen = {}
    manifest = {"source": "cuishiyao96/ECE@59f21d15fcaebc48d798b99090890de14c5af5bb",
                "seed": args.seed, "counts": {}, "source_sha256": {}}
    for split in SPLITS:
        path = SOURCE / "processed_data" / f"{split}.pickle"
        blob = path.read_bytes()
        records = pickle.loads(blob)
        ids = (SOURCE / "split_data" / f"{split}_ids.txt").read_text(encoding="utf-8").splitlines()
        record_ids = [str(record["text_id"]) for record in records]
        if len(records) != len(ids) or set(record_ids) != set(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Author split {split} does not align with processed records")
        count = getattr(args, split)
        if count > len(records):
            parser.error(f"{split} has only {len(records)} records")
        positions = set(rng.sample(range(len(records)), count))
        selected = [record for i, record in enumerate(records) if i in positions]
        chosen[split] = selected
        manifest["counts"][split] = count
        manifest["source_sha256"][split] = hashlib.sha256(blob).hexdigest()

    # 检查三套 ID 无交集，防止训练数据泄漏到验证集或测试集。
    selected_ids = {split: {str(record["text_id"]) for record in records}
                    for split, records in chosen.items()}
    if any(selected_ids[a] & selected_ids[b] for a, b in
           (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Overlapping IDs across original splits")

    raw_lines = {}
    wanted = set.union(*selected_ids.values())
    for line in (SOURCE / "raw_data.txt").read_text(encoding="utf-8").splitlines():
        item = ast.literal_eval(line)
        key = str(item["text_id"])
        if key in wanted:
            if key in raw_lines:
                raise ValueError(f"Duplicate raw ID: {key}")
            raw_lines[key] = line
    if set(raw_lines) != wanted:
        raise ValueError(f"Missing {len(wanted - set(raw_lines))} selected IDs from raw_data.txt")
    # The author's metric zips predictions and gold records, so order matters.
    for split, records in chosen.items():
        actual = [key for key in raw_lines if key in selected_ids[split]]
        expected = [str(record["text_id"]) for record in records]
        if actual != expected:
            raise ValueError(f"Raw and processed order differ in {split}")

    # 保存编码、ID、原文及来源摘要，支持追溯子集构建过程。
    (destination / "processed_data").mkdir(parents=True)
    (destination / "split_data").mkdir()
    for split, records in chosen.items():
        with (destination / "processed_data" / f"{split}.pickle").open("wb") as file:
            pickle.dump(records, file, protocol=pickle.HIGHEST_PROTOCOL)
        ids = [str(record["text_id"]) for record in records]
        (destination / "split_data" / f"{split}_ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8")
    (destination / "raw_data.txt").write_text(
        "\n".join(raw_lines[key] for key in raw_lines) + "\n", encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
    print(manifest["counts"])


if __name__ == "__main__":
    main()
