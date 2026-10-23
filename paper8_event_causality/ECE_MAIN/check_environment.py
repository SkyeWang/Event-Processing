# 论文八：检查输入文件、依赖版本，以及可选的真实 CUDA 运算。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Check the prepared ECE runtime and, when requested, CUDA execution."""

import argparse
import pickle
from pathlib import Path

import torch
import transformers


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="sample700")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    model = ROOT / "data" / "bert-base-zh"
    subset = ROOT / "data" / "subsets" / args.subset
    required = [model / "config.json", model / "model.safetensors", model / "vocab.txt",
                subset / "raw_data.txt"]
    for split in ("train", "dev", "test"):
        required.extend([subset / "processed_data" / f"{split}.pickle",
                         subset / "split_data" / f"{split}_ids.txt"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))
    # 从准备好的 pickle 统计 train/dev/test 数量，确认使用了预期划分。
    counts = {}
    for split in ("train", "dev", "test"):
        with (subset / "processed_data" / f"{split}.pickle").open("rb") as file:
            counts[split] = len(pickle.load(file))
    print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("transformers:", transformers.__version__)
    print("subset:", subset, counts)
    print("CUDA available:", torch.cuda.is_available())
    if args.require_cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU unavailable; connect GPU and check driver/runtime")
        print("GPU:", torch.cuda.get_device_name(0),
              "compute capability:", torch.cuda.get_device_capability(0))
        a = torch.randn(64, 64, device="cuda")
        print("CUDA matmul:", (a @ a).shape)


if __name__ == "__main__":
    main()
