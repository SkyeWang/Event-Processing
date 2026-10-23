"""从 HuggingFace 镜像下载 ZINC 子集，存成 JSONL。

官方 PyG 的 ZINC 下载地址指向 Dropbox，本机不可达；``graphs-datasets/ZINC``
镜像提供同样的字段（node_feat、edge_index、edge_attr、y、num_nodes），
这里按行流式读取，只保留前 N 条，避免下载完整的 22 万训练集。

输出：
  sagnn/ZINC_hf/train.jsonl  （10000 条）
  sagnn/ZINC_hf/val.jsonl    （1000 条）
  sagnn/ZINC_hf/test.jsonl   （1000 条）

注意：镜像的前 10k 条与官方 PyG ``ZINC(subset=True)`` 的 10k 索引不同，
因此结果与论文的 0.072 只能做到接近，不能逐位对齐。
"""

import os
import urllib.request

BASE = "https://hf-mirror.com/datasets/graphs-datasets/ZINC/resolve/main"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ZINC_hf")
# 与论文/官方配置一致的子集规模：10k train / 1k val / 1k test
LIMITS = {"train": 10000, "val": 1000, "test": 1000}


def download(split, limit):
    """流式下载某个 split 的前 ``limit`` 行。

    Args:
        split: ``train`` / ``val`` / ``test``。
        limit: 最多保留多少条；0 表示不限制。
    """
    url = f"{BASE}/{split}.jsonl"
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{split}.jsonl")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    count = 0
    # 逐行读取，达到 limit 就关闭连接，不把整个文件读进内存
    with urllib.request.urlopen(req, timeout=120) as response, open(path, "w", encoding="utf-8") as fh:
        for line in response:
            if count >= limit:
                break
            fh.write(line.decode("utf-8"))
            count += 1
    print(split, count, path)


if __name__ == "__main__":
    for split, limit in LIMITS.items():
        download(split, limit)
